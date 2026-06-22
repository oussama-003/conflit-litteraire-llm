"""
Pipeline v2 : 3 fichiers Excel → JSONL pour fine-tuning Mistral
- Paragraphes analytiques FLUIDES (pas de markdown, pas de listes)
- Verdict explicite en dernière phrase (pour que les métriques fonctionnent)
- Compatible avec finetune_with_metrics.py

INSTALLATION :
    pip install pandas openpyxl anthropic

UTILISATION :
    export ANTHROPIC_API_KEY=sk-ant-...   (Mac/Linux)
    set ANTHROPIC_API_KEY=sk-ant-...      (Windows)
    python generate_jsonl_local_v2.py
"""

import pandas as pd
import json
import time
import re
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# ① CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY         = ""  # ou variable d'environnement ANTHROPIC_API_KEY
OUTPUT_PATH     = "dataset_conflit_final_v2.jsonl"
CHECKPOINT_PATH = "checkpoint_v2.jsonl"
DELAY           = 0.8

FILES = [
    {
        "path": "Dataset_n_1_PE26_Compilation_Finale_-_VERIFIE_par_HERR_le_15-5-26.xlsx",
        "exclude_ids": {21},
        "conflit_col": "Conflit(OUI/NON)",
    },
    {
        "path": "Dataset_n_2_PE26_Compilation_Finale_-_VERIFIE_par_HERR_le_17-5-26.xlsx",
        "exclude_ids": {5, 47},
        "conflit_col": "Conflit\n(OUI/NON)",
    },
    {
        "path": "Dataset_n_3__PE26_Extraits_Philippe_-_VERIFIE_par_HERR_le_18-5-26.xlsx",
        "exclude_ids": set(),
        "conflit_col": "Conflit (OUI/NON)",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ② PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_FINETUNE = """Tu es un expert en analyse littéraire spécialisé dans la détection et l'analyse des conflits dans les textes narratifs.

Tu utilises la définition suivante du conflit :
Conflit = « Affrontement (1) dynamique (2) entre forces antagonistes (3), avec volonté d'éviction (4) ou opposition profonde (5), et avec un enjeu sérieux (6) — c'est-à-dire que les forces antagonistes engagent des valeurs, des intérêts ou des principes fondamentaux (identité, survie, morale, pouvoir, etc.) dont la résolution a des conséquences importantes ou structurantes pour au moins l'un des actants. »

Formule logique : (1) ET (2) ET (3) ET [(4) OU (5)] ET (6)

Lorsqu'on te soumet un extrait littéraire, tu analyses chaque critère et fournis une évaluation rédigée, fluide et argumentée — comme le ferait un expert humain. Tu conclus toujours par une phrase de verdict explicite."""

USER_TEMPLATE = """Analyse le passage littéraire suivant et détermine s'il contient un conflit selon la définition analytique. Identifie les forces en présence, vérifie chaque critère de la définition, et rédige une analyse en paragraphe fluide avec un verdict final explicite.

Extrait :
{texte}"""

# Prompt envoyé à Claude pour générer chaque paragraphe
GENERATOR_SYSTEM = """Tu es un expert en stylistique et en analyse littéraire. Tu génères des paragraphes analytiques fluides pour un dataset de fine-tuning.

Règles ABSOLUES :
- Rédige en français soutenu, en prose continue (aucune liste, aucun titre, aucun tiret, aucun markdown)
- 5 à 8 phrases maximum
- La DERNIÈRE phrase doit obligatoirement être l'une de ces trois formes exactes selon le verdict :
    * "En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI."
    * "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON."
    * "En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE."
- Ne jamais utiliser **gras**, puces, tirets ou numéros"""

# ─────────────────────────────────────────────────────────────────────────────
# ③ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def clean(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s else None


def normalize_columns(df):
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
    return df


def detect_verdict(row, conflit_col):
    detail     = clean(row.get("Détail")) or ""
    validation = clean(row.get("10. Validation globale")) or ""
    conflit    = str(row.get(conflit_col, "")).strip().upper()

    if re.search(r'PI[EÈ]GE', detail, re.IGNORECASE) or \
       re.search(r'PI[EÈ]GE', validation, re.IGNORECASE):
        return "PIÈGE"
    if conflit == "OUI":
        return "OUI"
    return "NON"


def build_structured_data(row, conflit_col):
    fields = {
        "verdict":             detect_verdict(row, conflit_col),
        "forces_antagonistes": clean(row.get("1. Forces antagonistes")),
        "nature_conflit":      clean(row.get("2. Nature du conflit")),
        "affrontement":        clean(row.get("3.1 Affrontement")),
        "dynamique":           clean(row.get("3.2 Dynamique")),
        "forces_critere":      clean(row.get("3.3 Forces antagonistes (critère)")),
        "volonte_eviction":    clean(row.get("3.4 Volonté d'éviction")),
        "opposition_profonde": clean(row.get("3.5 Opposition profonde")),
        "structure_dynamique": clean(
            row.get("4. Structure dynamique (séquence)") or
            row.get("4. Structure dynamique(séquence)")
        ),
        "modalite_intensite":  clean(row.get("5. Modalité de manifestation + Intensité")),
        "degre":               clean(row.get("6. Degré (explicite/implicite)")),
        "orientation":         clean(row.get("7. Orientation (offensif/défensif)")),
        "issue":               clean(row.get("8. Issue")),
        "synthese":            clean(row.get("9. Synthèse typologique")),
        "validation":          clean(row.get("10. Validation globale")),
        "detail":              clean(row.get("Détail")),
    }
    return {k: v for k, v in fields.items() if v is not None}


def build_generator_prompt(structured_data):
    verdict   = structured_data.get("verdict", "OUI")
    data_str  = json.dumps(structured_data, ensure_ascii=False, indent=2)

    verdict_sentence = {
        "OUI":   "En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI.",
        "NON":   "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON.",
        "PIÈGE": "En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE.",
    }.get(verdict, "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON.")

    return f"""Voici l'analyse structurée d'un extrait littéraire :

{data_str}

Rédige un paragraphe analytique fluide (5 à 8 phrases) en prose continue, sans markdown ni liste.
Ta dernière phrase doit être exactement : "{verdict_sentence}" """

# ─────────────────────────────────────────────────────────────────────────────
# ④ API ANTHROPIC
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key():
    key = API_KEY.strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("ERREUR : clé API Anthropic introuvable.")
        print("  → export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return key


def call_anthropic(prompt, api_key, retries=4):
    try:
        import anthropic as _anthropic
    except ImportError:
        print("ERREUR : pip install anthropic")
        sys.exit(1)

    client = _anthropic.Anthropic(api_key=api_key)
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=GENERATOR_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["overloaded", "529", "rate"]):
                wait = 20 * (attempt + 1)
                print(f"\n    ⚠ API surchargée, attente {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Échec API après retries")


def validate_paragraph(text, verdict):
    """
    Vérifie que le paragraphe généré contient bien le verdict explicite.
    Si non, ajoute la phrase de verdict à la fin.
    """
    verdict_phrases = {
        "OUI":   "En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI.",
        "NON":   "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON.",
        "PIÈGE": "En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE.",
    }
    expected = verdict_phrases.get(verdict, verdict_phrases["NON"])

    # Check if verdict keyword is present
    text_upper = text.upper()
    has_verdict = (
        (verdict == "OUI"   and "VERDICT OUI" in text_upper) or
        (verdict == "NON"   and "VERDICT NON" in text_upper) or
        (verdict == "PIÈGE" and ("VERDICT PIÈGE" in text_upper or "VERDICT PIEGE" in text_upper))
    )

    if not has_verdict:
        # Force append the verdict sentence
        text = text.rstrip() + " " + expected

    return text

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ CHARGEMENT DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_file(file_cfg):
    path = file_cfg["path"]
    if not os.path.exists(path):
        print(f"\nERREUR : fichier introuvable → {path}")
        sys.exit(1)

    df = pd.read_excel(path)
    df = normalize_columns(df)

    conflit_col_norm = re.sub(r'\s+', ' ', file_cfg["conflit_col"]).strip()
    if conflit_col_norm not in df.columns:
        candidates = [c for c in df.columns if 'Conflit' in c and 'NON' in c]
        if candidates:
            conflit_col_norm = candidates[0]
        else:
            raise ValueError(f"Colonne Conflit introuvable dans {path}")

    if "ID" in df.columns and file_cfg["exclude_ids"]:
        before = len(df)
        df = df[~df["ID"].isin(file_cfg["exclude_ids"])]
        excluded = before - len(df)
        if excluded:
            print(f"  → {excluded} enregistrement(s) exclus (IDs : {file_cfg['exclude_ids']})")

    return df, conflit_col_norm

# ─────────────────────────────────────────────────────────────────────────────
# ⑥ CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint():
    entries = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if entries:
            print(f"  ♻  Checkpoint : {len(entries)} entrées déjà générées — reprise.")
    return entries


def append_checkpoint(entry):
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# ⑦ PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def process_all():
    api_key = get_api_key()

    total_rows = 0
    for fc in FILES:
        if os.path.exists(fc["path"]):
            df_tmp = pd.read_excel(fc["path"])
            total_rows += len(df_tmp) - len(fc["exclude_ids"])

    entries         = load_checkpoint()
    already_done    = len(entries)
    total_processed = already_done
    total_skipped   = 0
    verdict_counts  = {"OUI": 0, "NON": 0, "PIÈGE": 0}
    global_idx      = 0

    for file_cfg in FILES:
        fname = os.path.basename(file_cfg["path"])
        print(f"\n{'='*60}")
        print(f"Fichier : {fname}")

        df, conflit_col = load_dataset_file(file_cfg)
        print(f"  Lignes à traiter : {len(df)}")

        for _, row in df.iterrows():
            global_idx += 1
            progress = f"[{global_idx:3d}/{total_rows}]"

            texte = clean(row.get("Texte de l'extrait"))
            if not texte:
                total_skipped += 1
                print(f"  {progress} SKIP — texte vide")
                continue

            if global_idx <= already_done:
                verdict = detect_verdict(row, conflit_col)
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                print(f"  {progress} ✓ (checkpoint)")
                continue

            verdict = detect_verdict(row, conflit_col)
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            print(f"  {progress} {verdict:<8} → génération...", end=" ", flush=True)

            try:
                structured  = build_structured_data(row, conflit_col)
                prompt      = build_generator_prompt(structured)
                paragraph   = call_anthropic(prompt, api_key)
                paragraph   = validate_paragraph(paragraph, verdict)  # ← garantit le verdict

                entry = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT_FINETUNE},
                        {"role": "user",      "content": USER_TEMPLATE.format(texte=texte)},
                        {"role": "assistant", "content": paragraph},
                    ]
                }
                entries.append(entry)
                append_checkpoint(entry)
                total_processed += 1
                print("✓")
                time.sleep(DELAY)

            except KeyboardInterrupt:
                print(f"\n\nInterruption à {global_idx}. Checkpoint sauvegardé.")
                sys.exit(0)
            except Exception as e:
                print(f"ERREUR : {e}")
                total_skipped += 1

    # Fichier final
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print(f"\n{'='*60}")
    print(f"TERMINÉ — {total_processed} entrées écrites, {total_skipped} ignorées")
    print(f"Distribution : {verdict_counts}")
    print(f"Fichier : {OUTPUT_PATH}")

    # Vérification finale : 0 verdicts manquants
    print("\nVérification des verdicts...")
    missing = 0
    with open(OUTPUT_PATH) as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            assistant = ex["messages"][-1]["content"].upper()
            if not any(v in assistant for v in ["VERDICT OUI", "VERDICT NON", "VERDICT PIÈGE", "VERDICT PIEGE"]):
                missing += 1
                print(f"  ⚠ Ligne {i} — verdict manquant")
    if missing == 0:
        print(f"  ✅ Tous les verdicts sont présents.")
    else:
        print(f"  ⚠ {missing} verdicts manquants.")


if __name__ == "__main__":
    process_all()
