"""
generate_synthetic.py — Générateur d'exemples synthétiques pour fine-tuning Mistral
Conflit littéraire — texte inventé + analyse en paragraphe fluide + verdict explicite

INSTALLATION :
    pip install anthropic

UTILISATION :
    export ANTHROPIC_API_KEY=sk-ant-...

    # Générer 200 exemples (équilibrés automatiquement)
    python generate_synthetic.py --n 200

    # Contrôler la distribution manuellement
    python generate_synthetic.py --oui 100 --non 80 --piege 120

    # Reprendre après interruption (checkpoint automatique)
    python generate_synthetic.py --n 200

    # Choisir le fichier de sortie
    python generate_synthetic.py --n 200 --output mon_dataset.jsonl

COÛT ESTIMÉ (Claude Sonnet 4) :
    100 exemples → ~$0.90
    200 exemples → ~$1.80
    500 exemples → ~$4.50
    1000 exemples → ~$9.00
"""

import argparse
import json
import os
import re
import sys
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# ① CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

API_KEY        = ""   # laisser vide → lit ANTHROPIC_API_KEY depuis l'environnement
MODEL          = "claude-sonnet-4-20250514"
DELAY          = 0.6  # secondes entre appels API

# Phrases de verdict explicites — alignées avec eval_model_v2.py
VERDICT_SENTENCES = {
    "OUI":   "En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI.",
    "NON":   "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON.",
    "PIÈGE": "En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE.",
}

# ─────────────────────────────────────────────────────────────────────────────
# ② DONNÉES DE DIVERSITÉ
# ─────────────────────────────────────────────────────────────────────────────

AUTHORS = [
    {"name": "Émile Zola",                      "style": "naturaliste, précis, social, descriptions physiques intenses"},
    {"name": "Victor Hugo",                      "style": "romantique, lyrique, grandiloquent, oppositions morales tranchées"},
    {"name": "Gustave Flaubert",                 "style": "réaliste, style indirect libre, ironie froide, précision clinique"},
    {"name": "Honoré de Balzac",                 "style": "réaliste, minutieux, social, ambitions et passions bourgeoises"},
    {"name": "Stendhal",                         "style": "psychologique, lucide, énergie et calcul, analyse des passions"},
    {"name": "Guy de Maupassant",                "style": "nouvelliste, concis, cruel, chute brutale"},
    {"name": "Alexandre Dumas",                  "style": "romanesque, action rythmée, dialogues vifs, duels et complots"},
    {"name": "Prosper Mérimée",                  "style": "sobre, sec, violence contenue, exotisme et passion"},
    {"name": "Jean Racine",                      "style": "tragique, alexandrins, passions fatales, fatalité et culpabilité"},
    {"name": "Molière",                          "style": "comédie, ironie sociale, conflits de caractère, dialogue piquant"},
    {"name": "Pierre Corneille",                 "style": "tragédie héroïque, devoir vs passion, noblesse et honneur"},
    {"name": "Albert Camus",                     "style": "absurde, sobre, distance émotionnelle, soleil et violence"},
    {"name": "Jean-Paul Sartre",                 "style": "existentialiste, huis clos, liberté et mauvaise foi"},
    {"name": "Marguerite Yourcenar",             "style": "historique, stoïque, intériorité profonde, latinité"},
    {"name": "Simone de Beauvoir",               "style": "analytique, féministe, relations de pouvoir subtiles"},
    {"name": "un auteur fictif du XIXe siècle",  "style": "romantique tardif, provincial, conflits bourgeois"},
    {"name": "un auteur fictif contemporain",    "style": "contemporain, urbain, dialogue rapide, ellipses"},
    {"name": "un auteur fictif médiéval stylisé","style": "épique, anachronique maîtrisé, honneur et trahison"},
]

CONFLICT_TYPES_OUI = [
    {"type": "interpersonnel",       "sous_type": "rivalité amoureuse",               "intensite": "forte"},
    {"type": "interpersonnel",       "sous_type": "conflit d'autorité familiale",      "intensite": "moyenne"},
    {"type": "interpersonnel",       "sous_type": "trahison et vengeance",             "intensite": "très forte"},
    {"type": "interpersonnel",       "sous_type": "jalousie et possession",            "intensite": "forte"},
    {"type": "social",               "sous_type": "lutte des classes",                 "intensite": "forte"},
    {"type": "social",               "sous_type": "conflit d'honneur public",          "intensite": "très forte"},
    {"type": "politique",            "sous_type": "complot et pouvoir",                "intensite": "très forte"},
    {"type": "politique",            "sous_type": "résistance à l'oppression",         "intensite": "forte"},
    {"type": "interne/psychologique","sous_type": "devoir contre désir",               "intensite": "forte"},
    {"type": "interne/psychologique","sous_type": "culpabilité et remords",            "intensite": "moyenne"},
    {"type": "idéologique",          "sous_type": "choc de valeurs morales opposées",  "intensite": "forte"},
    {"type": "existentiel",          "sous_type": "survie contre abandon",             "intensite": "très forte"},
    {"type": "économique",           "sous_type": "héritage disputé",                  "intensite": "moyenne"},
    {"type": "religieux",            "sous_type": "foi contre raison",                 "intensite": "forte"},
]

PIEGE_TYPES = [
    "tension latente sans affrontement actif — les personnages évitent le conflit",
    "opposition de valeurs sans confrontation directe entre actants",
    "charge émotionnelle intense (deuil, nostalgie) sans forces antagonistes identifiables",
    "hiérarchie de pouvoir acceptée sans résistance — soumission non conflictuelle",
    "désaccord exprimé mais sans volonté d'éviction ni enjeu structurant",
    "rivalité implicite non actualisée dans la scène",
    "conflit passé évoqué mais non rejoué dans le passage",
    "tension dramatique due à l'atmosphère, non à des actants en opposition",
]

NON_TYPES = [
    "description d'un lieu ou d'une atmosphère sans personnages en interaction",
    "monologue intérieur contemplatif sans tension oppositionnelle",
    "scène de retrouvailles affectueuses et harmonieuses",
    "portrait d'un personnage en état de repos ou d'acceptation",
    "narration d'un trajet ou d'une action quotidienne neutre",
    "dialogue de courtoisie sans enjeu ni opposition",
    "évocation nostalgique et paisible du passé",
    "description d'une cérémonie ou d'un rituel non conflictuel",
]

SETTINGS = [
    "Paris, XIXe siècle, salon bourgeois",
    "Province française, milieu rural, fin XIXe",
    "Cour royale fictive, époque médiévale",
    "Paris contemporain, appartement moderne",
    "Marseille, quartier populaire, années 1950",
    "Domaine aristocratique en déclin, fin XVIIIe",
    "Usine ou mine, contexte ouvrier, ère industrielle",
    "Tribunal ou salle d'audience solennelle",
    "Champ de bataille ou campement militaire",
    "Couvent ou institution religieuse",
    "Prison ou lieu de rétention",
    "Maison bourgeoise pendant une réunion familiale",
    "Café littéraire ou salon philosophique",
    "Navire en mer, huis clos maritime",
]

# ─────────────────────────────────────────────────────────────────────────────
# ③ PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_FINETUNE = """Tu es un expert en analyse littéraire spécialisé dans la détection et l'analyse des conflits dans les textes narratifs.

Tu utilises la définition suivante du conflit :
Conflit = « Affrontement (1) dynamique (2) entre forces antagonistes (3), avec volonté d'éviction (4) ou opposition profonde (5), et avec un enjeu sérieux (6) — c'est-à-dire que les forces antagonistes engagent des valeurs, des intérêts ou des principes fondamentaux (identité, survie, morale, pouvoir, etc.) dont la résolution a des conséquences importantes ou structurantes pour au moins l'un des actants. »

Formule logique : (1) ET (2) ET (3) ET [(4) OU (5)] ET (6)

Lorsqu'on te soumet un extrait littéraire, tu analyses chaque critère et fournis une évaluation rédigée, fluide et argumentée — comme le ferait un expert humain. Tu conclus toujours par une phrase de verdict explicite."""

USER_TEMPLATE = """Analyse le passage littéraire suivant et détermine s'il contient un conflit selon la définition analytique. Identifie les forces en présence, vérifie chaque critère de la définition, et rédige une analyse en paragraphe fluide avec un verdict final explicite.

Extrait :
{texte}"""

GENERATOR_SYSTEM = """Tu es un expert en littérature française et en analyse narratologique. Tu génères des exemples synthétiques pour un dataset de fine-tuning sur la détection de conflits littéraires.

Règles ABSOLUES :
- Texte littéraire : 4 à 10 phrases, crédible, bien écrit, dans le style demandé
- Analyse : paragraphe fluide (5 à 8 phrases), en prose continue, sans liste ni tiret ni markdown
- La DERNIÈRE phrase de l'analyse doit être EXACTEMENT l'une de ces trois formes :
    * "En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI."
    * "En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON."
    * "En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE."
- Cohérence totale : texte, analyse et verdict doivent être alignés
- Varier le vocabulaire et la structure à chaque génération
- Répondre UNIQUEMENT en JSON valide, sans markdown ni texte autour"""


def build_prompt(verdict, author, conflict_info, setting):
    verdict_sentence = VERDICT_SENTENCES[verdict]

    if verdict == "OUI":
        context = f"""- Type de conflit : {conflict_info['type']} — {conflict_info['sous_type']}
- Intensité : {conflict_info['intensite']}
- Le texte doit montrer clairement : affrontement, forces antagonistes, dynamique, enjeu sérieux"""

    elif verdict == "PIÈGE":
        context = f"""- Nature du piège : {conflict_info}
- Le texte doit sembler tendre vers un conflit mais ne pas en être un selon la définition
- Il manque au moins un critère (affrontement actif, volonté d'éviction, ou enjeu structurant)"""

    else:  # NON
        context = f"""- Type de passage : {conflict_info}
- Le texte doit être clairement sans conflit : descriptif, contemplatif ou harmonieux"""

    return f"""Génère un exemple synthétique avec verdict {verdict}.

Paramètres :
- Auteur / style : {author['name']} ({author['style']})
- Cadre : {setting}
{context}

La dernière phrase de l'analyse doit être exactement :
"{verdict_sentence}"

Réponds avec ce JSON :
{{
  "texte": "...",
  "analyse": "..."
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# ④ API ANTHROPIC
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key():
    key = API_KEY.strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("ERREUR : clé API introuvable.")
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
                model=MODEL,
                max_tokens=1200,
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


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    return json.loads(text.strip())


def validate_and_fix(analyse, verdict):
    """
    Vérifie que la phrase de verdict est bien présente à la fin.
    Si non, l'ajoute — garantie absolue pour les métriques.
    """
    expected = VERDICT_SENTENCES[verdict]
    text_up  = analyse.upper()

    has_verdict = (
        (verdict == "OUI"   and "VERDICT OUI"   in text_up) or
        (verdict == "NON"   and "VERDICT NON"   in text_up) or
        (verdict == "PIÈGE" and ("VERDICT PIÈGE" in text_up or "VERDICT PIEGE" in text_up))
    )

    if not has_verdict:
        analyse = analyse.rstrip() + " " + expected

    return analyse


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ PLANIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_distribution(n_oui, n_non, n_piege):
    """Retourne le dict de distribution et le total."""
    return {"OUI": n_oui, "NON": n_non, "PIÈGE": n_piege}, n_oui + n_non + n_piege


def plan_examples(distribution, seed=42):
    rng = random.Random(seed)
    plan = []

    for verdict, count in distribution.items():
        for _ in range(count):
            author  = rng.choice(AUTHORS)
            setting = rng.choice(SETTINGS)

            if verdict == "OUI":
                conflict_info = rng.choice(CONFLICT_TYPES_OUI)
            elif verdict == "PIÈGE":
                conflict_info = rng.choice(PIEGE_TYPES)
            else:
                conflict_info = rng.choice(NON_TYPES)

            plan.append({
                "verdict":       verdict,
                "author":        author,
                "conflict_info": conflict_info,
                "setting":       setting,
            })

    rng.shuffle(plan)
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path):
    entries = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if entries:
            print(f"  ♻  Checkpoint : {len(entries)} exemples déjà générés — reprise.")
    return entries


def append_checkpoint(entry, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Génère des exemples synthétiques de conflits littéraires",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python generate_synthetic.py --n 200
  python generate_synthetic.py --oui 100 --non 80 --piege 120
  python generate_synthetic.py --n 500 --output dataset_500.jsonl
        """
    )

    # Contrôle du nombre d'exemples
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n",     type=int, help="Nombre total (distribué automatiquement : 50%% OUI, 25%% NON, 25%% PIÈGE)")
    group.add_argument("--oui",   type=int, help="Nombre d'exemples OUI (utiliser avec --non et --piege)")

    parser.add_argument("--non",    type=int, default=0,   help="Nombre d'exemples NON")
    parser.add_argument("--piege",  type=int, default=0,   help="Nombre d'exemples PIÈGE")
    parser.add_argument("--output", type=str, default=None, help="Fichier de sortie (défaut : dataset_synthetic_N.jsonl)")
    parser.add_argument("--seed",   type=int, default=42,   help="Seed aléatoire (défaut : 42)")

    args = parser.parse_args()

    # Calculer la distribution
    if args.n:
        n_oui   = int(args.n * 0.50)
        n_non   = int(args.n * 0.25)
        n_piege = args.n - n_oui - n_non   # le reste va à PIÈGE
    else:
        if args.non == 0 or args.piege == 0:
            parser.error("--oui requiert aussi --non et --piege")
        n_oui, n_non, n_piege = args.oui, args.non, args.piege

    distribution, total = compute_distribution(n_oui, n_non, n_piege)

    output_path     = args.output or f"dataset_synthetic_{total}.jsonl"
    checkpoint_path = output_path.replace(".jsonl", "_checkpoint.jsonl")

    # Estimation du coût
    cost_low  = total * 0.009
    cost_high = total * 0.011

    print("=" * 60)
    print(f"Génération de {total} exemples synthétiques")
    print(f"  OUI   : {n_oui}")
    print(f"  NON   : {n_non}")
    print(f"  PIÈGE : {n_piege}")
    print(f"Modèle  : {MODEL}")
    print(f"Sortie  : {output_path}")
    print(f"Coût estimé : ${cost_low:.2f} – ${cost_high:.2f}")
    print("=" * 60)

    api_key      = get_api_key()
    entries      = load_checkpoint(checkpoint_path)
    already_done = len(entries)
    plan         = plan_examples(distribution, seed=args.seed)
    errors       = 0
    verdict_counts = {"OUI": 0, "NON": 0, "PIÈGE": 0}

    for i, params in enumerate(plan):
        progress = f"[{i+1:4d}/{total}]"
        verdict  = params["verdict"]
        verdict_counts[verdict] += 1

        if i < already_done:
            if (i + 1) % 50 == 0:
                print(f"  {progress} ✓ (checkpoint)")
            continue

        print(f"  {progress} {verdict:<8} {params['author']['name'][:25]:<25} →",
              end=" ", flush=True)

        try:
            prompt  = build_prompt(verdict, params["author"],
                                   params["conflict_info"], params["setting"])
            raw     = call_anthropic(prompt, api_key)
            parsed  = parse_json(raw)

            texte   = parsed.get("texte", "").strip()
            analyse = parsed.get("analyse", "").strip()

            if not texte or not analyse:
                raise ValueError("Champs vides dans la réponse")

            # Garantie absolue : verdict explicite présent
            analyse = validate_and_fix(analyse, verdict)

            entry = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT_FINETUNE},
                    {"role": "user",      "content": USER_TEMPLATE.format(texte=texte)},
                    {"role": "assistant", "content": analyse},
                ]
            }

            entries.append(entry)
            append_checkpoint(entry, checkpoint_path)
            print("✓")
            time.sleep(DELAY)

        except KeyboardInterrupt:
            print(f"\n\nInterruption à {i+1}/{total}. Checkpoint sauvegardé dans {checkpoint_path}")
            print("Relance avec les mêmes arguments pour reprendre.")
            sys.exit(0)

        except Exception as e:
            print(f"ERREUR : {e}")
            errors += 1
            time.sleep(2)

    # Écrire le fichier final
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if os.path.exists(checkpoint_path) and errors == 0:
        os.remove(checkpoint_path)

    # Vérification finale des verdicts
    missing = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            ex   = json.loads(line)
            text = ex["messages"][-1]["content"].upper()
            if not any(v in text for v in ["VERDICT OUI", "VERDICT NON",
                                           "VERDICT PIÈGE", "VERDICT PIEGE"]):
                missing += 1

    print("\n" + "=" * 60)
    print(f"TERMINÉ")
    print(f"  Exemples générés  : {len(entries)}")
    print(f"  Erreurs           : {errors}")
    print(f"  Verdicts manquants: {missing}")
    print(f"  Distribution      : {verdict_counts}")
    print(f"  Fichier de sortie : {output_path}")
    if missing == 0:
        print("  ✅ Tous les verdicts sont présents — compatible eval_model_v2.py")


if __name__ == "__main__":
    main()
