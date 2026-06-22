"""
chat_model.py — Interface de chat avec le modèle fine-tuné
Détection de conflits littéraires (Mistral-7B + LoRA)

Usage :
    python chat_model.py
    python chat_model.py --lora ./lora_model_V4
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME    = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_DIR      = "./lora_model_V4"   # modifiable via --lora
MAX_LENGTH    = 2048
MAX_NEW_TOKENS = 500

SYSTEM_PROMPT = """Tu es un expert en analyse littéraire spécialisé dans la détection et l'analyse des conflits dans les textes narratifs.

Tu utilises la définition suivante du conflit :
Conflit = « Affrontement (1) dynamique (2) entre forces antagonistes (3), avec volonté d'éviction (4) ou opposition profonde (5), et avec un enjeu sérieux (6) — c'est-à-dire que les forces antagonistes engagent des valeurs, des intérêts ou des principes fondamentaux (identité, survie, morale, pouvoir, etc.) dont la résolution a des conséquences importantes ou structurantes pour au moins l'un des actants. »

Formule logique : (1) ET (2) ET (3) ET [(4) OU (5)] ET (6)

Lorsqu'on te soumet un extrait littéraire, tu analyses chaque critère et fournis une évaluation rédigée, fluide et argumentée — comme le ferait un expert humain. Tu conclus toujours par une phrase de verdict explicite."""

USER_TEMPLATE = """Analyse le passage littéraire suivant et détermine s'il contient un conflit selon la définition analytique. Identifie les forces en présence, vérifie chaque critère de la définition, et rédige une analyse en paragraphe fluide avec un verdict final explicite.

Extrait :
{texte}"""

BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Détecteur de conflits littéraires                 ║
║        Mistral-7B + LoRA — Fine-tuné                     ║
╠══════════════════════════════════════════════════════════╣
║  Commandes :                                             ║
║    [Entrée vide]  → soumettre le texte                   ║
║    quit / exit    → quitter                              ║
║    clear          → effacer l'écran                      ║
║    model          → afficher le modèle chargé            ║
╚══════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def load_model(lora_dir):
    print(f"\n⏳ Chargement du modèle de base : {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"⏳ Chargement des adaptateurs LoRA : {lora_dir}")
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    # Afficher la distribution des GPUs
    if hasattr(model, "hf_device_map"):
        devices = set(str(v) for v in model.hf_device_map.values())
        print(f"✅ Modèle chargé sur : {', '.join(sorted(devices))}")
    else:
        print("✅ Modèle chargé")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# INFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────

def analyse(texte, model, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(texte=texte)},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(model.device)

    n_prompt_tokens = inputs["input_ids"].shape[1]

    print("\n⏳ Analyse en cours...\n")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,     # réduit les répétitions
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids  = output_ids[0][n_prompt_tokens:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return response


def extract_verdict(response):
    """Extrait le verdict pour l'afficher en évidence."""
    r = response.upper()
    if "VERDICT OUI"   in r: return "OUI",   "🟢"
    if "VERDICT NON"   in r: return "NON",   "⚪"
    if "VERDICT PIÈGE" in r or "VERDICT PIEGE" in r: return "PIÈGE", "🟡"
    return "INDÉTERMINÉ", "❓"


# ─────────────────────────────────────────────────────────────────────────────
# SAISIE MULTI-LIGNES
# ─────────────────────────────────────────────────────────────────────────────

def read_multiline_input():
    """
    Permet de saisir un texte sur plusieurs lignes.
    Terminer avec une ligne vide (Entrée deux fois).
    """
    print("📝 Entrez votre extrait (terminez avec une ligne vide) :")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(model, tokenizer, lora_dir):
    import os
    print(BANNER)

    session_count = 0

    while True:
        print("─" * 60)
        texte = read_multiline_input()

        # Commandes spéciales
        if not texte:
            continue
        if texte.lower() in ("quit", "exit", "q"):
            print("\nAu revoir !\n")
            break
        if texte.lower() == "clear":
            os.system("clear" if os.name == "posix" else "cls")
            print(BANNER)
            continue
        if texte.lower() == "model":
            print(f"\n  Modèle de base : {MODEL_NAME}")
            print(f"  Adaptateurs    : {lora_dir}")
            print(f"  Analyses faites: {session_count}\n")
            continue
        if len(texte) < 20:
            print("⚠  Texte trop court — entrez un extrait littéraire.\n")
            continue

        # Analyse
        session_count += 1
        response       = analyse(texte, model, tokenizer)
        verdict, emoji = extract_verdict(response)

        print("┌─ ANALYSE " + "─" * 49)
        print(response)
        print("└" + "─" * 59)
        print(f"\n{emoji}  VERDICT FINAL : {verdict}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat avec le modèle fine-tuné")
    parser.add_argument("--lora", default=LORA_DIR, help="Chemin vers les adaptateurs LoRA")
    parser.add_argument("--base", default=MODEL_NAME, help="Modèle de base HuggingFace")
    args = parser.parse_args()

    MODEL_NAME_USE = args.base
    model, tokenizer = load_model(args.lora)
    chat_loop(model, tokenizer, args.lora)
