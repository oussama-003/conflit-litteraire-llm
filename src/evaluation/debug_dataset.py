"""
Script de debug — à lancer AVANT de relancer l'entraînement.
Vérifie :
  1. Combien de tokens sont supervisés (non maskés) par exemple
  2. Le contenu réel de la fin des textes assistant dans le JSONL
"""

import json
import torch
from transformers import AutoTokenizer

# ── Config — adapte ces deux chemins ──────────────────────────────────────────
MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "dataset_conflit_final_v2.jsonl"
MAX_LENGTH   = 512
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

# ── 1. Vérification des tokens supervisés ────────────────────────────────────
print("=" * 60)
print("CHECK 1 — Tokens supervisés (non maskés) par exemple")
print("=" * 60)

with open(DATASET_PATH, encoding="utf-8") as f:
    lines = f.readlines()

zero_supervised = 0
for i, line in enumerate(lines[:10]):   # teste les 10 premiers
    example = json.loads(line)
    messages = example["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    encodings = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    input_ids      = encodings["input_ids"][0]
    attention_mask = encodings["attention_mask"][0]

    labels = input_ids.clone()
    labels[:prompt_len]         = -100
    labels[attention_mask == 0] = -100

    non_masked = int((labels != -100).sum())
    total      = MAX_LENGTH
    truncated  = full_text != tokenizer.decode(
        tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LENGTH]
    )

    if non_masked == 0:
        zero_supervised += 1

    status = "🔴 PROBLÈME" if non_masked == 0 else "✅ OK"
    print(f"  Exemple {i:3d} | tokens supervisés : {non_masked:4d}/{total} "
          f"| prompt_len : {prompt_len:4d} | {status}")

print()
if zero_supervised > 0:
    print(f"⚠  {zero_supervised}/10 exemples ont 0 tokens supervisés → NaN garanti")
    print(f"   → Fix : augmenter MAX_LENGTH (essaie 1024 ou 2048)")
else:
    print("✅ Tous les exemples ont des tokens supervisés — masquage OK")

# ── 2. Vérification du verdict dans les 200 derniers chars ───────────────────
print()
print("=" * 60)
print("CHECK 2 — Fin des textes assistant (verdict explicite ?)")
print("=" * 60)

found    = 0
missing  = 0
for i, line in enumerate(lines):
    example   = json.loads(line)
    assistant = example["messages"][-1]["content"]
    tail      = assistant[-200:]
    tail_up   = tail.upper()

    has_oui   = "VERDICT OUI"   in tail_up
    has_non   = "VERDICT NON"   in tail_up
    has_piege = "VERDICT PIÈGE" in tail_up or "VERDICT PIEGE" in tail_up

    if has_oui or has_non or has_piege:
        found += 1
        verdict = "OUI" if has_oui else ("NON" if has_non else "PIÈGE")
        if i < 5:
            print(f"  Exemple {i} ✅ verdict={verdict}")
            print(f"    ...{tail[-120:]!r}")
    else:
        missing += 1
        if missing <= 5:
            print(f"  Exemple {i} 🔴 VERDICT MANQUANT")
            print(f"    ...{tail[-120:]!r}")

print()
print(f"Résumé : {found} verdicts trouvés, {missing} manquants sur {len(lines)} exemples")
if missing == 0:
    print("✅ Tous les verdicts sont présents — métriques fonctionneront")
else:
    print("⚠  Verdicts manquants → régénérer avec generate_jsonl_local_v2.py")
