"""
eval_model_v2.py — Évaluation complète du modèle fine-tuné

Corrections :
  1. Split stratifié réel (même proportion OUI/NON/PIÈGE dans train et eval)
  2. F1 par classe correct (multiclass, pas micro sur label unique)
  3. Perplexité calculée sur l'eval set
  4. Évaluation sur TOUS les exemples (fix du bug 2/35)

Usage :
    python eval_model_v2.py
"""

import json
import re
import random
import math
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_DIR     = "./lora_model_V4"
DATASET_PATH = "dataset_conflit_final_v2.jsonl"
EVAL_RATIO   = 0.15
SEED         = 42
MAX_LENGTH   = 2048
OUTPUT_DIR   = "./eval_results"

LABEL_NAMES  = ["OUI", "NON", "PIÈGE"]

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

print("Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()
print("✅ Modèle chargé")

# ─────────────────────────────────────────────────────────────────────────────
# 2. SPLIT STRATIFIÉ RÉEL
# ─────────────────────────────────────────────────────────────────────────────

def extract_true_label(example):
    """Extrait le verdict ground-truth depuis le texte assistant."""
    text = example["messages"][-1]["content"].upper()
    if "VERDICT OUI"   in text: return 0
    if "VERDICT NON"   in text: return 1
    if re.search(r"VERDICT\s+PI[EÈ]GE", text): return 2
    return -1


def stratified_split(lines, eval_ratio, seed):
    """
    Split stratifié : garde les mêmes proportions OUI/NON/PIÈGE
    dans train et eval.
    """
    # Grouper par label
    groups = {0: [], 1: [], 2: []}
    unknowns = []

    for line in lines:
        ex    = json.loads(line)
        label = extract_true_label(ex)
        if label == -1:
            unknowns.append(line)
        else:
            groups[label].append(line)

    rng = random.Random(seed)
    eval_lines  = []
    train_lines = []

    for label, group in groups.items():
        rng.shuffle(group)
        n_eval = max(1, int(len(group) * eval_ratio))
        eval_lines.extend(group[:n_eval])
        train_lines.extend(group[n_eval:])
        print(f"  Label {LABEL_NAMES[label]:5s} : {len(group):4d} total "
              f"→ {n_eval:3d} eval / {len(group)-n_eval:3d} train")

    if unknowns:
        print(f"  ⚠ {len(unknowns)} exemples sans verdict ignorés")

    return train_lines, eval_lines


print("\n── Split stratifié ──────────────────────────────────")
with open(DATASET_PATH, encoding="utf-8") as f:
    all_lines = f.readlines()

train_lines, eval_lines = stratified_split(all_lines, EVAL_RATIO, SEED)
print(f"\nTrain : {len(train_lines)} | Eval : {len(eval_lines)}")

# Vérifier la distribution dans l'eval
eval_labels = [extract_true_label(json.loads(l)) for l in eval_lines]
dist = Counter(eval_labels)
print(f"Distribution eval : OUI={dist[0]}  NON={dist[1]}  PIÈGE={dist[2]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PERPLEXITÉ
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(lines, model, tokenizer, max_length):
    """
    Calcule la perplexité moyenne sur un ensemble de lignes JSONL.
    Perplexité = exp(loss moyenne) — mesure à quel point le modèle
    est "surpris" par les réponses correctes.
    Plus bas = mieux. Baseline Mistral non fine-tuné : ~8-15.
    """
    total_loss   = 0.0
    total_tokens = 0

    for i, line in enumerate(lines):
        ex       = json.loads(line)
        messages = ex["messages"]

        full_text   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )

        encodings = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_len = len(
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

        input_ids = encodings["input_ids"].to(model.device)
        labels    = input_ids.clone()
        labels[:, :prompt_len] = -100   # masque le prompt

        n_supervised = int((labels != -100).sum())
        if n_supervised == 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss    = outputs.loss.item()

        if math.isnan(loss) or math.isinf(loss):
            continue

        total_loss   += loss * n_supervised
        total_tokens += n_supervised

        if (i + 1) % 10 == 0:
            print(f"  Perplexité [{i+1}/{len(lines)}] loss_courante={loss:.4f}")

    if total_tokens == 0:
        return float("nan")

    avg_loss    = total_loss / total_tokens
    perplexity  = math.exp(avg_loss)
    return perplexity


print("\n── Calcul de la perplexité ──────────────────────────")
perplexity = compute_perplexity(eval_lines, model, tokenizer, MAX_LENGTH)
print(f"\n📊 Perplexité : {perplexity:.4f}")
if perplexity < 3.0:
    print("   → Excellent (modèle très sûr de ses réponses)")
elif perplexity < 6.0:
    print("   → Bon")
elif perplexity < 10.0:
    print("   → Acceptable")
else:
    print("   → Élevée — le modèle est incertain (plus d'entraînement nécessaire)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTRIQUES DE CLASSIFICATION (F1 correct)
# ─────────────────────────────────────────────────────────────────────────────

def predict_verdict(example, model, tokenizer, max_length):
    """Génère la réponse du modèle et extrait le verdict prédit."""
    messages    = example["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True).upper()

    if "VERDICT OUI"   in gen_text: return 0, gen_text
    if "VERDICT NON"   in gen_text: return 1, gen_text
    if re.search(r"VERDICT\s+PI[EÈ]GE", gen_text): return 2, gen_text
    return 1, gen_text   # default NON si non parseable


print("\n── Métriques de classification ──────────────────────")
y_true   = []
y_pred   = []
failures = 0   # exemples où le verdict n'a pas été trouvé

for i, line in enumerate(eval_lines):
    ex         = json.loads(line)
    true_label = extract_true_label(ex)
    if true_label == -1:
        continue

    pred_label, gen_text = predict_verdict(ex, model, tokenizer, MAX_LENGTH)

    # Détecter si le verdict était vraiment présent ou si on a utilisé le défaut
    verdict_found = any([
        "VERDICT OUI" in gen_text,
        "VERDICT NON" in gen_text,
        re.search(r"VERDICT\s+PI[EÈ]GE", gen_text),
    ])
    if not verdict_found:
        failures += 1

    y_true.append(true_label)
    y_pred.append(pred_label)

    status = "✅" if pred_label == true_label else "❌"
    print(f"  [{i+1:3d}/{len(eval_lines)}] {status} "
          f"true={LABEL_NAMES[true_label]:5s} "
          f"pred={LABEL_NAMES[pred_label]:5s}")

print(f"\n⚠ Verdicts non trouvés (défaut NON utilisé) : {failures}/{len(y_true)}")

# ── F1 correct : multiclass avec average='macro' ──────────────────────────
print("\n" + "=" * 55)
print("RAPPORT DE CLASSIFICATION COMPLET")
print("=" * 55)
report = classification_report(
    y_true, y_pred,
    labels=[0, 1, 2],
    target_names=LABEL_NAMES,
    zero_division=0,
    digits=4,
)
print(report)

acc      = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
f1_micro = f1_score(y_true, y_pred, average="micro",    zero_division=0)
f1_w     = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print(f"Accuracy        : {acc:.4f}")
print(f"F1 macro        : {f1_macro:.4f}  ← métrique principale (classes équilibrées)")
print(f"F1 micro        : {f1_micro:.4f}  ← métrique globale")
print(f"F1 weighted     : {f1_w:.4f}  ← tient compte du déséquilibre")
print(f"Perplexité      : {perplexity:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. MATRICE DE CONFUSION
# ─────────────────────────────────────────────────────────────────────────────

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABEL_NAMES,
    yticklabels=LABEL_NAMES,
    ax=ax,
)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
ax.set_title(f"Matrice de confusion\nAccuracy={acc:.3f}  F1 macro={f1_macro:.3f}  PPL={perplexity:.2f}")
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, "confusion_matrix_v2.png")
plt.savefig(path, dpi=150)
plt.close()
print(f"\n🗂  Matrice de confusion → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAUVEGARDE JSON
# ─────────────────────────────────────────────────────────────────────────────

results = {
    "perplexity":   round(perplexity, 4),
    "accuracy":     round(acc,        4),
    "f1_macro":     round(f1_macro,   4),
    "f1_micro":     round(f1_micro,   4),
    "f1_weighted":  round(f1_w,       4),
    "f1_per_class": {
        name: round(f1_score(y_true, y_pred, labels=[i],
                             average="micro", zero_division=0), 4)
        for i, name in enumerate(LABEL_NAMES)
    },
    "n_eval":       len(y_true),
    "n_failures":   failures,
    "distribution": {LABEL_NAMES[k]: v for k, v in Counter(y_true).items()},
}

out_path = os.path.join(OUTPUT_DIR, "eval_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"📋 Résultats complets → {out_path}")
print("\n✅ Évaluation terminée")
