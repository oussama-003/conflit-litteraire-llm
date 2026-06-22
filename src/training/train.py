"""
train_lora6.py — Fine-tuning Mistral-7B + LoRA
Dataset : conflits littéraires (OUI / NON / PIÈGE)

Ce script fait UNE seule chose : entraîner proprement.
Les métriques de classification détaillées (F1, matrice de confusion, perplexité)
sont dans eval_model_v2.py — à lancer séparément après l'entraînement.

Ce script affiche uniquement :
  - train_loss  (descend = le modèle apprend)
  - eval_loss   (descend sans diverger = pas d'overfitting)
  - Distribution du split stratifié (vérification)
  - Early stopping si eval_loss ne s'améliore plus
"""

import json
import os
import re
import random
import torch

from collections import Counter
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────────────────────
# ① CONFIGURATION — tout ce qu'on peut vouloir changer est ici
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME   = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = "dataset_conflits_final_v2.jsonl"
OUTPUT_DIR   = "./results_V5"
LORA_DIR     = "./lora_model_V5"

MAX_LENGTH   = 2048
EVAL_SPLIT   = 0.10        # 10% eval — plus de données en train
SEED         = 42

# LoRA
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# Entraînement
NUM_EPOCHS          = 5
BATCH_SIZE          = 4     # par GPU — ok sur 24GB avec MAX_LENGTH=2048
GRAD_ACCUM          = 4     # effective batch = 4 × 4 GPUs = 16
LEARNING_RATE       = 5e-5
MAX_GRAD_NORM       = 0.3
EARLY_STOP_PATIENCE = 3     # epochs sans amélioration avant arrêt

# ─────────────────────────────────────────────────────────────────────────────
# ② TOKENIZER & MODEL
# ─────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,   # fix du warning "torch_dtype deprecated"
    device_map="auto",
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────────────────────────────────────
# ③ DATASET — split stratifié réel
# ─────────────────────────────────────────────────────────────────────────────

def get_verdict(example):
    """Extrait le verdict depuis la fin du texte assistant (où il se trouve)."""
    text = example["messages"][-1]["content"].upper()
    if "VERDICT OUI"   in text: return 0
    if "VERDICT NON"   in text: return 1
    if re.search(r"VERDICT\s*PI[EÈ]GE", text): return 2
    return -1


def stratified_split(examples, eval_ratio, seed):
    """
    Split stratifié : garantit les mêmes proportions OUI/NON/PIÈGE
    dans train et eval — contrairement au train_test_split() aléatoire simple.
    """
    groups   = {0: [], 1: [], 2: []}
    unknown  = []

    for ex in examples:
        label = get_verdict(ex)
        if label == -1:
            unknown.append(ex)
        else:
            groups[label].append(ex)

    if unknown:
        print(f"  ⚠ {len(unknown)} exemples sans verdict ignorés")

    rng        = random.Random(seed)
    train_exs  = []
    eval_exs   = []
    names      = {0: "OUI", 1: "NON", 2: "PIÈGE"}

    print(f"\n── Split stratifié (eval={eval_ratio*100:.0f}%) ─────────────────")
    print(f"  {'Classe':<8} {'Total':>6} {'Train':>6} {'Eval':>6} "
          f"{'Train%':>8} {'Eval%':>8}")
    print("  " + "─" * 46)

    for label, group in groups.items():
        rng.shuffle(group)
        n_eval  = max(1, int(len(group) * eval_ratio))
        n_train = len(group) - n_eval
        eval_exs.extend(group[:n_eval])
        train_exs.extend(group[n_eval:])
        t_pct = n_train / len(group) * 100
        e_pct = n_eval  / len(group) * 100
        print(f"  {names[label]:<8} {len(group):>6} {n_train:>6} {n_eval:>6} "
              f"{t_pct:>7.1f}% {e_pct:>7.1f}%")

    print("  " + "─" * 46)
    print(f"  {'TOTAL':<8} {len(train_exs)+len(eval_exs):>6} "
          f"{len(train_exs):>6} {len(eval_exs):>6}")
    print()

    return train_exs, eval_exs


# Charger et splitter
with open(DATASET_PATH, encoding="utf-8") as f:
    all_examples = [json.loads(line) for line in f if line.strip()]

train_examples, eval_examples = stratified_split(all_examples, EVAL_SPLIT, SEED)

# Vérifier qu'on a assez de tokens supervisés sur quelques exemples
def check_supervision(examples, n=5):
    issues = 0
    for ex in examples[:n]:
        messages    = ex["messages"]
        full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        enc         = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        p_len       = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        attn        = enc["attention_mask"][0]
        labels      = enc["input_ids"][0].clone()
        labels[:p_len]       = -100
        labels[attn == 0]    = -100
        n_sup = int((labels != -100).sum())
        if n_sup == 0:
            issues += 1
            print(f"  ⚠ Exemple sans tokens supervisés — prompt_len={p_len} > MAX_LENGTH={MAX_LENGTH}")
    if issues == 0:
        print(f"  ✅ Vérification supervision OK (échantillon {n} exemples)")

print("── Vérification tokens supervisés ──────────────────")
check_supervision(train_examples)
print()

# Tokenisation
def format_and_tokenize(example):
    messages    = example["messages"]
    full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    encodings  = tokenizer(
        full_text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    prompt_len     = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    input_ids      = encodings["input_ids"][0]
    attention_mask = encodings["attention_mask"][0]
    labels         = input_ids.clone()
    labels[:prompt_len]         = -100
    labels[attention_mask == 0] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

train_dataset_raw = Dataset.from_list(train_examples)
eval_dataset_raw  = Dataset.from_list(eval_examples)

train_dataset = train_dataset_raw.map(format_and_tokenize, remove_columns=train_dataset_raw.column_names)
eval_dataset  = eval_dataset_raw.map(format_and_tokenize,  remove_columns=eval_dataset_raw.column_names)

# ─────────────────────────────────────────────────────────────────────────────
# ④ TRAINING
# ─────────────────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir                = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs          = NUM_EPOCHS,
    learning_rate             = LEARNING_RATE,
    bf16                      = True,
    max_grad_norm             = MAX_GRAD_NORM,
    logging_steps             = 10,
    save_strategy             = "epoch",
    eval_strategy             = "epoch",
    save_total_limit          = 2,
    load_best_model_at_end    = True,
    metric_for_best_model     = "eval_loss",
    greater_is_better         = False,
    report_to                 = "none",
    optim                     = "adamw_torch",
    dataloader_num_workers    = 4,
    seed                      = SEED,
)

trainer = Trainer(
    model         = model,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    args          = training_args,
    callbacks     = [
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE),
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# ⑤ RUN
# ─────────────────────────────────────────────────────────────────────────────

print("🚀 Démarrage de l'entraînement...\n")
print("   Ce script affiche train_loss et eval_loss uniquement.")
print("   Pour les métriques F1/perplexité → lancer eval_model_v2.py après.\n")

trainer.train()

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

print(f"\n✅ Fine-tuning terminé")
print(f"   Modèle sauvegardé → {LORA_DIR}")
print(f"   Checkpoints       → {OUTPUT_DIR}")
print(f"\n   Prochaine étape : python eval_model_v2.py --lora {LORA_DIR}")
