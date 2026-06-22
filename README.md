# Détection de conflits littéraires — Fine-tuning de Mistral-7B

Fine-tuning de **Mistral-7B-Instruct-v0.2** (via LoRA) pour la détection et l'analyse automatique de conflits dans des extraits littéraires, selon une définition analytique précise.

> Projet de recherche réalisé dans le cadre d'un travail supervisé en NLP / Machine Learning, sous la direction de Ph. HERR.

---

## 🎯 Objectif

Le modèle classifie chaque extrait littéraire en trois catégories :

- **OUI** — un conflit réel est présent
- **NON** — aucun conflit
- **PIÈGE** — tension forte mais ne constituant pas un conflit au sens strict de la définition

### Définition analytique utilisée

> Conflit = « Affrontement (1) dynamique (2) entre forces antagonistes (3), avec volonté d'éviction (4) ou opposition profonde (5), et avec un enjeu sérieux (6). »
>
> Formule logique : (1) ET (2) ET (3) ET [(4) OU (5)] ET (6)

---

## 📊 Résultats

Évaluation finale sur 58 exemples (split stratifié, jamais vus en entraînement) :

| Classe | Precision | Recall | F1-score |
|--------|-----------|--------|----------|
| OUI    | 0.95      | 0.95   | **0.95** |
| NON    | 0.94      | 1.00   | **0.97** |
| PIÈGE  | 0.95      | 0.90   | **0.92** |

```
Accuracy global : 0.948
F1 macro         : 0.949
Perplexité       : 2.37
```

📈 Voir [`docs/RESULTS.md`](docs/RESULTS.md) pour l'évolution complète des métriques au fil des itérations.

---

## 🏗️ Architecture du pipeline

```
1. Données Excel validées (180 ex.)  ─┐
                                       ├─→  JSONL formaté  ─→  Dataset final
2. Génération synthétique (618 ex.) ──┘     (texte + analyse +    (846 ex.)
   via API Anthropic (Claude)              verdict explicite)
                                                    │
                                                    ▼
                                          Fine-tuning LoRA
                                          (Mistral-7B-Instruct-v0.2)
                                                    │
                                                    ▼
                                          Évaluation (F1, perplexité,
                                          matrice de confusion)
```

---

## 📁 Structure du projet

```
.
├── src/
│   ├── data/
│   │   ├── 01_excel_to_jsonl.py       # Conversion Excel → JSONL (données réelles)
│   │   └── 02_generate_synthetic.py   # Génération de données synthétiques via API
│   ├── training/
│   │   └── train.py                    # Fine-tuning LoRA
│   ├── evaluation/
│   │   ├── evaluate.py                 # Métriques complètes (F1, perplexité, confusion matrix)
│   │   └── debug_dataset.py            # Outils de diagnostic du dataset
│   └── inference/
│       └── chat.py                     # Interface CLI pour tester le modèle
├── docs/
│   └── RESULTS.md                      # Historique détaillé des résultats
├── examples/
│   └── sample_outputs.md               # Exemples d'analyses générées
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
git clone https://github.com/<ton-username>/conflit-litteraire-llm.git
cd conflit-litteraire-llm
pip install -r requirements.txt
```

**Prérequis :**
- Python 3.10+
- GPU avec au moins 24 GB VRAM (testé sur 3× RTX 3090)
- Clé API Anthropic (pour la génération de données — `export ANTHROPIC_API_KEY=...`)

---

## 🔧 Utilisation

### 1. Préparer les données

```bash
# Convertir les fichiers Excel validés en JSONL
python src/data/01_excel_to_jsonl.py

# Générer des exemples synthétiques supplémentaires
python src/data/02_generate_synthetic.py --oui 280 --non 210 --piege 130
```

### 2. Entraîner le modèle

```bash
python src/training/train.py
```

### 3. Évaluer

```bash
python src/evaluation/evaluate.py
```

### 4. Tester en interactif

```bash
python src/inference/chat.py --lora ./lora_model_V5
```

---

## 🧠 Choix méthodologiques clés

- **Paragraphes analytiques fluides** plutôt que des réponses structurées en listes — pour que le modèle apprenne un raisonnement plutôt qu'un format.
- **Verdict explicite standardisé** en fin d'analyse (`"verdict OUI/NON/PIÈGE"`) — nécessaire pour permettre l'évaluation automatique.
- **Split stratifié** garantissant les mêmes proportions de classes en train/eval — évite les métriques biaisées.
- **LoRA (r=16, α=32)** ciblant les modules d'attention et de feed-forward — fine-tuning efficace en mémoire (0.58% des paramètres entraînés).

Voir [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) pour le détail des choix techniques et des problèmes rencontrés.

---

## ⚠️ Limites connues

- Les données synthétiques peuvent introduire des biais statistiques absents de la définition théorique (voir `docs/RESULTS.md`).
- Le modèle peut occasionnellement produire des contradictions internes dans son raisonnement — un comportement connu des LLMs génératifs.
- La frontière OUI/PIÈGE reste la plus difficile à trancher, y compris pour un lecteur humain.

---

## 📜 Licence

Ce projet est sous licence MIT — voir [`LICENSE`](LICENSE).

Le modèle de base utilisé, `Mistral-7B-Instruct-v0.2`, est soumis à la licence Apache 2.0 de Mistral AI.

---

## 🙏 Remerciements

Projet réalisé sous la supervision de **Ph. HERR**, dans le cadre d'un travail de recherche en analyse littéraire computationnelle.
