# Méthodologie

## 1. Choix du modèle de base

**Mistral-7B-Instruct-v0.2** a été retenu pour :
- de bonnes performances en français,
- une taille permettant le fine-tuning sur des GPU grand public (24 GB VRAM),
- un format conversationnel (system/user/assistant) adapté à la tâche.

## 2. Construction du dataset

### 2.1 Données réelles

Trois fichiers Excel ont été constitués et validés manuellement par le superviseur du projet. Chaque ligne contient :
- le texte de l'extrait littéraire,
- le verdict (OUI / NON),
- une analyse structurée en 10+ colonnes (forces antagonistes, nature du conflit, dynamique, volonté d'éviction, opposition profonde, etc.).

**Détection du verdict réel.** Une incohérence a été identifiée entre la colonne `Conflit(OUI/NON)` et la colonne `Validation globale` sur l'un des fichiers : certaines lignes marquées `NON` avaient une validation qui commençait par `OUI`, en réalité signalées `PIÈGE` dans la colonne `Détail`. La règle de résolution adoptée :

```python
def detect_verdict(row):
    if "PIÈGE" in (row["Détail"] or row["Validation globale"]):
        return "PIÈGE"
    if row["Conflit"] == "OUI":
        return "OUI"
    return "NON"
```

### 2.2 Génération des paragraphes analytiques

Les colonnes structurées ont été transformées en paragraphes analytiques fluides via l'API Anthropic (Claude Sonnet 4), plutôt que par des templates de phrases fixes.

**Justification :** des templates rigides (`"Ce passage présente un conflit de type X..."`) conduisent le modèle fine-tuné à mémoriser une structure plutôt qu'à apprendre un raisonnement analytique transférable. Une première tentative avec un format structuré (titres markdown, listes à puces) a confirmé ce risque — voir `docs/RESULTS.md`, itération V1.

**Verdict explicite obligatoire.** Chaque paragraphe généré se termine par une phrase standardisée :
```
"En conclusion, ce passage constitue bien un conflit au sens de la définition : verdict OUI."
"En conclusion, ce passage ne constitue pas un conflit au sens de la définition : verdict NON."
"En conclusion, ce passage constitue un cas limite ou piège analytique : verdict PIÈGE."
```
Cette contrainte est validée et corrigée automatiquement après génération (`validate_and_fix()`), garantissant que 100% des exemples contiennent un verdict parsable pour l'évaluation automatique.

### 2.3 Génération de données synthétiques

Pour pallier la taille réduite du dataset réel (180 exemples validés) et son déséquilibre entre classes, des exemples synthétiques ont été générés (texte littéraire inventé + analyse complète) via le même mécanisme.

**Axes de diversité contrôlés :**
- 18 profils auteur/style (classiques français, théâtre, modernes, fictifs)
- 14 types de conflits OUI, 8 types PIÈGE, 8 types NON
- 14 cadres narratifs (salon bourgeois, usine, tribunal, navire, etc.)

Distribution finale du dataset complet (846 exemples) : approximativement équilibrée entre les trois classes (50% OUI, 25% NON, 25% PIÈGE).

## 3. Fine-tuning

### 3.1 LoRA (Low-Rank Adaptation)

Plutôt que d'entraîner les 7 milliards de paramètres du modèle, LoRA insère de petites matrices de rang réduit sur les couches d'attention et de feed-forward, réduisant le nombre de paramètres entraînables à 0.58% du total (41.9M / 7.28B).

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0.10,
    task_type="CAUSAL_LM",
)
```

### 3.2 Masquage des labels (teacher forcing sélectif)

Seuls les tokens correspondant à la réponse de l'assistant participent au calcul de la loss ; le prompt système et la question utilisateur sont masqués (`label = -100`).

**Piège rencontré :** avec une longueur de séquence maximale (`MAX_LENGTH`) trop courte par rapport à la longueur réelle des prompts (system + user ≈ 800-900 tokens), l'intégralité des tokens de la réponse pouvait se retrouver hors de la fenêtre, masquant 100% des labels et produisant une loss `NaN`. Résolu en portant `MAX_LENGTH` à 2048.

### 3.3 Split stratifié

Le split aléatoire simple (`train_test_split`) ne garantit pas des proportions de classes identiques entre l'ensemble d'entraînement et l'ensemble d'évaluation, en particulier sur un dataset de taille modeste. Un split stratifié manuel a été implémenté pour garantir des proportions de classes identiques entre train et eval :

```python
def stratified_split(examples, eval_ratio, seed):
    groups = {0: [], 1: [], 2: []}  # OUI, NON, PIÈGE
    for ex in examples:
        groups[get_verdict(ex)].append(ex)
    # split indépendamment au sein de chaque groupe
    ...
```

### 3.4 Choix des hyperparamètres

| Paramètre | Valeur initiale | Valeur finale | Justification du changement |
|-----------|------------------|---------------|------------------------------|
| `learning_rate` | 2e-4 | 2e-5 | Valeur initiale trop agressive → divergence puis overfitting précoce |
| `lora_dropout` | 0.05 | 0.10 | Régularisation accrue pour limiter la mémorisation |
| `warmup_ratio` | — | 0.05 | Montée progressive du LR, évite les premiers pas trop brusques |
| `num_train_epochs` | 3 | 5 | Dataset élargi permettant plus d'epochs avant overfitting |
| `eval_strategy` / `save_strategy` | mismatch (`epoch` / `steps`) | les deux `epoch` | Requis par `load_best_model_at_end=True` |

## 4. Évaluation

L'évaluation se fait en deux temps, séparés du script d'entraînement pour éviter les instabilités liées à la génération de texte pendant un entraînement distribué multi-GPU :

1. **Pendant l'entraînement** (`train.py`) : uniquement `train_loss` / `eval_loss`, suivi de l'overfitting via `EarlyStoppingCallback`.
2. **Après l'entraînement** (`evaluate.py`) : génération réelle sur l'ensemble d'évaluation stratifié, calcul du F1 par classe (macro/micro/weighted), perplexité, matrice de confusion.

### Métriques retenues

- **Perplexité** — `exp(eval_loss)`, mesure la confiance du modèle dans ses réponses.
- **F1 macro** — moyenne non pondérée du F1 par classe, retenue comme métrique principale car elle ne favorise pas les classes majoritaires.
- **Matrice de confusion** — permet d'identifier les paires de classes les plus souvent confondues (ici, principalement OUI ↔ PIÈGE).

## 5. Limite méthodologique identifiée

Un test qualitatif post-entraînement (analyse d'un texte de La Fontaine, hors dataset) a révélé une **contradiction interne** dans une réponse du modèle, ainsi qu'un biais : le modèle associait à tort l'absence d'affrontement *physique* au verdict PIÈGE, alors que ce critère n'existe pas dans la définition théorique. Cette association résulte très probablement d'une corrélation statistique introduite involontairement par les prompts de génération synthétique. Voir `docs/RESULTS.md` pour le détail.
