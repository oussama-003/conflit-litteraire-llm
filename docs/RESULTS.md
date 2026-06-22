# Historique des résultats

## Évolution des métriques au fil des itérations

| Itération | Dataset | F1 OUI | F1 NON | F1 PIÈGE | F1 macro | Accuracy | Notes |
|-----------|---------|--------|--------|----------|----------|----------|-------|
| V1 | 183 ex. — format structuré (listes/titres) | 0.79 | 0.29 | 0.25 | 0.44 | 0.61 | Verdict mal détecté (callback cherchait dans les 200 premiers caractères) |
| V2 | 228 ex. — paragraphes fluides | 0.73 | 0.63 | 0.47 | 0.61 | 0.64 | Verdict explicite ajouté en fin de paragraphe |
| V3 | 346 ex. — split stratifié | 0.84 | 0.97 | 0.84 | 0.88 | 0.88 | `lr=5e-5`, overfitting dès epoch 2-3 |
| **V4** | **846 ex. — lr réduit + dropout** | **0.95** | **0.97** | **0.92** | **0.95** | **0.95** | `lr=2e-5`, `dropout=0.10`, `warmup_ratio=0.05` |

---

## Détail du run final (V4)

### Courbes d'entraînement

```
Epoch | Train Loss | Eval Loss
------|------------|----------
  1   |    1.03    |   0.971
  2   |    0.87    |   0.879
  3   |    0.65    |   0.859   ← meilleur checkpoint
  4   |    0.58    |   0.864
  5   |    0.58    |   0.885
```

### Rapport de classification (58 exemples, split stratifié 10%)

```
              precision    recall  f1-score   support
         OUI     0.9524    0.9524    0.9524        21
         NON     0.9444    1.0000    0.9714        17
       PIÈGE     0.9474    0.9000    0.9231        20

    accuracy                         0.9483        58
   macro avg     0.9481    0.9508    0.9490        58
weighted avg     0.9483    0.9483    0.9479        58

Perplexité : 2.3692
```

### Erreurs résiduelles (3 sur 58)

Toutes les erreurs se situent aux frontières naturelles entre classes adjacentes :

| Cas | Réel | Prédit | Interprétation |
|-----|------|--------|-----------------|
| 1 | OUI | PIÈGE | Conflit réel perçu comme cas limite |
| 2 | PIÈGE | NON | Cas limite perçu comme neutre |
| 3 | PIÈGE | OUI | Cas limite perçu comme conflit réel |

Ces erreurs sont analytiquement cohérentes — elles touchent des cas où même un expert humain pourrait hésiter dans la classification.

---

## Problèmes techniques rencontrés et résolus

| # | Problème observé | Cause racine | Solution |
|---|-------------------|--------------|----------|
| 1 | `eval_loss = NaN` | `MAX_LENGTH=512` trop court → tous les labels masqués (`-100`) | `MAX_LENGTH=2048` |
| 2 | F1 = 0.0 sur toutes les classes | Le verdict est en fin de texte, mais le parsing ne cherchait que les 200 premiers caractères | Recherche sur le texte complet (`re.search`) |
| 3 | Seulement 2/35 exemples évalués | `max_new_tokens=150` trop court → génération tronquée avant la phrase de verdict | `max_new_tokens=500` |
| 4 | Split non représentatif (2 classes sur 3 en eval) | `train_test_split` aléatoire simple, sans stratification | Split stratifié manuel par classe |
| 5 | `CUDA out of memory` avec `torchrun` | `device_map="auto"` + `torchrun` = double distribution du modèle | Conserver `device_map="auto"` seul (sans torchrun) |
| 6 | Overfitting dès epoch 2-3 | `learning_rate=5e-5` trop agressif pour la taille du dataset | `learning_rate=2e-5` + `lora_dropout=0.10` + `warmup_ratio=0.05` |
| 7 | Répétitions et incohérences dans le texte généré (ex: "ce qui conduit à penser" en boucle) | Conséquence directe du problème #1 (loss NaN ayant corrompu les poids LoRA) | Résolu automatiquement après le fix #1 |

---

## Biais identifié — frontière OUI/PIÈGE

Un test qualitatif sur *Les Deux Pèlerins et l'Huître* (La Fontaine) a révélé que le modèle associe à tort l'absence d'affrontement **physique** au verdict PIÈGE, alors que la définition théorique n'exige aucune matérialisation physique du conflit.

**Cause probable :** les exemples synthétiques PIÈGE généraient fréquemment des formulations du type *"tension sans affrontement physique"*, créant une corrélation statistique absente de la définition théorique.

**Recommandation :** revoir les prompts de génération synthétique pour éviter d'associer systématiquement PIÈGE à l'absence de manifestation physique, et valider un échantillon des exemples synthétiques par un expert avant intégration au dataset d'entraînement.
