# Exemples d'analyses générées

Ces exemples illustrent les sorties du modèle fine-tuné sur des extraits **jamais vus en entraînement**.

---

## Exemple 1 — Moby Dick (Herman Melville)

**Texte soumis :**
> Presque aussitôt, avec une volonté puissante, spontanée, rapide, la Baleine blanche fila comme une flèche dans la mer tumultueuse. Mais lorsque Achab cria à ses canotiers de prendre de nouveaux tours au taberin [...] Tout soudain elle chargea contre sa proue qui avançait, claquant des mâchoires dans une sauvage averse d'écume.

**Analyse générée :**
> Ce passage met en scène un affrontement dramatique entre deux forces naturelles incompatibles : Achab, représentant l'humanité armée, et Moby Dick, incarnation de la force maritime primordiale. L'opposition se manifeste par une dynamique conflictuelle particulièrement intense où les tensions escaladent rapidement depuis l'initiation par Achab jusqu'au moment culminatoire où la baleine attaque directement le navire. [...] En conclusion, ce passage constitue bien un conflit au sens de la définition : **verdict OUI**.

✅ Verdict correct.

---

## Exemple 2 — Les Deux Pèlerins et l'Huître (La Fontaine)

**Texte soumis :**
> Un jour deux Pèlerins sur le sable rencontrent Une Huître que le flot y venait d'apporter [...] Eh bien ! vous l'avez vue, et moi je l'ai sentie.

**Analyse générée :**
> Ce passage présente une situation de compétition pour l'appropriation d'une ressource alimentaire [...] Cependant, l'analyse révèle l'absence d'un élément crucial : l'affrontement ne se matérialise jamais physiquement [...] En conclusion, ce passage constitue un cas limite ou piège analytique : **verdict PIÈGE**.

⚠️ **Verdict incorrect** (devrait être OUI). Voir `docs/RESULTS.md` — section *Biais identifié* pour l'analyse complète de cette erreur, qui révèle une contradiction interne du modèle (il reconnaît un acte physique d'éviction — *"l'un pousse l'autre"* — puis le nie quelques phrases plus loin) et un biais introduit par les données synthétiques associant à tort PIÈGE à l'absence de manifestation physique.

---

## Ce que ces exemples illustrent

- Le modèle généralise correctement sur des conflits clairement dramatisés (Exemple 1).
- Il reste fragile sur les conflits de faible intensité ou purement verbaux, où la frontière avec PIÈGE est ambiguë (Exemple 2).
- Les contradictions internes occasionnelles sont un comportement connu des modèles génératifs séquentiels, à mentionner comme limite du système.
