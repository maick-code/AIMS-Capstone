# Taxonomie des intentions — VaxiMère-QA-CG v2

Règles de décision pour étiqueter (et générer) les questions. À lire **avant**
toute annotation ; les exemples ambigus se tranchent par la **règle d'or** de
chaque paire.

## Les 8 intentions

1. `UTILITE_VACCIN` — pourquoi vacciner, contre quoi
2. `SECURITE_VACCIN` — peur générique, danger, composition
3. `CALENDRIER_RDV` — dates, âges, prochain rendez-vous, nombre de doses
4. `RETARD_RATTRAPAGE` — vaccin manqué/retardé, rattrapage
5. `EFFET_SECONDAIRE` — réaction bénigne **après** le vaccin
6. `RUMEUR_CROYANCE` — rumeur/croyance **rapportée** avec contenu spécifique
7. `LOCALISATION_ACCES` — où, horaires, gratuité, documents
8. `HORS_DOMAINE_CLINIQUE` — urgence/gravité → transfert à un agent de santé

---

## Paire critique 1 — `EFFET_SECONDAIRE` vs `HORS_DOMAINE_CLINIQUE`

**Règle d'or** : `EFFET_SECONDAIRE` = symptôme **bénin** ET **explicitement lié au
vaccin**. Dès qu'il y a **gravité** ou **absence de lien vaccinal** → `HORS_DOMAINE`.

| Critère | EFFET_SECONDAIRE | HORS_DOMAINE_CLINIQUE |
|---|---|---|
| Lien au vaccin | explicite (« après le vaccin », « au point d'injection ») | absent OU explicite |
| Gravité | bénigne (fièvre légère, gonflement, pleurs, fatigue) | grave (convulsions, ne respire plus, inconscient, saigne) |
| Ton | « est-ce normal ? », « que faire ? » | « aidez-moi », « en urgence » |
| Exemple | « Mon bébé a 38 de fièvre après le vaccin, est-ce normal ? » | « Mon bébé convulse, que faire ? » (même post-vaccin) |

> ⚠️ Un symptôme **grave** reste `HORS_DOMAINE` **même s'il est post-vaccinal**
> (ex. « mon enfant convulse après le vaccin » → transfert, pas effet secondaire bénin).

---

## Paire critique 2 — `RUMEUR_CROYANCE` vs `SECURITE_VACCIN`

**Règle d'or** : `RUMEUR_CROYANCE` = une **croyance spécifique rapportée** (avec
une source ou un contenu faux identifié). `SECURITE_VACCIN` = une **peur
générique** exprimée à la première personne.

| Critère | RUMEUR_CROYANCE | SECURITE_VACCIN |
|---|---|---|
| Marqueur de source | « on dit que », « j'ai entendu que », « ma belle-mère dit », « le pasteur a dit » | absent |
| Contenu | **spécifique et faux** : stérilité, poison, complot, puces, sorcellerie, maladie provoquée | **générique** : « dangereux », « sûr », « contient quoi », « j'ai peur » |
| Point de vue | rapporté (tiers) | personnel |
| Exemple | « On dit que le vaccin rend stérile, est-ce vrai ? » | « Le vaccin est-il dangereux pour mon bébé ? » |

---

## Paire critique 3 — `UTILITE_VACCIN` vs `CALENDRIER_RDV`

**Règle d'or** : *pourquoi / contre quoi* → `UTILITE`. *quand / quel âge /
prochain RDV / combien de doses* → `CALENDRIER`.

> « À quel âge se fait le vaccin contre la rougeole ? » = `CALENDRIER`.
> « Le vaccin contre la rougeole protège contre quoi ? » = `UTILITE`.

---

## Paire critique 4 — `RETARD_RATTRAPAGE` vs `CALENDRIER_RDV`

**Règle d'or** : présence d'un **retard, oubli, manque, perte du carnet,
interruption** → `RETARD_RATTRAPAGE`. Question neutre sur une date/âge → `CALENDRIER`.

> « Mon enfant a raté le vaccin de 9 mois, que faire ? » = `RETARD`.
> « Quel vaccin à 9 mois ? » = `CALENDRIER`.

---

## Paire critique 5 — `LOCALISATION_ACCES` vs `CALENDRIER_RDV`

**Règle d'or** : *où / horaires / gratuité / documents / disponibilité* → `ACCES`.
*quel âge / quelle date de vaccin* → `CALENDRIER`.

---

## Rappel : ces règles servent aussi à GÉNÉRER les questions

Chaque question de la banque seed v2 est écrite pour contenir un **marqueur
explicite** de son intention (cf. règles ci-dessus). C'est ce qui a fait défaut
en v1 sur `EFFET_SECONDAIRE` / `RUMEUR_CROYANCE` / `SECURITE_VACCIN`.
