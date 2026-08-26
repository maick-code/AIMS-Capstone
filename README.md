# VaxiMère-QA-CG

Pipeline reproductible de construction du dataset **VaxiMère-QA-CG** : un
dataset d'intentions multilingue (français `fra`, lingala `lin`, kituba `mkw`)
sur la **vaccination pédiatrique au Congo-Brazzaville**, prêt pour le fine-tuning
**LoRA** d'un classifieur d'intentions (Gemma 3).

## Dataset

- **8 intentions** : `UTILITE_VACCIN`, `SECURITE_VACCIN`, `CALENDRIER_RDV`,
  `RETARD_RATTRAPAGE`, `EFFET_SECONDAIRE`, `RUMEUR_CROYANCE`,
  `LOCALISATION_ACCES`, `HORS_DOMAINE_CLINIQUE`
- **Volume v2** : ~100 questions FR maîtresses/intention × 3 langues ≈ **2 400 exemples**
- **Méthode** : banque seed rédigée (seed v1 + v2, ~54/intention) + augmentation
  par back-translation NLLB (FR→EN→FR) + traductions lingala/kituba
- Sortie JSONL + FAQ validée + DATA_CARD

## Arborescence

```
vaximere/
├── config.py            # chemins, seuils, modèles, intentions, langues
├── utils.py             # logging, dédoublonnage, I/O JSONL/CSV
├── textutils.py         # nettoyage/dédoublonnage (stdlib)
├── keywords.py          # mots-clés de domaine + hypothèses zero-shot
├── seed_questions.py    # banque seed v1 (~34 questions/intention)
├── seed_questions_v2.py # banque seed v2 (v1 + ~20/intention ≈ 54/intention)
├── intent_rubric.md     # taxonomie v2 + règles d'or des paires ambiguës
├── extract.py           # Étape 1 : chargement Hugging Face
├── filter_clean.py      # Étape 2 : filtrage mots-clés + nettoyage
├── classify.py          # Étape 3 : zero-shot (mDeBERTa-v3, seuil 0.70)
├── augment.py           # Étape 4 : back-translation FR→EN→FR (NLLB)
├── translate.py         # Étape 5 : NLLB FR → lin_Latn / kon_Latn
├── build_dataset.py     # Étapes 6-7 : assemblage, équilibrage, qualité
├── faq_validate.py      # Bonus : FAQ 40 entrées « à valider par un médecin »
├── pipeline.py          # Orchestrateur (étapes 0 à 8)
└── training/            # Fine-tuning du petit modèle (Phase 1 & 2)
    ├── data_prep.py     #   split sans fuite (70/15/15 stratifié)
    ├── train_encoder.py #   Phase 1 : classifieur encodeur (glot500-base)
    └── train_decoder.py #   Phase 2 : petit LLM + LoRA (Qwen2.5-0.5B)
run_pipeline.py          # Point d'entrée CLI du pipeline
push_to_hub.py           # Publication du dataset sur Hugging Face
selftest.py              # Auto-test hors-ligne (45 vérifs, stdlib)
selftest_split.py        # Auto-test du split sans fuite (24 vérifs, stdlib)
VaxiMere_Colab.ipynb     # Notebook Colab : génération + entraînement + push HF
DATA_CARD.md             # Sources / méthode / biais / limites / éthique
```

## Démarrage rapide

### Google Colab (recommandé, GPU T4)

Ouvrir **`VaxiMere_Colab.ipynb`** : il installe les dépendances, clone le dépôt,
génère le dataset v2 (~2 400 exemples), entraîne le petit modèle (Phase 1
encodeur `glot500-base` + Phase 2 LoRA `Qwen2.5-0.5B`), puis pousse le dataset
et le modèle sur Hugging Face (cellules optionnelles).

### Local

```bash
pip install -r requirements.txt        # pipeline
pip install -r requirements_train.txt  # entraînement (en plus du précédent)
python selftest.py                     # validation hors-ligne (stdlib)
python run_pipeline.py --mode dryrun   # test rapide (aucun modèle téléchargé)
python run_pipeline.py --mode full     # pipeline complet (modèles HF)
```

## Sorties (`data/final/`)

| Fichier | Description |
|---|---|
| `vaximere_qa_cg_train.jsonl` | Dataset complet (3 langues) |
| `vaximere_qa_cg_train_{fra,lin,mkw}.jsonl` | Versions par langue |
| `vaximere_qa_cg_train.csv` | Version tabulaire |
| `faq_validee.json` | 40 réponses de référence (à valider) |
| `stats_report.json` | Statistiques et contrôles qualité |

## Schéma d'un exemple

```json
{
  "query_id": "Q_001_FR",
  "texte": "À quel âge mon bébé doit-il recevoir le premier vaccin ?",
  "langue": "fra",
  "intention": "CALENDRIER_RDV",
  "faq_target_id": "FAQ_003",
  "source": "seed_curated",
  "score": 1.0
}
```

Voir `DATA_CARD.md` pour les sources, les biais linguistiques, les limites et les
considérations éthiques. Licence du dataset : `cc-by-nc-4.0`.
