# VaxiMère-QA-CG

Pipeline reproductible de construction du dataset **VaxiMère-QA-CG** : un
dataset d'intentions multilingue (français `fra`, lingala `lin`, kituba `mkw`)
sur la **vaccination pédiatrique au Congo-Brazzaville**, prêt pour le fine-tuning
**LoRA** d'un classifieur d'intentions (Gemma 3).

- **~700–800 exemples** (8 intentions × ~30 questions FR maîtresses × 3 langues)
- **8 intentions** : `UTILITE_VACCIN`, `SECURITE_VACCIN`, `CALENDRIER_RDV`,
  `RETARD_RATTRAPAGE`, `EFFET_SECONDAIRE`, `RUMEUR_CROYANCE`,
  `LOCALISATION_ACCES`, `HORS_DOMAINE_CLINIQUE`
- Sortie JSONL au format instruction-tuning + FAQ validée + DATA_CARD

## Arborescence

```
vaximere/
├── config.py          # chemins, seuils, modèles, intentions, langues
├── utils.py           # logging, nettoyage, dédoublonnage, I/O JSONL/CSV
├── keywords.py        # mots-clés de domaine + hypothèses zero-shot
├── seed_questions.py  # ~270 questions FR rédigées (socle garanti)
├── extract.py         # Étape 1 : chargement Hugging Face (+ cache)
├── filter_clean.py    # Étape 2 : filtrage mots-clés + nettoyage + dédup
├── classify.py        # Étape 3 : zero-shot (mDeBERTa-v3, seuil 0.70)
├── translate.py       # Étape 4 : NLLB-200 FR -> lin_Latn / kon_Latn
├── build_dataset.py   # Étapes 5-6 : assemblage, équilibrage, qualité
├── faq_validate.py    # Bonus : FAQ 40 entrées « à valider par un médecin »
└── pipeline.py        # Orchestrateur (étapes 0 à 7)
run_pipeline.py        # Point d'entrée CLI
selftest.py            # Auto-test hors-ligne (stdlib seule)
VaxiMere_Colab.ipynb   # Notebook Colab
DATA_CARD.md           # Sources / méthode / biais / limites / éthique
```

## Démarrage rapide

### Google Colab (recommandé, GPU T4)

Ouvrir `VaxiMere_Colab.ipynb`, ou exécuter :

```bash
pip install -r requirements.txt
python run_pipeline.py --mode dryrun   # test rapide (aucun modèle téléchargé)
python run_pipeline.py --mode full     # pipeline complet (modèles HF)
```

### Local

```bash
python selftest.py                    # validation hors-ligne (stdlib)
python run_pipeline.py --mode full    # nécessite torch/transformers/datasets
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
considérations éthiques.
