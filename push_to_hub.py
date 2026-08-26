#!/usr/bin/env python3
"""Pousse le dataset VaxiMère-QA-CG vers le Hugging Face Hub.

Prérequis :
    pip install datasets huggingface_hub

Authentification (une seule fois) :
    huggingface-cli login                 # ou, dans Colab :
    from huggingface_hub import notebook_login; notebook_login()

Exemples :
    python push_to_hub.py --repo-id votre-org/vaximere-qa-cg
    python push_to_hub.py --repo-id votre-org/vaximere-qa-cg --private
    python push_to_hub.py --repo-id votre-org/vaximere-qa-cg --dry-run

IMPORTANT (licence) : les traductions lingala/kituba proviennent de NLLB-200
(CC-BY-NC-4.0). Le dataset est donc publié avec la licence `cc-by-nc-4.0`
(usage non commercial). Voir DATA_CARD.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset card (README) générée automatiquement sur le Hub
# --------------------------------------------------------------------------- #
DATASET_CARD = """---
license: cc-by-nc-4.0
task_categories:
  - text-classification
language:
  - fr
  - ln
  - kg
pretty_name: VaxiMère-QA-CG
size_categories:
  - n<1K
tags:
  - vaccination
  - pediatric
  - intent-classification
  - multilingual
  - french
  - lingala
  - kituba
  - congo-brazzaville
  - low-resource-languages
---

# VaxiMère-QA-CG

Dataset d'**intentions** multilingue (français `fra`, lingala `lin`,
kituba/munukutuba `mkw`) sur la **vaccination pédiatrique au Congo-Brazzaville**,
destiné au fine-tuning LoRA d'un classifieur d'intentions (ex. Gemma 3).

## Structure

Chaque exemple est un JSON avec 7 champs :

```json
{
  "query_id": "Q_001_FR",
  "texte": "Pourquoi vacciner mon enfant même s'il semble en bonne santé",
  "langue": "fra",
  "intention": "UTILITE_VACCIN",
  "faq_target_id": "FAQ_001",
  "source": "seed_curated",
  "score": 1.0
}
```

- **720 exemples** = 240 français + 240 lingala + 240 kituba
- **8 intentions** × 90 exemples :

  `UTILITE_VACCIN`, `SECURITE_VACCIN`, `CALENDRIER_RDV`, `RETARD_RATTRAPAGE`,
  `EFFET_SECONDAIRE`, `RUMEUR_CROYANCE`, `LOCALISATION_ACCES`,
  `HORS_DOMAINE_CLINIQUE`

## Construction

1. **Extraction** : `qanastek/frenchmedmcqa` (Apache-2.0), `ANR-MALADES/MediQAl`
   config `oeq` (CC-BY-4.0), complétés par une banque de ~272 questions rédigées
   (`source=seed_curated`).
2. **Filtrage** : mots-clés du domaine vaccination pédiatrique (rougeole, polio,
   BCG, Penta, carnet vaccinal, fièvre après vaccin…).
3. **Classification zero-shot** : `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`,
   seuil de confiance ≥ 0.70.
4. **Traduction** : `facebook/nllb-200-distilled-600M` (CC-BY-NC-4.0) vers
   `lin_Latn` (lingala) et `kon_Latn` (kikongo, utilisé pour simuler le kituba,
   NLLB n'ayant pas de code kituba dédié).

## Licence

- **`cc-by-nc-4.0`** : usage **non commercial**, car les traductions
  lingala/kituba sont dérivées de NLLB-200 (CC-BY-NC-4.0).
- La partie française (seed + FrenchMedMCQA Apache-2.0 + MediQAl CC-BY-4.0)
  serait compatible commercial ; re-traduire avec un modèle permissif
  (ex. MADLAD-400, Apache-2.0) permettrait une licence plus permissive.

## Biais & limites

- Les questions lingala/kituba sont des **traductions automatiques**, pas des
  énoncés natifs ; les tournures idiomatiques locales sont sous-représentées.
- `kon_Latn` (kikongo) **approxime** le kituba/munukutuba.
- La classification zero-shot est automatique : une relecture humaine est
  recommandée malgré le seuil 0.70.
- Les questions seed sont **synthétiques** (modélisation de situations
  plausibles), en particulier pour `RUMEUR_CROYANCE` et `HORS_DOMAINE_CLINIQUE`.

## Éthique

- Aucune donnée personnelle ni identifiant patient.
- Les réponses FAQ sont alignées sur le PEV congolais / OMS et portent la
  mention **« à valider par un médecin »**.
- `HORS_DOMAINE_CLINIQUE` doit déclencher un **transfert vers un agent de
  santé** ; le modèle ne doit jamais formuler d'avis clinique.

*Voir le dépôt source pour le pipeline complet :*
https://github.com/maick-code/AIMS-Capstone
"""

# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_dataset(jsonl_path: Path):
    """Convertit le JSONL en `datasets.Dataset` avec schéma de colonnes explicite."""
    from datasets import Dataset, Features, Value

    rows = load_jsonl(jsonl_path)
    features = Features(
        {
            "query_id": Value("string"),
            "texte": Value("string"),
            "langue": Value("string"),
            "intention": Value("string"),
            "faq_target_id": Value("string"),
            "source": Value("string"),
            "score": Value("float64"),
        }
    )
    return Dataset.from_list(rows, features=features)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Push VaxiMère-QA-CG vers HF Hub")
    parser.add_argument("--repo-id", required=True,
                        help="ex. votre-org/vaximere-qa-cg")
    parser.add_argument("--private", action="store_true", help="dépôt privé (gated)")
    parser.add_argument("--token", default=None, help="token HF (sinon variable HF_TOKEN / cache)")
    parser.add_argument("--jsonl", default="data/vaximere_qa_cg_train.jsonl")
    parser.add_argument("--faq", default="data/faq_validee.json")
    parser.add_argument("--stats", default="data/stats_report.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="ne pousse pas, affiche seulement le contenu")
    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"Fichier introuvable : {jsonl_path}")

    ds = build_dataset(jsonl_path)
    print(f"Dataset prêt : {len(ds)} lignes, colonnes {ds.column_names}")

    if args.dry_run:
        print("\n--- Aperçu (5 premières lignes) ---")
        for row in ds.select(range(min(5, len(ds)))):
            print(json.dumps(row, ensure_ascii=False))
        print("\n(Dry-run : aucun push effectué.)")
        return

    from huggingface_hub import login

    token = args.token or None
    if token:
        login(token=token)

    ds.push_to_hub(
        repo_id=args.repo_id,
        private=args.private,
        token=token,
        commit_message="Add VaxiMère-QA-CG train split (720 ex, fr/lin/kituba)",
    )
    print(f"✅ Dataset poussé : https://huggingface.co/datasets/{args.repo_id}")

    # Upload des fichiers annexes (FAQ, stats, data card)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(Path(args.faq)),
        path_in_repo="faq_validee.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(Path(args.stats)),
        path_in_repo="stats_report.json",
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    # README = dataset card (YAML frontmatter inclus)
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )
    print("✅ faq_validee.json, stats_report.json et dataset card poussés.")


if __name__ == "__main__":
    main()
