"""Phase 1 — Entraînement d'un classifieur d'intentions encodeur léger.

Backbone par défaut : `cis-lmu/glot500-base` (XLM-R étendu, Apache-2.0), qui
couvre le FRANÇAIS, le LINGALA (`lin`) et le KIKONGO (`kon`, utilisé pour le
kituba) — contrairement à XLM-R standard ou afro-XLM-R.

Déroulement :
    1. `data_prep.prepare()` génère les splits train/val/test SANS fuite
       inter-langues (au niveau de la question maîtresse).
    2. Tokenisation + fine-tuning avec `Trainer` (GPU T4 : quelques minutes).
    3. Évaluation : accuracy globale + par langue, F1 macro, matrice de confusion,
       rapport de classification (sklearn) sauvegardés dans `--out`.
    4. Optionnel : `--push-to-hub` publie le modèle (avec model card) sur le Hub.

Usage (Colab) :
    python vaximere/training/train_encoder.py \
        --jsonl data/vaximere_qa_cg_train.jsonl \
        --model-name cis-lmu/glot500-base \
        --epochs 6 --batch-size 32 --lr 3e-5 \
        --push-to-hub --hub-model-id Semence/vaximere-intent-glot500
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Permet de lancer ce fichier SOIT en module (`python -m vaximere.training.train_encoder`),
# SOIT en script direct (`python vaximere/training/train_encoder.py`) : dans ce
# dernier cas, la racine du dépôt est ajoutée au sys.path pour que les imports
# absolus du package fonctionnent.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vaximere.training.common import safe_init  # noqa: E402
from vaximere.training.data_prep import load_jsonl, prepare  # noqa: E402


# --------------------------------------------------------------------------- #
# Préparation
# --------------------------------------------------------------------------- #
def load_splits(splits_dir: Path):
    """Charge train/val/test et renvoie (datasets HF, label maps)."""
    from datasets import Dataset, DatasetDict

    data = {}
    for split in ("train", "val", "test"):
        rows = load_jsonl(splits_dir / f"{split}.jsonl")
        data[split] = Dataset.from_list(rows)
    ds = DatasetDict(data)

    label_maps = json.loads((splits_dir / "id2label.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in label_maps["id2label"].items()}
    label2id = {v: int(k) for k, v in label_maps["id2label"].items()}
    return ds, id2label, label2id


def encode(examples, tokenizer, label2id, max_length: int):
    tok = tokenizer(
        examples["texte"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    tok["labels"] = [label2id[i] for i in examples["intention"]]
    return tok


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred, id2label):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def train(
    jsonl_path: Path,
    out_dir: Path,
    model_name: str = "cis-lmu/glot500-base",
    max_length: int = 128,
    batch_size: int = 32,
    epochs: int = 8,
    lr: float = 5e-5,
    seed: int = 42,
    keep_all: bool = False,
    freeze_backbone: bool = False,
) -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    # 1) splits
    splits_dir = out_dir / "splits"
    manifest = prepare(jsonl_path, splits_dir, seed=seed, keep_all=keep_all)
    ds, id2label, label2id = load_splits(splits_dir)

    # 2) modèle & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # Option "feature extraction" : gèle tout sauf la tête de classification.
    # Sur un petit jeu (~500 exemples), fine-tuner le backbone entier (394 M)
    # surapprend ; n'entraîner que la tête généralise souvent mieux.
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] Backbone gelé — paramètres entraînables : {n_trainable:,}")

    # 3) tokenisation
    def _encode(examples):
        return encode(examples, tokenizer, label2id, max_length)

    ds = ds.map(_encode, batched=True, remove_columns=["texte", "intention", "langue",
                                                       "query_id", "faq_target_id", "source", "score"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4) entraînement
    args = safe_init(
        TrainingArguments,
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",   # pas de sauvegarde par époque (le "Writing model shards"
        # de ~15 s disparaît) : on sauvegarde le modèle final nous-mêmes à la fin.
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=10,      # remplace warmup_ratio (supprimé dans les versions récentes)
        fp16=True,            # ~2x plus rapide sur GPU ; ignoré si non supporté
        bf16=False,
        seed=seed,
        logging_steps=10,
        report_to="none",
    )
    # `tokenizer` a été renommé `processing_class` dans les transformers récents
    # (et l'ancien nom a fini par être supprimé). On passe les deux via safe_init :
    # le nom non supporté par la version installée est simplement ignoré.
    trainer = safe_init(
        Trainer,
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )
    trainer.train()

    # 5) sauvegarde du modèle final AVANT le calcul des métriques (défensif :
    #    si un calcul échoue, le modèle reste quand même sur disque).
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    # 6) évaluation sur le test (global + par langue).
    #    Les langues sont relues depuis le JSONL brut (ordre identique à ds["test"],
    #    qui a été construit via Dataset.from_list puis .map sans réordonnancement).
    eval_results = trainer.evaluate(ds["test"], metric_key_prefix="test")
    preds_out = trainer.predict(ds["test"])
    y_true = preds_out.label_ids
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    test_langues = [r["langue"] for r in load_jsonl(splits_dir / "test.jsonl")]

    report = _classification_report(y_true, y_pred, id2label, test_langues)
    report["eval_results"] = eval_results
    report["manifest"] = manifest

    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Résultats (test) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _classification_report(y_true, y_pred, id2label, langues: list[str]) -> dict:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    labels_sorted = sorted(set(y_true))
    target_names = [id2label[i] for i in labels_sorted]

    report = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels_sorted, target_names=target_names,
            output_dict=True, zero_division=0,
        ),
    }

    # par langue (les langues sont alignées sur y_true/y_pred, même ordre)
    langues = list(langues)
    report["accuracy_par_langue"] = {}
    for lang in sorted(set(langues)):
        idx = [i for i, l in enumerate(langues) if l == lang]
        report["accuracy_par_langue"][lang] = float(
            accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx])
        )

    # matrice de confusion (CSV)
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        "/tmp/vaximere_confusion_matrix.csv"
    )
    report["confusion_matrix"] = cm.tolist()
    return report


# --------------------------------------------------------------------------- #
# Push Hub
# --------------------------------------------------------------------------- #
def push_to_hub(model_dir: Path, hub_model_id: str, model_name: str, metrics: dict):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    card = _model_card(hub_model_id, model_name, metrics)
    # Pousser modèle et tokenizer séparément (évite l'argument `tokenizer=` de
    # push_to_hub, lui aussi sujet aux changements de signature entre versions).
    model.push_to_hub(hub_model_id, commit_message="Train VaxiMère intent classifier")
    tokenizer.push_to_hub(hub_model_id)
    # model card via README upload
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=hub_model_id,
    )
    print(f"✅ Modèle poussé : https://huggingface.co/{hub_model_id}")


def _model_card(hub_model_id: str, base_model: str, metrics: dict) -> str:
    acc = metrics.get("test_accuracy", "?")
    f1 = metrics.get("test_f1_macro", "?")
    lang = metrics.get("accuracy_par_langue", {})
    return f"""---
license: apache-2.0
language: [fr, ln, kg]
tags: [intent-classification, vaccination, multilingual, lingala, kituba, congo-brazzaville]
metrics: [accuracy, f1]
base_model: {base_model}
pipeline_tag: text-classification
---

# {hub_model_id.split('/')[-1]}

Classifieur d'intentions multilingue (fr / lingala / kituba) sur la vaccination
pédiatrique au Congo-Brazzaville. Backbone : `{base_model}`, fine-tuné sur
[Semence/vaximere-qa-cg](https://huggingface.co/datasets/Semence/vaximere-qa-cg).

## Performances (split de test)
- Accuracy : {acc}
- F1 macro : {f1}
- Par langue : {lang}

## Labels
8 intentions : UTILITE_VACCIN, SECURITE_VACCIN, CALENDRIER_RDV, RETARD_RATTRAPAGE,
EFFET_SECONDAIRE, RUMEUR_CROYANCE, LOCALISATION_ACCES, HORS_DOMAINE_CLINIQUE.

> Modèle de démonstration — à valider avant tout usage clinique.
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Phase 1 : classifieur encodeur VaxiMère")
    p.add_argument("--jsonl", default="data/vaximere_qa_cg_train.jsonl")
    p.add_argument("--out", default="outputs/encoder")
    p.add_argument("--model-name", default="cis-lmu/glot500-base")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-all", action="store_true")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="n'entraîner que la tête de classification (feature extraction)")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--push-only", action="store_true",
                   help="pousse le modèle déjà entraîné (--out/model) sans ré-entraîner")
    p.add_argument("--hub-model-id", default=None)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    hub_id = args.hub_model_id or "Semence/vaximere-intent-glot500"

    if args.push_only:
        model_dir = out_dir / "model"
        if not model_dir.exists():
            raise SystemExit(f"❌ Modèle introuvable : {model_dir}. Entraînez d'abord (sans --push-only).")
        metrics = {}
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        push_to_hub(model_dir, hub_id, args.model_name, metrics)
        return

    metrics = train(
        Path(args.jsonl), out_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        keep_all=args.keep_all,
        freeze_backbone=args.freeze_backbone,
    )
    if args.push_to_hub:
        push_to_hub(out_dir / "model", hub_id, args.model_name, metrics)


if __name__ == "__main__":
    main()
