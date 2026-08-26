"""Phase 2 — Petit LLM décodeur + LoRA (instruction tuning) pour les intentions.

Reproduit la cible finale (Gemma 3 + LoRA) avec un petit modèle tenant sur T4 :
par défaut `Qwen/Qwen2.5-0.5B-Instruct` (non gated, Apache-2.0 — ajustable via
`--model-name`). Les modèles Gemma sont "gated" : il faut accepter leur licence
sur le Hub avant de pouvoir les télécharger.

Déroulement :
    1. Splits via `data_prep.prepare()` (mêmes splits que la Phase 1).
    2. Formatage instruction : un texte d'entrée, l'intention comme sortie.
    3. QLoRA (4-bit via bitsandbytes) + TRL `SFTTrainer` si disponible, sinon
       `Trainer` classique sur `peft`.
    4. Évaluation : exact-match de l'intention prédite sur le split de test
       (global + par langue), sauvegardée dans `metrics.json`.
    5. Optionnel : `--push-to-hub` publie l'adaptateur LoRA.

Usage (Colab) :
    python vaximere/training/train_decoder.py \
        --jsonl data/vaximere_qa_cg_train.jsonl \
        --model-name google/gemma-2-2b-it \
        --epochs 3 --push-to-hub --hub-model-id Semence/vaximere-intent-gemma-lora
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Supporte le lancement en script direct OU en module (cf. train_encoder.py).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vaximere.training.common import safe_init  # noqa: E402
from vaximere.training.data_prep import load_jsonl, prepare  # noqa: E402

INTENTS_STR = (
    "UTILITE_VACCIN, SECURITE_VACCIN, CALENDRIER_RDV, RETARD_RATTRAPAGE, "
    "EFFET_SECONDAIRE, RUMEUR_CROYANCE, LOCALISATION_ACCES, HORS_DOMAINE_CLINIQUE"
)

INSTRUCTION = (
    "Classe la question suivante en une seule intention de vaccination pédiatrique. "
    f"Réponds uniquement avec l'une de ces étiquettes : {INTENTS_STR}."
)

EOS = "\n"


def build_input(row: dict) -> str:
    """Texte d'entrée (prompt) pour l'instruction tuning."""
    return (
        "### Instruction:\n" + INSTRUCTION + "\n\n"
        "### Question:\n" + row["texte"] + "\n\n"
        "### Réponse:\n"
    )


def build_target(row: dict) -> str:
    return row["intention"] + EOS


def build_dataset(splits_dir: Path, split: str):
    from datasets import Dataset

    rows = load_jsonl(splits_dir / f"{split}.jsonl")
    return Dataset.from_list(
        [{"text": build_input(r) + build_target(r), "intention": r["intention"]} for r in rows]
    )


def train_lora(
    jsonl_path: Path,
    out_dir: Path,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4,
    max_length: int = 256,
    seed: int = 42,
    use_4bit: bool = True,
) -> dict:
    import torch

    # 1) splits (les mêmes qu'en Phase 1)
    splits_dir = out_dir / "splits"
    prepare(jsonl_path, splits_dir, seed=seed)
    train_ds = build_dataset(splits_dir, "train")
    eval_ds = build_dataset(splits_dir, "val")
    test_rows = load_jsonl(splits_dir / "test.jsonl")

    # 2) tokenizer + modèle quantifié (avec message clair si le modèle est gated)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001 — message explicite pour l'utilisateur
        raise SystemExit(
            f"\n❌ Impossible de charger le tokenizer de `{model_name}`.\n"
            "   Cause fréquente : modèle « gated » (ex. Gemma, Llama) nécessitant\n"
            "   d'accepter la licence sur sa page Hugging Face. Sinon, utilisez un\n"
            "   modèle non gated : --model-name Qwen/Qwen2.5-0.5B-Instruct\n"
            f"   Détail : {exc}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb = None
    if use_4bit and torch.cuda.is_available():
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"\n❌ Impossible de charger le modèle `{model_name}`.\n"
            "   Cause fréquente : modèle « gated » ou quota de téléchargement.\n"
            f"   Détail : {exc}"
        )

    # 3) LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text", "intention"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text", "intention"])

    # 4) entraînement (TRL si dispo, sinon Trainer générique ; repli sur erreur)
    try:
        from trl import SFTConfig, SFTTrainer

        _HAS_TRL = True
    except ImportError:
        _HAS_TRL = False

    trainer_ok = False
    if _HAS_TRL:
        try:
            sft = safe_init(
                SFTConfig,
                output_dir=str(out_dir / "checkpoints"),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=lr,
                logging_steps=5,
                save_strategy="epoch",
                report_to="none",
                max_length=max_length,          # nom actuel (remplace max_seq_length)
                dataset_text_field="text",
                seed=seed,
            )
            trainer = SFTTrainer(
                model=model,
                args=sft,
                train_dataset=build_dataset(splits_dir, "train"),
                tokenizer=tokenizer,
            )
            trainer.train()
            trainer_ok = True
        except Exception as exc:  # noqa: BLE001 — repli propre
            print(f"[train_decoder] SFTTrainer indisponible ({exc}) -> repli sur Trainer générique.")

    if not trainer_ok:
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        args = safe_init(
            TrainingArguments,
            output_dir=str(out_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            seed=seed,
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
        trainer.train()

    model.save_pretrained(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    # 5) évaluation (exact-match de l'intention)
    report = evaluate(model, tokenizer, test_rows)
    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== Résultats (test, exact-match) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def evaluate(model, tokenizer, test_rows: list[dict]) -> dict:
    import torch
    from tqdm import tqdm

    model.eval()
    ok = 0
    per_lang = {}
    per_lang_n = {}
    for row in tqdm(test_rows, desc="eval"):
        prompt = build_input(row)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = gen.strip().split("\n")[0].strip()
        hit = pred == row["intention"]
        ok += int(hit)
        per_lang[row["langue"]] = per_lang.get(row["langue"], 0) + int(hit)
        per_lang_n[row["langue"]] = per_lang_n.get(row["langue"], 0) + 1

    return {
        "exact_match_accuracy": ok / len(test_rows),
        "n_test": len(test_rows),
        "accuracy_par_langue": {l: per_lang[l] / per_lang_n[l] for l in per_lang},
    }


def push_lora(out_dir: Path, hub_model_id: str, model_name: str, metrics: dict):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hub_model_id, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir / "adapter"), repo_id=hub_model_id)
    card = f"""---
library_name: peft
base_model: {model_name}
tags: [intent-classification, vaccination, multilingual, lora]
pipeline_tag: text-generation
---

# {hub_model_id.split('/')[-1]}

Adaptateur LoRA (instruction tuning) pour classifier les intentions de vaccination
pédiatrique (fr / lingala / kituba). Basé sur `{model_name}`, entraîné sur
[Semence/vaximere-qa-cg](https://huggingface.co/datasets/Semence/vaximere-qa-cg).

Exact-match (test) : {metrics.get('exact_match_accuracy')}
"""
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    repo_id=hub_model_id)
    print(f"✅ Adapter poussé : https://huggingface.co/{hub_model_id}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Phase 2 : LLM + LoRA VaxiMère")
    p.add_argument("--jsonl", default="data/vaximere_qa_cg_train.jsonl")
    p.add_argument("--out", default="outputs/decoder_lora")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="modèle décodeur (non gated par défaut ; Gemma/Llama sont gated)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    args = p.parse_args(argv)

    metrics = train_lora(
        Path(args.jsonl), Path(args.out),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        seed=args.seed,
        use_4bit=not args.no_4bit,
    )
    if args.push_to_hub:
        hub_id = args.hub_model_id or "Semence/vaximere-intent-gemma-lora"
        push_lora(Path(args.out), hub_id, args.model_name, metrics)


if __name__ == "__main__":
    main()
