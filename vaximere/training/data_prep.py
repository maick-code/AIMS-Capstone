"""Phase 0 — Préparation des splits train/val/test SANS fuite inter-langues.

Principe : chaque question maîtresse possède 3 lignes (fra, lin, mkw) partageant
le même numéro dans `query_id` (ex. Q_001_FR / Q_001_LN / Q_001_KT). Le split est
donc effectué au niveau de la question MAÎTRESSE (le numéro), puis étendu aux
3 langues : une question et ses traductions restent toujours dans le même split.
Sinon, la traduction d'une question vue à l'entraînement fuirait dans le test et
gonflerait artificiellement les métriques.

Ce module est volontairement 100 % bibliothèque standard, pour pouvoir être
testé hors Colab (voir selftest_split.py à la racine).

Sorties (dans --out, par défaut data/splits/) :
    train.jsonl / val.jsonl / test.jsonl
    all_splits.csv            (colonnes + colonne `split`)
    id2label.json / label2id.json
    split_manifest.json       (répartition, classes, graine, fractions)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #
# Questions maîtresses à écarter : l'énoncé d'examen « Décrivez l'examen
# scanner… » (issu de MediQAl) n'est pas une question de parent et a été classé
# à tort. On le détecte par sous-chaîne sur le texte français, puis on retire
# TOUTES les lignes (fra/lin/mkw) du même numéro de question.
BAD_FR_SUBSTRINGS: tuple[str, ...] = ("Décrivez l'examen scanner",)

DEFAULT_FRACTIONS: tuple[float, float, float] = (0.70, 0.15, 0.15)


# --------------------------------------------------------------------------- #
# Lecture / parsing
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_master_id(query_id: str) -> int:
    """`Q_001_FR` -> 1."""
    m = re.search(r"Q_(\d+)_", query_id)
    if not m:
        raise ValueError(f"query_id inattendu : {query_id!r}")
    return int(m.group(1))


# --------------------------------------------------------------------------- #
# Nettoyage (questions maîtresses hors sujet)
# --------------------------------------------------------------------------- #
def find_bad_master_ids(rows: list[dict], substrings: tuple[str, ...] = BAD_FR_SUBSTRINGS) -> set[int]:
    """Retourne les numéros de questions maîtresses à écarter (via texte FR)."""
    bad: set[int] = set()
    for r in rows:
        if r.get("langue") == "fra" and any(s in (r.get("texte") or "") for s in substrings):
            bad.add(parse_master_id(r["query_id"]))
    return bad


def drop_bad(rows: list[dict]) -> tuple[list[dict], set[int]]:
    bad = find_bad_master_ids(rows)
    if not bad:
        return list(rows), bad
    kept = [r for r in rows if parse_master_id(r["query_id"]) not in bad]
    return kept, bad


# --------------------------------------------------------------------------- #
# Vérifications de structure (lève une exception si la donnée est incohérente)
# --------------------------------------------------------------------------- #
def validate_structure(rows: list[dict]) -> dict[int, str]:
    """Vérifie 1 maître = 3 langues = même intention, et renvoie {master_id: intention}."""
    by_master: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_master[parse_master_id(r["query_id"])].append(r)

    master_intent: dict[int, str] = {}
    for mid, grp in by_master.items():
        langues = sorted(r["langue"] for r in grp)
        intents = {r["intention"] for r in grp}
        if set(langues) != {"fra", "lin", "mkw"}:
            raise ValueError(f"Maître {mid}: langues inattendues {langues}")
        if len(intents) != 1:
            raise ValueError(f"Maître {mid}: intentions incohérentes {intents}")
        master_intent[mid] = intents.pop()
    return master_intent


# --------------------------------------------------------------------------- #
# Split stratifié au niveau des questions maîtresses
# --------------------------------------------------------------------------- #
def split_master_ids(
    master_intent: dict[int, str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[set[int], set[int], set[int]]:
    """Répartit les numéros de questions maîtresses en train/val/test.

    Stratifié par intention : chaque intention est répartie indépendamment avec
    les mêmes proportions, ce qui garantit un équilibre des classes dans chaque
    split. Le reste (arrondi) va au test.
    """
    rng = random.Random(seed)
    train, val, test = set(), set(), set()
    for intent in sorted(set(master_intent.values())):
        ids = sorted(m for m, i in master_intent.items() if i == intent)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(math.floor(n * train_frac))
        n_val = int(math.floor(n * val_frac))
        n_test = n - n_train - n_val
        train.update(ids[:n_train])
        val.update(ids[n_train : n_train + n_val])
        test.update(ids[n_train + n_val :])
    return train, val, test


# --------------------------------------------------------------------------- #
# Pipeline complet
# --------------------------------------------------------------------------- #
def prepare(
    jsonl_path: Path,
    out_dir: Path,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
    keep_all: bool = False,
) -> dict:
    rows = load_jsonl(jsonl_path)
    n_raw = len(rows)

    if keep_all:
        cleaned, dropped = list(rows), set()
    else:
        cleaned, dropped = drop_bad(rows)
    if dropped:
        print(f"[data_prep] {len(dropped)} question(s) maîtresse(s) écartée(s) : {sorted(dropped)}")

    master_intent = validate_structure(cleaned)
    train_ids, val_ids, test_ids = split_master_ids(
        master_intent, fractions[0], fractions[1], seed=seed
    )

    train = [r for r in cleaned if parse_master_id(r["query_id"]) in train_ids]
    val = [r for r in cleaned if parse_master_id(r["query_id"]) in val_ids]
    test = [r for r in cleaned if parse_master_id(r["query_id"]) in test_ids]

    # Cohérence : aucun chevauchement de maîtres entre splits
    sets = [set(), set(), set()]
    for i, part in enumerate((train, val, test)):
        sets[i] = {parse_master_id(r["query_id"]) for r in part}
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Fuites entre splits !"

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    # label maps
    intents = sorted({r["intention"] for r in cleaned})
    id2label = {str(i): intent for i, intent in enumerate(intents)}
    label2id = {intent: i for i, intent in enumerate(intents)}
    write_json({"id2label": id2label, "label2id": label2id}, out_dir / "id2label.json")
    write_json({"label2id": label2id, "id2label": id2label}, out_dir / "label2id.json")

    # CSV combiné (pour inspection)
    fieldnames = ["split", "query_id", "texte", "langue", "intention", "faq_target_id", "source", "score"]
    with (out_dir / "all_splits.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for part, name in ((train, "train"), (val, "val"), (test, "test")):
            for r in part:
                row = dict(r)
                row["split"] = name
                w.writerow(row)

    manifest = {
        "source_jsonl": str(jsonl_path),
        "seed": seed,
        "fractions": list(fractions),
        "dropped_master_ids": sorted(dropped),
        "n_raw_rows": n_raw,
        "n_kept_rows": len(cleaned),
        "n_masters": len(master_intent),
        "n_intents": len(intents),
        "per_split_rows": {"train": len(train), "val": len(val), "test": len(test)},
        "per_split_masters": {
            "train": len(train_ids), "val": len(val_ids), "test": len(test_ids)
        },
        "per_split_langue": {
            name: dict(Counter(r["langue"] for r in part))
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "per_split_intent": {
            name: dict(Counter(r["intention"] for r in part))
            for name, part in (("train", train), ("val", val), ("test", test))
        },
    }
    write_json(manifest, out_dir / "split_manifest.json")

    print("[data_prep] Splits écrits :")
    for name in ("train", "val", "test"):
        print(f"  {name:5s} : {manifest['per_split_rows'][name]:4d} lignes | "
              f"{manifest['per_split_masters'][name]:3d} questions maîtresses | "
              f"langues {manifest['per_split_langue'][name]}")
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Prépare les splits train/val/test de VaxiMère-QA-CG")
    p.add_argument("--jsonl", default="data/vaximere_qa_cg_train.jsonl")
    p.add_argument("--out", default="data/splits")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-all", action="store_true", help="ne pas écarter les questions hors sujet")
    args = p.parse_args(argv)
    prepare(
        Path(args.jsonl),
        Path(args.out),
        fractions=(args.train_frac, args.val_frac, 1.0 - args.train_frac - args.val_frac),
        seed=args.seed,
        keep_all=args.keep_all,
    )


if __name__ == "__main__":
    main()
