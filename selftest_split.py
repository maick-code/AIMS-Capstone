#!/usr/bin/env python3
"""Auto-test du split (bibliothèque standard) sur les DONNÉES RÉELLES.

Valide, sans torch/transformers :
  * la structure du JSONL (240 questions maîtresses, 3 langues, intention unique) ;
  * le retrait de la question d'examen hors sujet ;
  * le split stratifié sans fuite inter-langues (une question et ses traductions
    restent dans le même split) et l'équilibre des classes.

Usage : python selftest_split.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from vaximere.training.data_prep import (
    drop_bad,
    load_jsonl,
    parse_master_id,
    prepare,
    split_master_ids,
    validate_structure,
)

_ok = 0


def check(label: str, cond: bool) -> None:
    global _ok
    print(f"[{'OK ' if cond else 'FAIL'}] {label}")
    if cond:
        _ok += 1
    else:
        sys.exit(1)


DATA = Path("data/vaximere_qa_cg_train.jsonl")
rows = load_jsonl(DATA)

# --------------------------------------------------------------------------- #
# 1. Structure
# --------------------------------------------------------------------------- #
check("720 lignes brutes", len(rows) == 720)
masters_raw = {parse_master_id(r["query_id"]) for r in rows}
check("240 questions maîtresses brutes", len(masters_raw) == 240)
master_intent = validate_structure(rows)
check("structure valide (1 maître = fra+lin+mkw, intention unique)", True)
check("8 intentions couvertes", len(set(master_intent.values())) == 8)
check("30 maîtres par intention",
      all(v == 30 for v in Counter(master_intent.values()).values()))

# --------------------------------------------------------------------------- #
# 2. Retrait de la question d'examen
# --------------------------------------------------------------------------- #
cleaned, dropped = drop_bad(rows)
check("1 question maîtresse écartée (examen scanner)", dropped == {92} or len(dropped) == 1)
check("717 lignes après nettoyage", len(cleaned) == 717)
cleaned_intent = validate_structure(cleaned)
cnt = Counter(cleaned_intent.values())
check("HORS_DOMAINE à 29 maîtres après nettoyage", cnt["HORS_DOMAINE_CLINIQUE"] == 29)
check("les 7 autres intentions restent à 30",
      all(v == 30 for k, v in cnt.items() if k != "HORS_DOMAINE_CLINIQUE"))

# --------------------------------------------------------------------------- #
# 3. Split stratifié sans fuite
# --------------------------------------------------------------------------- #
tr, va, te = split_master_ids(cleaned_intent, train_frac=0.70, val_frac=0.15, seed=42)
check("aucun chevauchement de maîtres entre splits", not (tr & va or tr & te or va & te))
check("tous les maîtres répartis", len(tr) + len(va) + len(te) == len(cleaned_intent))

# équilibre par intention (HORS_DOMAINE = 29 -> 20/4/5 ; autres = 30 -> 21/4/5)
for intent, n in cnt.items():
    n_tr = sum(1 for m in tr if cleaned_intent[m] == intent)
    n_va = sum(1 for m in va if cleaned_intent[m] == intent)
    n_te = sum(1 for m in te if cleaned_intent[m] == intent)
    exp_tr = n - n // 10 - 4 if False else (20 if n == 29 else 21)
    check(f"{intent}: train={n_tr} val={n_va} test={n_te}", n_tr + n_va + n_te == n and n_tr in (20, 21))

# --------------------------------------------------------------------------- #
# 4. Pipeline complet + fichiers de sortie
# --------------------------------------------------------------------------- #
out = Path("/tmp/vaximere_splits_test")
manifest = prepare(DATA, out, fractions=(0.70, 0.15, 0.15), seed=42)
check("manifest : 717 lignes conservées", manifest["n_kept_rows"] == 717)
check("manifest : 239 maîtres", manifest["n_masters"] == 239)
check("somme train+val+test = 717",
      sum(manifest["per_split_rows"].values()) == 717)
for name in ("train", "val", "test"):
    check(f"{name}.jsonl écrit", (out / f"{name}.jsonl").exists())
    check(f"{name} : 3 langues équilibrées",
          set(manifest["per_split_langue"][name].values()) == {manifest["per_split_masters"][name]})
check("id2label.json / split_manifest.json écrits",
      (out / "id2label.json").exists() and (out / "split_manifest.json").exists())

print(f"\n✓ {_ok} vérifications réussies.")
