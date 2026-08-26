#!/usr/bin/env python3
"""Auto-test du split (bibliothèque standard) sur le dataset v2.

Valide, sans torch/transformers :
  * la banque seed v2 (>= 50/intention) ;
  * `prepare()` : split stratifié 70/15/15 SANS fuite inter-langues, sur un JSONL
    v2 synthétique (1 maître = fra + lin + mkw, intention unique) ;
  * l'équilibre par langue et par intention dans chaque split.

Usage : python selftest_split.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from vaximere.config import INTENTS, LANGUES
from vaximere.seed_questions_v2 import SEED_QUESTIONS_V2, validate_seed_v2
from vaximere.training.data_prep import (
    load_jsonl,
    parse_master_id,
    prepare,
    split_master_ids,
    validate_structure,
    write_jsonl,
)

_ok = 0


def check(label: str, cond: bool) -> None:
    global _ok
    print(f"[{'OK ' if cond else 'FAIL'}] {label}")
    if cond:
        _ok += 1
    else:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# 1. Banque seed v2
# --------------------------------------------------------------------------- #
validate_seed_v2()
texts = [t for t, _ in SEED_QUESTIONS_V2]
intents = [i for _, i in SEED_QUESTIONS_V2]
n_masters = len(texts)
check(f"seed v2 : {n_masters} questions maîtresses", n_masters >= 400)
cnt = Counter(intents)
check("8 intentions, >= 50 chacune", all(cnt[i] >= 50 for i in INTENTS))

# --------------------------------------------------------------------------- #
# 2. Construction d'un JSONL v2 synthétique (1 maître -> 3 langues)
# --------------------------------------------------------------------------- #
tmp = Path("/tmp/vaximere_v2_synth.jsonl")
rows = []
for idx, (texte, intent) in enumerate(SEED_QUESTIONS_V2, start=1):
    for lang, meta in LANGUES.items():
        suffix = meta["suffix"]
        rows.append({
            "query_id": f"Q_{idx:03d}_{suffix}",
            "texte": texte if lang == "fra" else f"[{suffix}] {texte}",
            "langue": lang,
            "intention": intent,
            "faq_target_id": "FAQ_001",
            "source": "seed_curated",
            "score": 1.0,
        })
write_jsonl(rows, tmp)
check(f"JSONL v2 synthétique : {len(rows)} lignes ({n_masters} maîtres x 3 langues)",
      len(rows) == n_masters * 3)

# --------------------------------------------------------------------------- #
# 3. Structure
# --------------------------------------------------------------------------- #
loaded = load_jsonl(tmp)
master_intent = validate_structure(loaded)
check("structure valide (1 maître = fra+lin+mkw, intention unique)", True)
check(f"{n_masters} maîtres détectés", len(master_intent) == n_masters)

# --------------------------------------------------------------------------- #
# 4. Split stratifié sans fuite
# --------------------------------------------------------------------------- #
tr, va, te = split_master_ids(master_intent, train_frac=0.70, val_frac=0.15, seed=42)
check("aucun chevauchement de maîtres entre splits", not (tr & va or tr & te or va & te))
check("tous les maîtres répartis", len(tr) + len(va) + len(te) == n_masters)

for intent in INTENTS:
    n = cnt[intent]
    n_tr = sum(1 for m in tr if master_intent[m] == intent)
    n_va = sum(1 for m in va if master_intent[m] == intent)
    n_te = sum(1 for m in te if master_intent[m] == intent)
    check(f"{intent}: train={n_tr} val={n_va} test={n_te} (sur {n})",
          n_tr + n_va + n_te == n and n_tr > n_va > 0 and n_te > 0 and n_tr > n_te)

# --------------------------------------------------------------------------- #
# 5. Pipeline prepare() complet + fichiers de sortie
# --------------------------------------------------------------------------- #
out = Path("/tmp/vaximere_v2_splits")
manifest = prepare(tmp, out, fractions=(0.70, 0.15, 0.15), seed=42)
check("manifest : tous les maîtres conservés", manifest["n_masters"] == n_masters)
check("somme train+val+test = 3 x maîtres",
      sum(manifest["per_split_rows"].values()) == n_masters * 3)
for name in ("train", "val", "test"):
    check(f"{name}.jsonl écrit", (out / f"{name}.jsonl").exists())
    check(f"{name} : 3 langues équilibrées",
          set(manifest["per_split_langue"][name].values()) == {manifest["per_split_masters"][name]})
check("id2label.json / split_manifest.json écrits",
      (out / "id2label.json").exists() and (out / "split_manifest.json").exists())

print(f"\n✓ {_ok} vérifications réussies.")
