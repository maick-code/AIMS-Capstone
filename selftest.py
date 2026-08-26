#!/usr/bin/env python3
"""Auto-test hors-ligne (bibliothèque standard uniquement) — dataset v2.

Valide, SANS pandas / transformers / torch :
  * config (8 intentions, 3 langues, mapping FAQ, query_id) ;
  * mots-clés de domaine + hypothèses zero-shot ;
  * banque seed v2 (>= 50 questions/intention, pas de doublons) ;
  * FAQ validée (40 entrées) ;
  * simulation du pipeline v2 : seed -> back-translation (x2) -> équilibrage
    100/intention -> 3 langues = ~2400 exemples.

Usage : python selftest.py
"""

from __future__ import annotations

import json
import random
import re
import sys

from vaximere.config import (
    INTENTS,
    INTENT_TO_FAQ,
    LANGUES,
    MIN_PER_INTENT_TOTAL,
    TARGET_PER_INTENT_FR,
    make_query_id,
)
from vaximere.faq_validate import FAQ_INTENTS, VALIDATION_STATUS, build_faq
from vaximere.keywords import (
    CANDIDATE_LABELS,
    DOMAIN_KEYWORDS,
    INTENT_HYPOTHESES,
    LABEL_TO_INTENT,
)
from vaximere.seed_questions_v2 import SEED_QUESTIONS_V2, validate_seed_v2
from vaximere.textutils import clean_question, dedupe_texts

_ok = 0
_domain_re = re.compile("|".join(DOMAIN_KEYWORDS), re.IGNORECASE)


def check(label: str, cond: bool) -> None:
    global _ok
    print(f"[{'OK ' if cond else 'FAIL'}] {label}")
    if cond:
        _ok += 1
    else:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# 1. Config & intentions
# --------------------------------------------------------------------------- #
check("8 intentions définies", len(INTENTS) == 8)
check("Mapping FAQ couvre les 8 intentions", set(INTENT_TO_FAQ) == set(INTENTS))
check("3 langues définies (fra/lin/mkw)", set(LANGUES) == {"fra", "lin", "mkw"})
check("query_id format", make_query_id(1, "fra") == "Q_001_FR")
check("cible v2 : 100 maîtres/intention", TARGET_PER_INTENT_FR == 100)

# --------------------------------------------------------------------------- #
# 2. Mots-clés de domaine
# --------------------------------------------------------------------------- #
check("mots-clés de domaine non vides", len(DOMAIN_KEYWORDS) >= 20)
for t in ("quand vacciner mon bébé contre la rougeole ?",
          "le carnet vaccinal de mon enfant est perdu",
          "mon bébé a de la fièvre après le vaccin Penta"):
    check(f"domaine match : {t[:40]}...", bool(_domain_re.search(t)))
check("hors domaine ne matche pas les mots-clés vaccinaux",
      not _domain_re.search("mon bébé convulse, que faire ?"))

# --------------------------------------------------------------------------- #
# 3. Hypothèses zero-shot
# --------------------------------------------------------------------------- #
check("8 hypothèses zero-shot", len(INTENT_HYPOTHESES) == 8)
check("candidate_labels alignés", len(CANDIDATE_LABELS) == 8)
check("mapping label->intention bijectif", set(LABEL_TO_INTENT.values()) == set(INTENTS))

# --------------------------------------------------------------------------- #
# 4. Banque seed v2
# --------------------------------------------------------------------------- #
validate_seed_v2()
texts = [t for t, _ in SEED_QUESTIONS_V2]
intents = [i for _, i in SEED_QUESTIONS_V2]
check(f"banque seed v2 : {len(texts)} questions", len(texts) >= 400)
check("aucun doublon exact", len(texts) == len(set(texts)))
for intent in INTENTS:
    check(f"seed v2 {intent} : >= 50 questions", intents.count(intent) >= 50)

# --------------------------------------------------------------------------- #
# 5. FAQ validée
# --------------------------------------------------------------------------- #
faq = build_faq(translator=None)
check(f"FAQ : {len(faq)} entrées (attendu 40)", len(faq) == 40)
for intent in INTENTS:
    n = sum(1 for e in faq if e["intention"] == intent)
    check(f"FAQ {intent} : 5 entrées", n == 5)
check("statut 'à valider par un médecin'", all(e["statut_validation"] == VALIDATION_STATUS for e in faq))

# --------------------------------------------------------------------------- #
# 6. Simulation (stdlib) du pipeline v2 complet
# --------------------------------------------------------------------------- #
rng = random.Random(42)
cleaned = dedupe_texts([t for t, _ in SEED_QUESTIONS_V2])
by_intent: dict[str, list[str]] = {i: [] for i in INTENTS}
for t, intent in SEED_QUESTIONS_V2:
    ct = clean_question(t)
    if ct in cleaned:
        by_intent[intent].append(ct)

# Étape 4 : back-translation simulée (paraphrase = préfixe distinct)
augmented: dict[str, list[str]] = {}
for intent in INTENTS:
    for t in by_intent[intent]:
        augmented.setdefault(intent, []).append(t)
        augmented[intent].append("Autrement dit : " + t)

# Étape 5 : équilibrage à TARGET_PER_INTENT_FR (100) par intention
masters: list[tuple[str, str]] = []
for intent in INTENTS:
    pool = augmented[intent][:]
    rng.shuffle(pool)
    for t in pool[:TARGET_PER_INTENT_FR]:
        masters.append((t, intent))

check(f"questions FR maîtresses : {len(masters)} (attendu {len(INTENTS) * TARGET_PER_INTENT_FR})",
      len(masters) == len(INTENTS) * TARGET_PER_INTENT_FR)

# Étape 5b : expansion 3 langues
rows = []
for idx, (texte, intent) in enumerate(masters, start=1):
    for lang in LANGUES:
        rows.append({
            "query_id": make_query_id(idx, lang),
            "texte": texte if lang == "fra" else f"[{LANGUES[lang]['suffix']}] {texte}",
            "langue": lang,
            "intention": intent,
            "faq_target_id": INTENT_TO_FAQ[intent],
            "source": "seed_curated",
            "score": 1.0,
        })

check(f"total exemples : {len(rows)} (attendu {len(INTENTS) * TARGET_PER_INTENT_FR * len(LANGUES)})",
      len(rows) == len(INTENTS) * TARGET_PER_INTENT_FR * len(LANGUES))
for intent in INTENTS:
    n = sum(1 for r in rows if r["intention"] == intent)
    check(f"intention {intent} : {n} exemples (>= {MIN_PER_INTENT_TOTAL})", n >= MIN_PER_INTENT_TOTAL)

sample = json.dumps(rows[0], ensure_ascii=False)
check("JSON sérialisable", isinstance(json.loads(sample), dict))
check("champs attendus présents",
      set(rows[0]) == {"query_id", "texte", "langue", "intention", "faq_target_id", "source", "score"})

print(f"\n✓ {_ok} vérifications réussies.")
print(f"  Aperçu : {sample}")
