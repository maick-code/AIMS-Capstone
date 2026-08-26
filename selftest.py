#!/usr/bin/env python3
"""Auto-test hors-ligne (bibliothèque standard uniquement).

Valide la logique pure du pipeline SANS pandas / transformers / torch :
  * cohérence de la banque seed (couverture, doublons, longueur) ;
  * mots-clés de domaine (les questions HORS_DOMAINE ne doivent PAS matcher) ;
  * hypothèses zero-shot (8 intentions, mapping inverse) ;
  * mapping intention -> FAQ et génération des query_id ;
  * FAQ validée (40 entrées, 5 par intention, statut de validation).

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
from vaximere.seed_questions import SEED_QUESTIONS, validate_seed
from vaximere.textutils import clean_question, dedupe_texts

_ok = 0
_domain_re = re.compile("|".join(DOMAIN_KEYWORDS), re.IGNORECASE)


def check(label: str, cond: bool) -> None:
    global _ok
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {label}")
    if cond:
        _ok += 1
    else:
        sys.exit(1)  # fail fast


# --------------------------------------------------------------------------- #
# 1. Config & intentions
# --------------------------------------------------------------------------- #
check("8 intentions définies", len(INTENTS) == 8)
check("Mapping FAQ couvre les 8 intentions", set(INTENT_TO_FAQ) == set(INTENTS))
check("3 langues définies (fra/lin/mkw)", set(LANGUES) == {"fra", "lin", "mkw"})
check("query_id format", make_query_id(1, "fra") == "Q_001_FR")
check("query_id suffixe lingala", make_query_id(12, "lin") == "Q_012_LN")
check("query_id suffixe kituba", make_query_id(99, "mkw") == "Q_099_KT")

# --------------------------------------------------------------------------- #
# 2. Mots-clés de domaine
# --------------------------------------------------------------------------- #
check("mots-clés de domaine non vides", len(DOMAIN_KEYWORDS) >= 20)
domain_pos = [
    "quand vacciner mon bébé contre la rougeole ?",
    "le carnet vaccinal de mon enfant est perdu",
    "mon bébé a de la fièvre après le vaccin Penta",
    "le BCG protège contre la tuberculose",
]
for t in domain_pos:
    check(f"domaine match : {t[:40]}...", bool(_domain_re.search(t)))
# une urgence clinique ne contient aucun mot-clé vaccinal
check("hors domaine ne matche pas les mots-clés vaccinaux",
      not _domain_re.search("mon bébé convulse, que faire ?"))

# --------------------------------------------------------------------------- #
# 3. Hypothèses zero-shot
# --------------------------------------------------------------------------- #
check("8 hypothèses zero-shot", len(INTENT_HYPOTHESES) == 8)
check("candidate_labels alignés", len(CANDIDATE_LABELS) == 8)
check("mapping label->intention bijectif", set(LABEL_TO_INTENT.values()) == set(INTENTS))

# --------------------------------------------------------------------------- #
# 4. Banque seed
# --------------------------------------------------------------------------- #
validate_seed()
texts = [t for t, _ in SEED_QUESTIONS]
intents = [i for _, i in SEED_QUESTIONS]
check(f"banque seed : {len(texts)} questions", len(texts) >= 240)
check("aucun doublon exact", len(texts) == len(set(texts)))
for intent in INTENTS:
    check(f"seed {intent} : >= 30 questions", intents.count(intent) >= 30)

# --------------------------------------------------------------------------- #
# 5. FAQ validée
# --------------------------------------------------------------------------- #
faq = build_faq(translator=None)
check(f"FAQ : {len(faq)} entrées (attendu 40)", len(faq) == 40)
for intent in INTENTS:
    n = sum(1 for e in faq if e["intention"] == intent)
    check(f"FAQ {intent} : 5 entrées", n == 5)
check("faq_id cohérents", all(e["faq_id"] == INTENT_TO_FAQ[e["intention"]] for e in faq))
check("statut 'à valider par un médecin'", all(e["statut_validation"] == VALIDATION_STATUS for e in faq))
check("réponses FR non vides", all(len(e["reponse_fr"]) > 50 for e in faq))

# --------------------------------------------------------------------------- #
# 6. Simulation (stdlib) du pipeline complet : volumes attendus + format JSONL
# --------------------------------------------------------------------------- #
# Reproduit le chemin logique : nettoyage -> dédup -> équilibrage -> 3 langues.
rng = random.Random(42)
cleaned = dedupe_texts([t for t, _ in SEED_QUESTIONS])
by_intent: dict[str, list[str]] = {i: [] for i in INTENTS}
for t, intent in SEED_QUESTIONS:
    ct = clean_question(t)
    if ct in cleaned:  # conserve l'ordre de dédup
        by_intent[intent].append(ct)
# équilibrage : jusqu'à TARGET_PER_INTENT_FR par intention (ordre mélangé)
masters: list[tuple[str, str]] = []  # (texte, intention)
for intent in INTENTS:
    pool = by_intent[intent][:]
    rng.shuffle(pool)
    for t in pool[:TARGET_PER_INTENT_FR]:
        masters.append((t, intent))

check(f"questions FR maîtresses : {len(masters)} (attendu {len(INTENTS) * TARGET_PER_INTENT_FR})",
      len(masters) == len(INTENTS) * TARGET_PER_INTENT_FR)

# expansion en 3 langues + construction des enregistrements JSONL
rows = []
for idx, (texte, intent) in enumerate(masters, start=1):
    for lang in LANGUES:
        rows.append({
            "query_id": make_query_id(idx, lang),
            "texte": f"{texte}" if lang == "fra" else f"[{LANGUES[lang]['suffix']}] {texte}",
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

# format : sérialisation JSON + champs attendus
sample = json.dumps(rows[0], ensure_ascii=False)
check("JSON sérialisable", isinstance(json.loads(sample), dict))
check("champs attendus présents",
      set(rows[0]) == {"query_id", "texte", "langue", "intention", "faq_target_id", "source", "score"})
check("exemple complet FR/LN/KT", {r["langue"] for r in rows[:3]} == set(LANGUES))

print(f"\n✓ {_ok} vérifications réussies.")
print(f"  Aperçu : {sample}")
