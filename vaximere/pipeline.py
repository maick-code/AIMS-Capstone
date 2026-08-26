"""Orchestrateur du pipeline VaxiMère-QA-CG (dataset v2, étapes 0 à 8).

Usage :
    python -m vaximere.pipeline --mode full
    python -m vaximere.pipeline --mode dryrun       # test sans modèle ni réseau
    python -m vaximere.pipeline --mode full --skip-translation

Dataset v2 (refonte après le diagnostic 0.375 de la Phase 1) :
    0. banque seed v2 (`seed_questions_v2.py`, ~54 questions FR/intention) ;
    1. extraction HF ; 2. filtrage ; 3. zero-shot des sources HF ;
    3b. chargement NLLB (partagé) ;
    4. augmentation par back-translation FR->EN->FR (`augment.py`) ;
    5. traduction lingala/kituba ; 6. assemblage ; 7. contrôles ; 8. sorties.
    Équilibrage final : ~100 maîtres FR/intention x 3 langues ~= 2400 exemples.

Le mode `dryrun` utilise des classifieur/traducteur factices (aucun téléchargement
de modèle) et écrit dans `data/dryrun/` : il permet de valider le câblage complet
du pipeline avant l'exécution coûteuse sur Colab.
"""

from __future__ import annotations

import argparse
import hashlib
from typing import Optional

import pandas as pd

from .config import (
    AUGMENT_PARAPHRASES,
    DRYRUN_DIR,
    FINAL_DIR,
    INCLUDE_MASAKHANEWS,
    INCLUDE_VACCINE_TWEETS,
    INTENTS,
    MIN_SCORE,
    RANDOM_SEED,
    TARGET_PER_INTENT_FR,
)
from .utils import (
    LOG,
    Timer,
    dedupe_df,
    ensure_dirs,
    print_intent_distribution,
    save_json,
    set_seed,
)
from .augment import augment_df
from .build_dataset import assemble_rows, balance_intents, quality_checks, write_outputs
from .extract import extract_all
from .faq_validate import build_faq
from .filter_clean import run_filter
from .seed_questions_v2 import build_seed_df, validate_seed_v2


# --------------------------------------------------------------------------- #
# Substituts factices pour le mode dryrun (aucun modèle requis)
# --------------------------------------------------------------------------- #
class _MockClassifier:
    def __init__(self) -> None:
        LOG.info("DRYRUN : classifieur factice (intention déterministe par hash).")

    def classify_df(self, df):
        if df is None or df.empty:
            return df
        df = df.copy()
        df["intention"] = [
            INTENTS[int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % len(INTENTS)]
            for t in df["texte"]
        ]
        df["score"] = 0.90
        return df


class _MockTranslator:
    def __init__(self) -> None:
        LOG.info("DRYRUN : traducteur factice (préfixes [code] factices).")

    def translate(self, texts, src_code=None, tgt_code=None):
        tag = tgt_code or "?"
        return [f"[{tag}] {t}" for t in texts]


def _filter_by_score(df, min_score=MIN_SCORE):
    if df is None or df.empty or "score" not in df.columns:
        return df
    return df[df["score"] >= min_score].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Pipeline principal
# --------------------------------------------------------------------------- #
def run_pipeline(
    mode: str = "full",
    target_per_intent: Optional[int] = None,
    include_masakhanews: Optional[bool] = None,
    include_tweets: Optional[bool] = None,
    skip_translation: bool = False,
) -> dict:
    """Exécute toutes les étapes et retourne le dictionnaire de statistiques."""
    dryrun = mode == "dryrun"
    target = target_per_intent or TARGET_PER_INTENT_FR

    set_seed(RANDOM_SEED)
    ensure_dirs()
    LOG.info("=== Pipeline VaxiMère-QA-CG (mode=%s) ===", mode)

    # ---------------------------------------------------------------- Étape 0
    with Timer("Étape 0 — Banque seed v2"):
        validate_seed_v2()
        seed_df = build_seed_df()
        print_intent_distribution(seed_df, label="seed v2 (avant filtrage)")

    # ---------------------------------------------------------------- Étape 1
    with Timer("Étape 1 — Extraction Hugging Face"):
        real_df = extract_all(include_masakhanews, include_tweets)

    # ---------------------------------------------------------------- Étape 2
    with Timer("Étape 2 — Filtrage & nettoyage"):
        # La banque seed est déjà ciblée : pas de filtre de domaine.
        seed_clean = run_filter(seed_df, apply_domain=False)
        real_clean = run_filter(real_df, apply_domain=True)
        print_intent_distribution(seed_clean, label="seed v2 (après filtrage)")

    # ---------------------------------------------------------------- Étape 3
    with Timer("Étape 3 — Classification zero-shot (sources HF)"):
        if real_clean is None or real_clean.empty:
            real_kept = pd.DataFrame(columns=seed_clean.columns)
            LOG.info("Aucune source HF : classification zero-shot sautée.")
        elif dryrun:
            real_kept = _MockClassifier().classify_df(real_clean)
            real_kept = _filter_by_score(real_kept)
        else:
            from .classify import ZeroShotIntentClassifier, filter_by_score

            classifier = ZeroShotIntentClassifier()
            real_kept = classifier.classify_df(real_clean)
            real_kept = filter_by_score(real_kept)

    # ---------------------------------------------------------------- Étape 3b
    # Chargement du traducteur NLLB une seule fois : il sert à la fois à
    # l'augmentation (back-translation) et à la traduction lin/kt.
    translator = None
    if skip_translation:
        LOG.info("Traduction/augmentation sautées (--skip-translation).")
    else:
        if dryrun:
            translator = _MockTranslator()
        else:
            from .translate import NLLBTranslator

            translator = NLLBTranslator()

    # ---------------------------------------------------------------- Étape 4
    with Timer("Étape 4 — Augmentation par back-translation (FR->EN->FR)"):
        if translator is None:
            seed_aug = seed_clean
        else:
            seed_aug = augment_df(seed_clean, translator, n_paraphrases=AUGMENT_PARAPHRASES)
        print_intent_distribution(seed_aug, label="seed v2 augmentée")

    # --------------------------------------------------- Pool + équilibrage
    with Timer("Pool + équilibrage par intention"):
        pool = pd.concat([seed_aug, real_kept], ignore_index=True)
        pool = dedupe_df(pool)
        balanced_fr = balance_intents(pool, per_intent=target)
        print_intent_distribution(balanced_fr, label="questions FR maîtresses équilibrées")

    # ---------------------------------------------------------------- Étape 5
    translations = pd.DataFrame(columns=["master_id", "langue", "texte"])
    if translator is not None:
        with Timer("Étape 5 — Traduction lingala / kituba"):
            from .translate import translate_df

            translations = translate_df(balanced_fr, translator)

    # ---------------------------------------------------------------- Étape 6
    with Timer("Étape 6 — Assemblage du dataset final"):
        final = assemble_rows(balanced_fr, translations)
        LOG.info("Dataset final : %d lignes.", len(final))

    # ---------------------------------------------------------------- Étape 7
    with Timer("Étape 7 — Contrôles qualité"):
        stats, issues = quality_checks(final)
        stats["mode"] = mode
        stats["target_per_intent_fr"] = target
        # clés "intention|langue" (JSON exige des clés str, pas de tuples)
        stats["exemples_par_intention_et_langue"] = (
            {f"{i}__{l}": int(n) for (i, l), n in final.groupby(["intention", "langue"]).size().items()}
            if not final.empty
            else {}
        )
        if issues:
            for msg in issues:
                LOG.warning("QUALITÉ : %s", msg)
        else:
            LOG.info("Tous les contrôles qualité sont satisfaits.")

    # ---------------------------------------------------------------- Étape 8
    with Timer("Étape 8 — Écriture des sorties"):
        out_dir = DRYRUN_DIR if dryrun else FINAL_DIR
        paths = write_outputs(final, stats, out_dir=out_dir)

        # Bonus FAQ
        faq = build_faq(translator if (not dryrun and not skip_translation) else None)
        faq_path = out_dir / "faq_validee.json"
        save_json(faq, faq_path)
        paths["faq"] = faq_path
        LOG.info("FAQ validée : %d entrées -> %s", len(faq), faq_path)

    LOG.info("=== Terminé. %d exemples / %d intentions / %d langues ===",
             stats["total_exemples"], stats["nb_intentions_presentes"], len(stats["per_langue"]))
    return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline VaxiMère-QA-CG")
    parser.add_argument("--mode", choices=["full", "dryrun"], default="full",
                        help="full = avec modèles HF ; dryrun = test sans modèle/réseau")
    parser.add_argument("--target-per-intent", type=int, default=None,
                        help="nb de questions FR maîtresses par intention (défaut : config)")
    parser.add_argument("--include-masakhanews", action="store_true", default=None,
                        help="activer la source masakhanews (lingala)")
    parser.add_argument("--include-tweets", action="store_true", default=None,
                        help="activer la source tweets vaccinaux (si texte dispo)")
    parser.add_argument("--skip-translation", action="store_true",
                        help="ne pas lancer NLLB (sortie FR seule)")
    args = parser.parse_args(argv)

    run_pipeline(
        mode=args.mode,
        target_per_intent=args.target_per_intent,
        include_masakhanews=args.include_masakhanews if args.include_masakhanews else None,
        include_tweets=args.include_tweets if args.include_tweets else None,
        skip_translation=args.skip_translation,
    )


if __name__ == "__main__":
    main()
