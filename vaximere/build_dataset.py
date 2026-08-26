"""Étapes 5-6 — Assemblage du dataset final, équilibrage et contrôles qualité.

Produit le JSONL au format du cahier des charges :

    {"query_id": "Q_001_FR", "texte": "...", "langue": "fra",
     "intention": "UTILITE_VACCIN", "faq_target_id": "FAQ_003"}

+ deux colonnes supplémentaires demandées : `source` et `score` (confiance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import (
    FINAL_DIR,
    INTENTS,
    INTENT_TO_FAQ,
    LANGUES,
    MIN_PER_INTENT_TOTAL,
    RANDOM_SEED,
    make_query_id,
)
from .utils import LOG, save_json, save_jsonl

# Colonnes du schéma final (ordre stable).
OUTPUT_COLUMNS = ["query_id", "texte", "langue", "intention", "faq_target_id", "source", "score"]


# --------------------------------------------------------------------------- #
# Équilibrage par intention
# --------------------------------------------------------------------------- #
def balance_intents(df: pd.DataFrame, per_intent: int, rng: int = RANDOM_SEED) -> pd.DataFrame:
    """Sélectionne jusqu'à `per_intent` exemples par intention.

    Stratégie : ~1/3 de la banque seed (socle garanti) + complément issu des
    sources HF réelles (triées par score décroissant), afin d'allier couverture
    et authenticité. Si une source manque, l'autre comble.
    """
    if df is None or df.empty:
        return df
    parts: list[pd.DataFrame] = []
    for intent in INTENTS:
        g = df[df["intention"] == intent]
        if g.empty:
            LOG.warning("Intention %s : aucun exemple disponible.", intent)
            continue
        seed_g = g[g["source"] == "seed_curated"].sample(frac=1.0, random_state=rng).reset_index(drop=True)
        real_g = g[g["source"] != "seed_curated"].sort_values("score", ascending=False).reset_index(drop=True)

        n_seed = min(len(seed_g), max(1, per_intent // 3))
        picked = pd.concat(
            [seed_g.iloc[:n_seed], real_g.iloc[: max(0, per_intent - n_seed)]],
            ignore_index=True,
        )
        if len(picked) < per_intent:
            need = per_intent - len(picked)
            rest_seed = seed_g.iloc[n_seed:]
            picked = pd.concat([picked, rest_seed.iloc[:need]], ignore_index=True)

        picked = picked.iloc[:per_intent].reset_index(drop=True)
        parts.append(picked)

    out = pd.concat(parts, ignore_index=True)
    LOG.info("Équilibrage : %d exemples (max %d/intention).", len(out), per_intent)
    return out


# --------------------------------------------------------------------------- #
# Assemblage
# --------------------------------------------------------------------------- #
def assemble_rows(fr_df: pd.DataFrame, translations_long: pd.DataFrame) -> pd.DataFrame:
    """Assemble les lignes FR maîtresses et leurs traductions en format long.

    `fr_df` doit contenir `texte`, `intention`, `source`, `score`.
    `translations_long` contient `master_id`, `langue`, `texte`.
    """
    fr = fr_df.reset_index(drop=True).copy()
    fr["master_id"] = fr.index + 1

    # Partie "contenu" : FR + traductions (texte, langue, master_id uniquement).
    fr_part = fr[["master_id", "texte"]].copy()
    fr_part["langue"] = "fra"
    tr = translations_long[["master_id", "texte", "langue"]].copy()
    merged = pd.concat([fr_part, tr], ignore_index=True)

    # Métadonnées (intention/source/score) portées par la question FR maîtresse,
    # jointes via master_id : aucune collision de colonnes possible.
    meta = fr[["master_id", "intention", "source", "score"]]
    merged = merged.merge(meta, on="master_id", how="left")

    merged["faq_target_id"] = merged["intention"].map(INTENT_TO_FAQ)
    merged["query_id"] = [
        make_query_id(int(mid), lang)
        for mid, lang in zip(merged["master_id"], merged["langue"])
    ]
    return merged[OUTPUT_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Contrôles qualité
# --------------------------------------------------------------------------- #
def quality_checks(df: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Retourne (statistiques, liste de problèmes éventuels)."""
    stats: dict[str, Any] = {
        "total_exemples": int(len(df)),
        "nb_intentions_presentes": int(df["intention"].nunique()) if not df.empty else 0,
        "per_intention_total": df.groupby("intention").size().to_dict() if not df.empty else {},
        "per_langue": df.groupby("langue").size().to_dict() if not df.empty else {},
    }
    issues: list[str] = []

    if df.empty:
        issues.append("Dataset vide.")
        return stats, issues

    for intent in INTENTS:
        n = stats["per_intention_total"].get(intent, 0)
        if n < MIN_PER_INTENT_TOTAL:
            issues.append(
                f"Intention {intent} : {n} exemples (< {MIN_PER_INTENT_TOTAL} requis)."
            )
        else:
            LOG.info("Intention %s : %d exemples (OK >= %d).", intent, n, MIN_PER_INTENT_TOTAL)

    # doublons (texte, langue)
    n_dup = int(df.duplicated(subset=["texte", "langue"]).sum())
    stats["doublons"] = n_dup
    if n_dup:
        issues.append(f"{n_dup} doublon(s) détecté(s).")

    # langues attendues
    for lang in LANGUES:
        if stats["per_langue"].get(lang, 0) == 0:
            issues.append(f"Aucun exemple en langue `{lang}`.")

    # cohérence faq_target_id
    bad_faq = int((df["faq_target_id"].isna()).sum())
    stats["faq_target_id_manquants"] = bad_faq
    if bad_faq:
        issues.append(f"{bad_faq} ligne(s) sans faq_target_id.")

    return stats, issues


# --------------------------------------------------------------------------- #
# Écriture des sorties
# --------------------------------------------------------------------------- #
def write_outputs(df: pd.DataFrame, stats: dict[str, Any], out_dir: Path = FINAL_DIR) -> dict[str, Path]:
    """Écrit le JSONL complet, les versions par langue et le rapport de stats."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    n = save_jsonl(df.to_dict(orient="records"), out_dir / "vaximere_qa_cg_train.jsonl")
    LOG.info("JSONL principal écrit : %d lignes.", n)
    paths["full"] = out_dir / "vaximere_qa_cg_train.jsonl"

    for lang in LANGUES:
        sub = df[df["langue"] == lang]
        p = out_dir / f"vaximere_qa_cg_train_{lang}.jsonl"
        save_jsonl(sub.to_dict(orient="records"), p)
        paths[lang] = p

    save_json(stats, out_dir / "stats_report.json")
    paths["stats"] = out_dir / "stats_report.json"

    # version tabulaire pour inspection rapide
    df.to_csv(out_dir / "vaximere_qa_cg_train.csv", index=False)
    paths["csv"] = out_dir / "vaximere_qa_cg_train.csv"
    return paths
