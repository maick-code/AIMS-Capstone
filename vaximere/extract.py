"""Étape 1 — Extraction : chargement des sources Hugging Face.

Le dataset `blinoff/medical_qa_fr` demandé initialement n'existe pas (404
vérifié). Il est remplacé par des sources françaises réelles et vérifiées :

* `qanastek/frenchmedmcqa`  — QCM médicaux FR (champ `question`)
* `ANR-MALADES/MediQAl`    — QA ouverte FR, config `oeq` (champs `question`,
  `answer`, `medical_subject`)
* `masakhane/masakhanews`  — config `lin` (lingala), catégorie `health`
  (optionnel, désactivé par défaut car non classable par le ZSC)

Chaque loader renvoie un DataFrame normalisé `[texte, source, langue]` et est
mis en cache (CSV) pour éviter de re-télécharger à chaque exécution.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from .config import (
    FRENCHMEDMCQA,
    INCLUDE_MASAKHANEWS,
    INCLUDE_VACCINE_TWEETS,
    MEDIQAL,
    MEDIQAL_CONFIG,
    MASAKHANEWS,
    MASAKHANEWS_CONFIG,
    RAW_DIR,
)
from .utils import LOG, Timer, load_df, save_df

# Colonnes d'entrée candidates (par priorité) pour identifier le champ texte.
_TEXT_COLUMNS = ("question", "text", "texte", "headline")


def _cached_or_build(name: str, builder: Callable[[], Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    """Lit le cache CSV si présent, sinon construit puis met en cache."""
    cache_path = RAW_DIR / f"{name}.csv"
    cached = load_df(cache_path)
    if cached is not None:
        return cached
    df = builder()
    if df is not None and not df.empty:
        save_df(df, cache_path)
    return df


def _pick_text_column(df: pd.DataFrame) -> Optional[str]:
    """Trouve la première colonne texte disponible."""
    for col in _TEXT_COLUMNS:
        if col in df.columns:
            return col
    return None


def _normalize(df: pd.DataFrame, text_col: str, source: str, langue: str) -> pd.DataFrame:
    """Projette vers le schéma commun `[texte, source, langue]`."""
    out = pd.DataFrame({"texte": df[text_col], "source": source, "langue": langue})
    out["texte"] = out["texte"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# Source 1 : FrenchMedMCQA
# --------------------------------------------------------------------------- #
def load_frenchmedmcqa(split: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Charge FrenchMedMCQA (questions médicales FR, champ `question`)."""
    from .utils import safe_load_dataset  # import local (pandas déjà requis ici)

    if split is None:
        # concatène les splits disponibles (train surtout)
        parts = []
        for s in ("train", "validation", "test"):
            try:
                p = safe_load_dataset(FRENCHMEDMCQA, split=s)
                if p is not None:
                    parts.append(p)
            except Exception:
                pass
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
    else:
        df = safe_load_dataset(FRENCHMEDMCQA, split=split)
        if df is None:
            return None

    col = _pick_text_column(df)
    if col is None:
        LOG.warning("FrenchMedMCQA : aucune colonne texte trouvée, source ignorée.")
        return None
    LOG.info("FrenchMedMCQA : %d questions chargées (colonne `%s`)", len(df), col)
    return _normalize(df, col, source="frenchmedmcqa", langue="fra")


# --------------------------------------------------------------------------- #
# Source 2 : MediQAl (config "oeq" — questions ouvertes)
# --------------------------------------------------------------------------- #
def load_mediqal_oeq(split: Optional[str] = "test") -> Optional[pd.DataFrame]:
    """Charge MediQAl (config `oeq`). Garde `question` (+ `medical_subject`).

    Note : la config `oeq` de MediQAl ne contient qu'un split `test`.
    """
    from .utils import safe_load_dataset

    df = safe_load_dataset(MEDIQAL, config=MEDIQAL_CONFIG, split=split)
    if df is None:
        return None

    col = _pick_text_column(df)
    if col is None:
        LOG.warning("MediQAl : aucune colonne texte trouvée, source ignorée.")
        return None
    out = _normalize(df, col, source="mediqal_oeq", langue="fra")
    if "medical_subject" in df.columns:
        out["medical_subject"] = df["medical_subject"].astype(str)
    LOG.info("MediQAl (oeq) : %d questions chargées", len(df))
    return out


# --------------------------------------------------------------------------- #
# Source 3 (optionnelle) : masakhanews — lingala, catégorie health
# --------------------------------------------------------------------------- #
def load_masakhanews_health(split: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Charge la catégorie `health` de masakhanews (config `lin`).

    Ces textes sont des actualités, pas des questions : ils ne sont PAS classés
    en intentions (le classifieur zero-shot ne couvre pas le lingala). On les
    garde éventuellement comme corpus de référence / pool d'annotation humaine.
    """
    from .utils import safe_load_dataset

    parts = []
    for s in (["train", "validation", "test"] if split is None else [split]):
        p = safe_load_dataset(MASAKHANEWS, config=MASAKHANEWS_CONFIG, split=s)
        if p is not None:
            parts.append(p)
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)

    if "category" in df.columns:
        df = df[df["category"].astype(str).str.lower() == "health"]
        LOG.info("masakhanews (lin) : %d articles health", len(df))
    else:
        LOG.warning("masakhanews : colonne `category` absente, aucune sélection health.")

    col = _pick_text_column(df)
    if col is None or df.empty:
        return None
    return _normalize(df, col, source="masakhanews_lin", langue="lin")


# --------------------------------------------------------------------------- #
# Source 4 (optionnelle) : tweets vaccinaux — essai puis skip propre
# --------------------------------------------------------------------------- #
# Les datasets publics de tweets vaccinaux FR contiennent des identifiants sans
# texte (hydratation Twitter requise). Ce loader vérifie la présence d'une
# colonne texte et se retire proprement sinon.
_VACCINE_TWEET_CANDIDATES = [
    "webimmunization/COVID-19-vaccine-attitude-tweets",
]


def load_vaccine_tweets_optional() -> Optional[pd.DataFrame]:
    """Tente de charger des tweets vaccinaux ; retourne None si pas de texte."""
    from .utils import safe_load_dataset

    for name in _VACCINE_TWEET_CANDIDATES:
        df = safe_load_dataset(name, split="train")
        if df is None:
            continue
        col = _pick_text_column(df)
        if col is None:
            LOG.info(
                "Tweets vaccinaux `%s` : pas de colonne texte (IDs seulement), "
                "source ignorée — l'intention RUMEUR_CROYANCE reste couverte par "
                "la banque seed.", name,
            )
            continue
        LOG.info("Tweets vaccinaux `%s` : %d textes chargés", name, len(df))
        return _normalize(df, col, source=name, langue="fra")
    return None


# --------------------------------------------------------------------------- #
# Extraction globale
# --------------------------------------------------------------------------- #
def extract_all(
    include_masakhanews: Optional[bool] = None,
    include_tweets: Optional[bool] = None,
) -> pd.DataFrame:
    """Extrait toutes les sources et renvoie un pool brut normalisé."""
    if include_masakhanews is None:
        include_masakhanews = INCLUDE_MASAKHANEWS
    if include_tweets is None:
        include_tweets = INCLUDE_VACCINE_TWEETS

    frames: list[pd.DataFrame] = []

    with Timer("Extraction FrenchMedMCQA"):
        f = _cached_or_build("frenchmedmcqa", load_frenchmedmcqa)
        if f is not None:
            frames.append(f)

    with Timer("Extraction MediQAl (oeq)"):
        m = _cached_or_build("mediqal_oeq", load_mediqal_oeq)
        if m is not None:
            frames.append(m)

    if include_masakhanews:
        with Timer("Extraction masakhanews (lin, health)"):
            ln = _cached_or_build("masakhanews_lin_health", load_masakhanews_health)
            if ln is not None:
                frames.append(ln)
    else:
        LOG.info("masakhanews désactivé (INCLUDE_MASAKHANEWS=False).")

    if include_tweets:
        with Timer("Extraction tweets vaccinaux (optionnel)"):
            tw = load_vaccine_tweets_optional()
            if tw is not None:
                frames.append(tw)
    else:
        LOG.info("Tweets vaccinaux désactivés (INCLUDE_VACCINE_TWEETS=False).")

    if not frames:
        LOG.warning("Aucune source HF chargée : le pipeline s'appuiera sur la banque seed.")
        return pd.DataFrame(columns=["texte", "source", "langue"])

    pool = pd.concat(frames, ignore_index=True)
    LOG.info("Pool brut extrait : %d lignes (%d sources distinctes).",
             len(pool), pool["source"].nunique())
    return pool
