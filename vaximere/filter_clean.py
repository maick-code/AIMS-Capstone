"""Étape 2 — Filtrage par mots-clés + nettoyage + dédoublonnage.

On ne garde que les questions liées à la vaccination pédiatrique (rougeole,
polio, BCG, Penta, carnet vaccinal, fièvre après vaccin, etc.) grâce à la liste
de mots-clés robuste de `keywords.DOMAIN_KEYWORDS`, puis on nettoie (longueur,
casse, espaces) et on supprime les doublons.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import MAX_CHARS, MIN_CHARS
from .keywords import DOMAIN_KEYWORDS, PEDIATRIC_KEYWORDS
from .utils import LOG, clean_question, dedupe_df, is_too_long, is_too_short

# Compilation des motifs une seule fois (insensible à la casse).
_DOMAIN_RE = re.compile("|".join(DOMAIN_KEYWORDS), re.IGNORECASE)
_PED_RE = re.compile("|".join(PEDIATRIC_KEYWORDS), re.IGNORECASE)


def matches_domain(text: str) -> bool:
    """True si le texte contient au moins un mot-clé du domaine vaccination."""
    return bool(_DOMAIN_RE.search(text or ""))


def is_pediatric(text: str) -> bool:
    """True si le texte contient un mot-clé pédiatrique (optionnel)."""
    return bool(_PED_RE.search(text or ""))


def filter_domain(df: pd.DataFrame, text_col: str = "texte") -> pd.DataFrame:
    """Garde uniquement les lignes du domaine vaccination pédiatrique."""
    if df is None or df.empty:
        return df
    before = len(df)
    mask = df[text_col].map(matches_domain)
    out = df[mask].copy()
    out["is_pediatric"] = out[text_col].map(is_pediatric)
    LOG.info("Filtrage domaine : %d/%d lignes conservées.", len(out), before)
    return out


def filter_length(df: pd.DataFrame, text_col: str = "texte") -> pd.DataFrame:
    """Supprime les questions trop courtes ou trop longues."""
    if df is None or df.empty:
        return df
    before = len(df)
    out = df[
        ~df[text_col].map(lambda t: is_too_short(t, MIN_CHARS))
        & ~df[text_col].map(lambda t: is_too_long(t, MAX_CHARS))
    ].copy()
    LOG.info("Filtrage longueur (%d-%d caractères) : %d/%d lignes conservées.",
             MIN_CHARS, MAX_CHARS, len(out), before)
    return out


def clean_column(df: pd.DataFrame, text_col: str = "texte") -> pd.DataFrame:
    """Applique `clean_question` à la colonne texte."""
    df = df.copy()
    df[text_col] = df[text_col].map(clean_question)
    df = df[df[text_col].str.len() > 0]
    return df


def run_filter(df: pd.DataFrame, text_col: str = "texte", apply_domain: bool = True) -> pd.DataFrame:
    """Enchaîne filtrage domaine -> longueur -> nettoyage -> dédoublonnage.

    `apply_domain=False` pour les données déjà ciblées (ex. banque seed), qui ne
    doivent pas être éliminées par le filtre de mots-clés.
    """
    if df is None or df.empty:
        return df
    out = filter_domain(df, text_col) if apply_domain else df.copy()
    out = filter_length(out, text_col)
    out = clean_column(out, text_col)
    before = len(out)
    out = dedupe_df(out, text_col=text_col)
    LOG.info("Dédoublonnage : %d -> %d lignes uniques.", before, len(out))
    return out.reset_index(drop=True)
