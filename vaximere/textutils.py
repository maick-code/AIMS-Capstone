"""Fonctions de traitement de texte (bibliothèque standard uniquement).

Séparées de `utils.py` (qui dépend de pandas) afin d'être importables et
testables sans dépendances — utile pour les auto-tests hors-ligne.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from .config import NEAR_DUP_JACCARD

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+")


def normalize_text(text: Any) -> str:
    """Normalise les espaces (y compris insécables) et retire les bordures."""
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    return _WS_RE.sub(" ", text).strip()


def clean_question(text: Any) -> str:
    """Nettoie une question : casse, espaces, guillemets et ponctuation isolée."""
    text = normalize_text(text).strip(' \t"\'“”')
    return text.strip(".,;:!? \t")


def token_set(text: str) -> set[str]:
    """Ensemble des tokens (minuscules)."""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Similarité de Jaccard entre deux textes (0 à 1)."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_too_short(text: str, min_chars: int) -> bool:
    return len(text.strip()) < min_chars


def is_too_long(text: str, max_chars: int) -> bool:
    return len(text.strip()) > max_chars


def dedupe_texts(
    texts: Iterable[str], jaccard_threshold: float = NEAR_DUP_JACCARD
) -> list[str]:
    """Supprime les doublons exacts et quasi-doublons d'une liste de textes.

    Conserve la première occurrence et l'ordre d'origine.
    """
    seen: list[str] = []
    for raw in texts:
        t = clean_question(raw)
        if not t:
            continue
        if any(jaccard(t, s) >= jaccard_threshold for s in seen):
            continue
        seen.append(t)
    return seen
