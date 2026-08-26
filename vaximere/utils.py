"""Utilitaires partagés : logging, seed, device, nettoyage, dédoublonnage, I/O."""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import (
    DATA_DIR,
    FINAL_DIR,
    INTERIM_DIR,
    RAW_DIR,
    NEAR_DUP_JACCARD,
)
from .textutils import (  # noqa: F401 — réexportées pour la rétro-compatibilité
    clean_question,
    dedupe_texts,
    is_too_long,
    is_too_short,
    jaccard,
    normalize_text,
    token_set,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "vaximere") -> logging.Logger:
    """Retourne un logger unique avec format lisible (prints clairs requis)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG: logging.Logger = get_logger()


class Timer:
    """Context manager qui mesure et affiche la durée d'une étape."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "Timer":
        self.t0 = time.time()
        LOG.info(">>> %s ...", self.label)
        return self

    def __exit__(self, *exc: Any) -> bool:
        LOG.info("<<< %s terminé en %.1fs", self.label, time.time() - self.t0)
        return False


# --------------------------------------------------------------------------- #
# Reproducibilité & device
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Fixe les graines (random / numpy / torch) pour la reproductibilité."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device() -> str:
    """Retourne `cuda:0` si un GPU est disponible, sinon `cpu`."""
    try:
        import torch

        if torch.cuda.is_available():
            LOG.info("GPU détecté : %s", torch.cuda.get_device_name(0))
            return "cuda:0"
    except ImportError:
        pass
    LOG.info("Pas de GPU détecté : utilisation du CPU (lent mais fonctionnel).")
    return "cpu"


def ensure_dirs() -> None:
    """Crée les dossiers de travail (raw / interim / final / dryrun)."""
    for d in (DATA_DIR, RAW_DIR, INTERIM_DIR, FINAL_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dédoublonnage (version DataFrame ; la version texte est dans textutils.py)
# --------------------------------------------------------------------------- #
def dedupe_df(df, text_col: str = "texte", threshold: float = NEAR_DUP_JACCARD):
    """Version DataFrame du dédoublonnage (conserve la 1re occurrence)."""
    if df.empty:
        return df
    # 1) doublons exacts (rapide)
    df = df.copy()
    df["_norm"] = df[text_col].map(clean_question)
    df = df.drop_duplicates(subset="_norm").reset_index(drop=True)
    # 2) quasi-doublons (Jaccard)
    keep_idx: list[int] = []
    seen: list[str] = []
    for i, row in df.iterrows():
        t = row["_norm"]
        if any(jaccard(t, s) >= threshold for s in seen):
            continue
        seen.append(t)
        keep_idx.append(i)
    return df.loc[keep_idx].drop(columns="_norm").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# I/O JSONL / JSON / cache CSV
# --------------------------------------------------------------------------- #
def save_jsonl(rows: Iterable[dict], path: Path) -> int:
    """Écrit des dictionnaires en JSONL (un objet par ligne, UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_jsonl(path: Path) -> list[dict]:
    """Relit un fichier JSONL."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(obj: Any, path: Path) -> Path:
    """Écrit un objet en JSON indenté (UTF-8, sans échappement d'accents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def save_df(df, path: Path) -> None:
    """Cache un DataFrame au format CSV (pas de dépendance à pyarrow)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.info("Cache écrit : %s (%d lignes)", path, len(df))


def load_df(path: Path) -> Optional[Any]:
    """Relit un cache CSV ; retourne None s'il n'existe pas."""
    path = Path(path)
    if not path.exists():
        return None
    import pandas as pd

    LOG.info("Cache lu : %s", path)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
def print_intent_distribution(df, col: str = "intention", label: str = "") -> None:
    """Affiche la répartition des intentions (prints clairs)."""
    if df is None or df.empty:
        LOG.info("Distribution %s : vide", label)
        return
    counts = df[col].value_counts().sort_index()
    LOG.info("Distribution %s :\n%s", label, counts.to_string())


# --------------------------------------------------------------------------- #
# Chargement Hugging Face (avec dégradation propre)
# --------------------------------------------------------------------------- #
def safe_load_dataset(name: str, config: Optional[str] = None, split: Optional[str] = None):
    """Charge un dataset HF en DataFrame avec gestion d'erreur propre.

    Retourne `None` (au lieu de lever une exception) si `datasets` est absent,
    si le dataset n'existe pas, ou si le réseau est indisponible.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        LOG.warning("`datasets` non installé : impossible de charger %s", name)
        return None
    try:
        kwargs: dict[str, Any] = {"path": name}
        if config:
            kwargs["name"] = config
        if split:
            kwargs["split"] = split
        ds = load_dataset(**kwargs)
    except Exception as exc:  # noqa: BLE001 — on dégrade proprement
        LOG.warning("Échec du chargement de %s : %s", name, exc)
        return None
    try:
        return ds.to_pandas()
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Conversion DataFrame impossible pour %s : %s", name, exc)
        return None
