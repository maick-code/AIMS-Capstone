"""Étape 3 — Classification zero-shot des intentions.

Utilise `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` (pipeline zero-shot de
transformers) pour attribuer à chaque question française l'une des 8 intentions.
Seuls les exemples avec un score >= `MIN_SCORE` (0.70) sont conservés.

La classification est faite UNIQUEMENT sur le français : le modèle ne couvre
pas fiablement le lingala / kituba. L'intention attribuée à une question
maîtresse FR est ensuite reportée sur ses traductions (étape 4).
"""

from __future__ import annotations

from typing import Iterable, Optional

from .config import (
    MIN_SCORE,
    USE_FP16,
    ZSC_BATCH_SIZE,
    ZSC_MODEL,
)
from .keywords import CANDIDATE_LABELS, LABEL_TO_INTENT
from .utils import LOG, get_device

# Modèle d'hypothèse : gabarit NLI recommandé pour mDeBERTa-v3 (multilingue).
HYPOTHESIS_TEMPLATE = "This example is {}."


class ZeroShotIntentClassifier:
    """Classifieur zero-shot -> intention, avec import de transformers à la demande."""

    def __init__(self, device: Optional[str] = None, model_name: str = ZSC_MODEL):
        from transformers import pipeline  # import local (évite une dépendance au démarrage)

        self.model_name = model_name
        self.device = device or get_device()
        self._pipeline_kwargs = dict(model=model_name, top_k=1, multi_label=False)
        if "cuda" in self.device:
            self._pipeline_kwargs["device"] = 0
            if USE_FP16:
                import torch

                self._pipeline_kwargs["torch_dtype"] = torch.float16
        else:
            self._pipeline_kwargs["device"] = -1

        LOG.info("Chargement du classifieur zero-shot `%s` sur %s ...", model_name, self.device)
        self.pipe = pipeline("zero-shot-classification", **self._pipeline_kwargs)

    def classify(self, texts: Iterable[str]) -> list[tuple[str, float]]:
        """Retourne une liste de (intention, score) pour chaque texte."""
        texts = list(texts)
        if not texts:
            return []
        results = self.pipe(
            texts,
            candidate_labels=CANDIDATE_LABELS,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
            batch_size=ZSC_BATCH_SIZE,
        )
        out: list[tuple[str, float]] = []
        for r in results:
            label = r["labels"][0]
            score = float(r["scores"][0])
            intent = LABEL_TO_INTENT[label]
            out.append((intent, score))
        return out

    def classify_df(self, df, text_col: str = "texte"):
        """Ajoute les colonnes `intention` et `score` au DataFrame."""
        import pandas as pd

        if df is None or df.empty:
            return df
        preds = self.classify(df[text_col].tolist())
        df = df.copy()
        df["intention"] = [p[0] for p in preds]
        df["score"] = [p[1] for p in preds]
        return df


def filter_by_score(df, min_score: float = MIN_SCORE):
    """Ne garde que les exemples dont le score zero-shot >= min_score."""
    import pandas as pd

    if df is None or df.empty or "score" not in df.columns:
        return df
    before = len(df)
    out = df[df["score"] >= min_score].reset_index(drop=True)
    LOG.info(
        "Seuil de confiance %.2f : %d/%d exemples conservés.", min_score, len(out), before
    )
    return out
