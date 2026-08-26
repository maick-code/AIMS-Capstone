"""Étape 4 — Traduction FR -> lingala / kituba avec NLLB-200.

Utilise `facebook/nllb-200-distilled-600M` :

* lingala  -> `lin_Latn`
* kituba   -> `kon_Latn` (kikongo, utilisé pour simuler le kituba/munukutuba,
  NLLB-200 n'ayant pas de code dédié au kituba — voir DATA_CARD).

Les traductions sont mises en cache par langue pour éviter de re-traduire.
"""

from __future__ import annotations

from typing import Optional

from .config import (
    LANGUES,
    NLLB_BATCH_SIZE,
    NLLB_MAX_LENGTH,
    NLLB_MODEL,
    USE_FP16,
)
from .utils import LOG, get_device


class NLLBTranslator:
    """Traducteur NLLB-200 avec imports torch/transformers à la demande."""

    def __init__(self, device: Optional[str] = None, model_name: str = NLLB_MODEL):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_name = model_name
        self.device = device or get_device()
        LOG.info("Chargement du traducteur NLLB `%s` sur %s ...", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        import torch

        dtype = torch.float16 if (USE_FP16 and "cuda" in self.device) else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()

    def translate(
        self,
        texts: list[str],
        src_code: str = "fra_Latn",
        tgt_code: str = "lin_Latn",
    ) -> list[str]:
        """Traduit une liste de textes (batching + forced_bos_token_id)."""
        import torch

        self.tokenizer.src_lang = src_code
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_code)
        outputs: list[str] = []
        total = len(texts)
        for i in range(0, total, NLLB_BATCH_SIZE):
            batch = texts[i : i + NLLB_BATCH_SIZE]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    max_length=NLLB_MAX_LENGTH,
                    num_beams=1,
                )
            outputs.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
            if (i // NLLB_BATCH_SIZE) % 10 == 0:
                LOG.info("  traduction %s : %d/%d", tgt_code, min(i + NLLB_BATCH_SIZE, total), total)
        return outputs


def translate_df(fr_df, translator, src: str = "fra", targets: tuple[str, ...] = ("lin", "mkw")):
    """Traduit chaque question maîtresse FR vers les langues cibles.

    Retourne un DataFrame "long" : `master_id, langue, texte`. Le `master_id`
    correspond à l'index (1-based) de la question FR d'origine.
    """
    import pandas as pd

    if fr_df is None or fr_df.empty:
        return pd.DataFrame(columns=["master_id", "langue", "texte"])

    work = fr_df.reset_index(drop=True)
    work["master_id"] = work.index + 1

    parts: list[pd.DataFrame] = []
    src_code = LANGUES[src]["nllb"]
    for tgt in targets:
        tgt_code = LANGUES[tgt]["nllb"]
        LOG.info("Traduction FR -> %s (%s) de %d questions ...", tgt, tgt_code, len(work))
        translated = translator.translate(work["texte"].tolist(), src_code=src_code, tgt_code=tgt_code)
        parts.append(
            pd.DataFrame(
                {
                    "master_id": work["master_id"].tolist(),
                    "langue": tgt,
                    "texte": translated,
                }
            )
        )
    long_df = pd.concat(parts, ignore_index=True)
    LOG.info("Traduction terminée : %d lignes générées.", len(long_df))
    return long_df
