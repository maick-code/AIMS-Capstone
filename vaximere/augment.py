"""Étape 4b — Augmentation par paraphrase (back-translation NLLB-200).

Le dataset v1 plafonnait car il n'avait que ~21 patrons français *distincts*
par intention (les traductions lin/kt étant des miroirs). La back-translation
(FR -> EN -> FR) produit des paraphrases naturelles qui conservent l'intention
mais diversifient le vocabulaire et la syntaxe — une technique éprouvée pour
augmenter les jeux de données en langues peu dotées.

Chaque question maîtresse FR est ainsi doublée (original + paraphrase), puis les
quasi-doublons sont éliminés. Les paraphrases héritent de l'intention et de la
source de la question d'origine.
"""

from __future__ import annotations

from .utils import LOG, dedupe_df


def backtranslate(translator, texts: list[str]) -> list[str]:
    """FR -> EN -> FR via NLLB (paraphrase par pivot anglais)."""
    en = translator.translate(texts, src_code="fra_Latn", tgt_code="eng_Latn")
    fr = translator.translate(en, src_code="eng_Latn", tgt_code="fra_Latn")
    return fr


def augment_df(fr_df, translator, n_paraphrases: int = 1, max_length: int = 300):
    """Retourne le DataFrame FR d'origine + ses paraphrases back-translatées.

    Les paraphrases gardent `intention`, `source` (suffixée `_paraphrase`) et
    `score` de la question d'origine. Les textes vides/trop longs sont écartés.
    """
    import pandas as pd

    if fr_df is None or fr_df.empty:
        return fr_df
    if translator is None:
        LOG.warning("Augmentation sautée : aucun traducteur fourni.")
        return fr_df

    parts = [fr_df.copy()]
    work = fr_df.copy()
    for k in range(n_paraphrases):
        LOG.info("Back-translation FR->EN->FR (passe %d/%d) sur %d questions ...",
                 k + 1, n_paraphrases, len(work))
        paraphrased = backtranslate(translator, work["texte"].tolist())
        par = work.copy()
        par["texte"] = paraphrased
        par["source"] = par["source"].astype(str) + "_paraphrase"
        par = par[par["texte"].str.len().between(1, max_length)]
        parts.append(par)

    out = pd.concat(parts, ignore_index=True)
    out = dedupe_df(out, text_col="texte")
    LOG.info("Augmentation : %d -> %d questions FR (après dédup).", len(fr_df), len(out))
    return out.reset_index(drop=True)
