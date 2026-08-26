"""Helpers partagés par les scripts d'entraînement (bibliothèque standard)."""

from __future__ import annotations

import inspect
from typing import Any


def safe_init(cls, **kwargs: Any):
    """Construit `cls(**kwargs)` en ignorant les kwargs non supportés.

    Les versions récentes de `transformers` / `trl` renomment ou retirent
    régulièrement des arguments (ex. `evaluation_strategy` -> `eval_strategy`,
    `warmup_ratio`, `max_seq_length` -> `max_length`). Ce helper inspecte la
    signature du constructeur et ne transmet que les kwargs valides, ce qui rend
    les scripts compatibles avec plusieurs versions.
    """
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kwargs:
        return cls(**kwargs)
    valid = {k: v for k, v in kwargs.items() if k in params}
    dropped = set(kwargs) - set(valid)
    if dropped:
        print(f"[safe_init] kwargs ignorés (non supportés par cette version) : {sorted(dropped)}")
    return cls(**valid)
