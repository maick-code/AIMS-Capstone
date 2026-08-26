"""Sous-package d'entraînement / évaluation du classifieur VaxiMère-QA-CG.

Contient :
    data_prep.py     — split train/val/test sans fuite inter-langues (stdlib)
    train_encoder.py — Phase 1 : classifieur encodeur (backbone multilingue)
    train_decoder.py — Phase 2 : petit LLM décodeur + LoRA (instruction tuning)
"""
