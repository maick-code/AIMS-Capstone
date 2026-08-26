"""Configuration centrale du pipeline VaxiMère-QA-CG.

Toutes les constantes du projet (chemins, modèles, seuils, intentions, langues)
sont regroupées ici afin de garantir la reproductibilité et de faciliter
l'adaptation au contexte Google Colab (GPU T4).

Ce module ne dépend que de la bibliothèque standard : il est importable même
sans `pandas` / `transformers` (utile pour les tests hors-ligne).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
# Racine du dépôt = parent du dossier `vaximere/`.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"          # extraits bruts (cache)
INTERIM_DIR: Path = DATA_DIR / "interim"  # étapes intermédiaires (cache)
FINAL_DIR: Path = DATA_DIR / "final"      # sorties livrables (jsonl/json)
DRYRUN_DIR: Path = DATA_DIR / "dryrun"    # sorties du mode --dryrun (test)

# --------------------------------------------------------------------------- #
# Les 8 intentions (ordre canonique du cahier des charges)
# --------------------------------------------------------------------------- #
INTENTS: tuple[str, ...] = (
    "UTILITE_VACCIN",
    "SECURITE_VACCIN",
    "CALENDRIER_RDV",
    "RETARD_RATTRAPAGE",
    "EFFET_SECONDAIRE",
    "RUMEUR_CROYANCE",
    "LOCALISATION_ACCES",
    "HORS_DOMAINE_CLINIQUE",
)

# Mapping intention -> identifiant de FAQ validée (utilisé par `faq_target_id`
# dans le JSONL et par `faq_validee.json`).
INTENT_TO_FAQ: dict[str, str] = {
    "UTILITE_VACCIN": "FAQ_001",
    "SECURITE_VACCIN": "FAQ_002",
    "CALENDRIER_RDV": "FAQ_003",
    "RETARD_RATTRAPAGE": "FAQ_004",
    "EFFET_SECONDAIRE": "FAQ_005",
    "RUMEUR_CROYANCE": "FAQ_006",
    "LOCALISATION_ACCES": "FAQ_007",
    "HORS_DOMAINE_CLINIQUE": "FAQ_008",
}

# --------------------------------------------------------------------------- #
# Langues
# --------------------------------------------------------------------------- #
# code interne -> libellé, code NLLB-200, suffixe des query_id.
#
# Note importante (à reporter dans la DATA_CARD) : NLLB-200 ne possède pas de
# code dédié au kituba/munukutuba ; on utilise le kikongo `kon_Latn` comme
# approximation, conformément au cahier des charges.
LANGUES: dict[str, dict[str, str]] = {
    "fra": {"label": "Français", "nllb": "fra_Latn", "suffix": "FR"},
    "lin": {"label": "Lingala", "nllb": "lin_Latn", "suffix": "LN"},
    "mkw": {"label": "Kituba/Munukutuba", "nllb": "kon_Latn", "suffix": "KT"},
}


def make_query_id(master_id: int, lang: str) -> str:
    """Construit un identifiant du type `Q_001_FR` (voir cahier des charges)."""
    if lang not in LANGUES:
        raise ValueError(f"Langue inconnue : {lang}")
    return f"Q_{int(master_id):03d}_{LANGUES[lang]['suffix']}"


# --------------------------------------------------------------------------- #
# Modèles Hugging Face
# --------------------------------------------------------------------------- #
ZSC_MODEL: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  # zero-shot classification
NLLB_MODEL: str = "facebook/nllb-200-distilled-600M"        # traduction FR -> lin / kon

# --------------------------------------------------------------------------- #
# Seuils et objectifs
# --------------------------------------------------------------------------- #
MIN_SCORE: float = 0.70          # score zero-shot minimal pour garder un exemple
MIN_CHARS: int = 15              # longueur minimale d'une question (nettoyage)
MAX_CHARS: int = 400             # longueur maximale d'une question (nettoyage)
TARGET_PER_INTENT_FR: int = 30   # questions FR maîtresses par intention
MIN_PER_INTENT_TOTAL: int = 70   # minimum d'exemples par intention (3 langues)
RANDOM_SEED: int = 42            # graine globale (reproductibilité)
NEAR_DUP_JACCARD: float = 0.90   # seuil de Jaccard pour détecter les quasi-doublons

# --------------------------------------------------------------------------- #
# Sources Hugging Face
# --------------------------------------------------------------------------- #
# `blinoff/medical_qa_fr` n'existe pas (404 vérifié) : on le remplace par des
# sources françaises réelles et vérifiées, complétées par une banque de
# questions rédigées (seed_questions.py) qui garantit la couverture des
# 8 intentions.
FRENCHMEDMCQA: str = "qanastek/frenchmedmcqa"   # QCM médicaux FR (Apache-2.0)
MEDIQAL: str = "ANR-MALADES/MediQAl"            # QA ouverte FR, config "oeq" (cc-by-4.0)
MEDIQAL_CONFIG: str = "oeq"
MASAKHANEWS: str = "masakhane/masakhanews"      # actualités multilingues (AFL-3.0)
MASAKHANEWS_CONFIG: str = "lin"                 # config lingala

# Loaders optionnels. Les datasets publics de tweets vaccinaux en français sont
# soit en anglais, soit des identifiants Twitter sans texte (hydratation
# nécessaire). Ils sont donc désactivés par défaut et documentés dans la
# DATA_CARD ; la banque seed couvre l'intention RUMEUR_CROYANCE.
INCLUDE_MASAKHANEWS: bool = False   # natif lingala : non classable (pas de couverture ZSC)
INCLUDE_VACCINE_TWEETS: bool = False  # tenté puis ignoré s'il n'y a pas de colonne texte

# --------------------------------------------------------------------------- #
# Inférence / GPU
# --------------------------------------------------------------------------- #
USE_FP16: bool = True
ZSC_BATCH_SIZE: int = 32
NLLB_BATCH_SIZE: int = 16
NLLB_MAX_LENGTH: int = 128   # longueur max de génération (questions courtes)
