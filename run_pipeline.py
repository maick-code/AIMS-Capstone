#!/usr/bin/env python3
"""Point d'entrée unique du pipeline VaxiMère-QA-CG (compatible Colab).

Exemples :
    python run_pipeline.py --mode dryrun          # test rapide, sans modèle ni réseau
    python run_pipeline.py --mode full            # exécution complète (modèles HF)
    python run_pipeline.py --mode full --skip-translation
"""

from vaximere.pipeline import main

if __name__ == "__main__":
    main()
