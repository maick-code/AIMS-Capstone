"""Mots-clés de domaine et hypothèses zero-shot pour la classification.

Deux listes distinctes sont définies ici :

1. `DOMAIN_KEYWORDS` — motifs (regex) qui identifient le *domaine* de la
   vaccination pédiatrique. Utilisés à l'étape 2 (filtrage) pour ne garder que
   les questions pertinentes.
2. `INTENT_HYPOTHESES` — descriptions en français des 8 intentions, utilisées
   comme `candidate_labels` par le classifieur zero-shot à l'étape 3.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Mots-clés du DOMAINE (vaccination pédiatrique)
# --------------------------------------------------------------------------- #
# Chaque motif est une expression régulière, testée en mode insensible à la
# casse. La liste est volontairement large : rougeole, polio, BCG, Penta,
# carnet vaccinal, fièvre après vaccin, etc.
DOMAIN_KEYWORDS: list[str] = [
    r"vaccin",                       # vaccin, vaccination, vacciné, revaccination…
    r"immunis",                      # immunisation
    r"rougeole",
    r"polio",                        # polio, poliomyélite
    r"\bbcg\b",
    r"\bpenta\b",
    r"\bdtc\b",                      # DTC (diphtérie-tétanos-coqueluche)
    r"dipht[ée]rie",
    r"t[ée]tanos",
    r"coqueluche",
    r"fi[èe]vre\s+jaune",
    r"h[ée]patite",
    r"pneumocoque",
    r"\brota\b",                     # vaccin antirotavirus
    r"\bhib\b",                      # Haemophilus influenzae type b
    r"antig[èe]ne",
    r"carnet\s+(de\s+)?vaccin",
    r"calendrier\s+vaccin",
    r"\bpev\b",                      # Programme Élargi de Vaccination
    r"programme\s+[ée]largi",
    r"injection",
    r"piq[ûu]re",
    r"seringue",
    r"aiguille",
    r"fi[èe]vre\s+apr[èe]s\s+vaccin",
    r"gonfl[ée]ment",
    r"point\s+d['’]injection",
    r"dose\s+(de\s+)?vaccin",
    r"rappel\s+(de\s+)?vaccin",
    r"s[ée]ance\s+de\s+vaccin",
    r"st[ée]ril(e|it[ée])",          # rumeur de stérilité
    r"empoisonn",                    # rumeur d'empoisonnement
    r"complot",
]

# Mots-clés *pédiatriques* : présence optionnelle, utilisée pour caractériser
# (et éventuellement trier) les exemples, sans être obligatoire (une question
# « contre quoi protège le BCG ? » n'en contient pas).
PEDIATRIC_KEYWORDS: list[str] = [
    r"enfant", r"b[ée]b[ée]", r"nourrisson", r"nouveau-n[ée]", r"n[ée]onatal",
    r"petit", r"fils", r"fille", r"maman", r"papa", r"m[èe]re", r"parent",
]

# --------------------------------------------------------------------------- #
# Hypothèses zero-shot (libellés riches en français -> meilleur accord NLI)
# --------------------------------------------------------------------------- #
INTENT_HYPOTHESES: dict[str, str] = {
    "UTILITE_VACCIN": (
        "La personne demande à quoi servent les vaccins et contre quelles "
        "maladies ils protègent les enfants"
    ),
    "SECURITE_VACCIN": (
        "La personne a peur du vaccin et s'interroge sur sa dangerosité, sa "
        "composition ou sa sécurité pour son enfant"
    ),
    "CALENDRIER_RDV": (
        "La personne demande les dates, les âges et le prochain rendez-vous "
        "de vaccination de son enfant"
    ),
    "RETARD_RATTRAPAGE": (
        "La personne demande quoi faire car son enfant a raté ou retardé un "
        "vaccin et comment le rattraper"
    ),
    "EFFET_SECONDAIRE": (
        "La personne décrit une réaction après le vaccin comme la fièvre, un "
        "gonflement ou des pleurs de son enfant"
    ),
    "RUMEUR_CROYANCE": (
        "La personne rapporte une rumeur ou une croyance locale sur les vaccins "
        "comme la stérilité, l'empoisonnement ou un complot"
    ),
    "LOCALISATION_ACCES": (
        "La personne demande où aller pour vacciner son enfant, les horaires "
        "et si la vaccination est gratuite"
    ),
    "HORS_DOMAINE_CLINIQUE": (
        "La personne décrit un problème médical urgent qui nécessite d'être "
        "transféré à un agent de santé"
    ),
}

# Libellés dans l'ordre pour la pipeline zero-shot, et mapping inverse.
CANDIDATE_LABELS: list[str] = list(INTENT_HYPOTHESES.values())
LABEL_TO_INTENT: dict[str, str] = {v: k for k, v in INTENT_HYPOTHESES.items()}
