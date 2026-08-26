"""Bonus — FAQ validée (30-40 réponses) avec mention « à valider par un médecin ».

5 formulations de question par intention (soit 8 x 5 = 40 entrées), chacune
associée à une réponse de référence alignée sur le PEV (Programme Élargi de
Vaccination) du Congo et les recommandations OMS. Les réponses lingala/kituba
sont générées par NLLB si un traducteur est fourni, sinon laissées vides.

IMPORTANT (éthique) : ces réponses sont des aides à la rédaction destinées à
être relues et validées par un professionnel de santé avant tout usage réel.
"""

from __future__ import annotations

from typing import Optional

from .config import INTENT_TO_FAQ

# --------------------------------------------------------------------------- #
# FAQ rédigée (5 questions par intention + 1 réponse de référence)
# --------------------------------------------------------------------------- #
FAQ_INTENTS: list[dict] = [
    {
        "intention": "UTILITE_VACCIN",
        "answer_fr": (
            "Les vaccins apprennent au corps de votre enfant à se défendre contre des "
            "maladies graves : rougeole, poliomyélite, tuberculose, coqueluche, tétanos, "
            "diphtérie, hépatite B, pneumonie et fièvre jaune. Vacciner votre enfant le "
            "protège lui-même et protège aussi les autres enfants de la communauté."
        ),
        "questions_fr": [
            "À quoi servent les vaccins pour les bébés ?",
            "Contre quelles maladies le vaccin protège-t-il mon enfant ?",
            "Pourquoi faut-il vacciner mon enfant dès la naissance ?",
            "Le vaccin Penta protège contre quoi exactement ?",
            "Pourquoi vacciner contre la rougeole ?",
        ],
    },
    {
        "intention": "SECURITE_VACCIN",
        "answer_fr": (
            "Les vaccins utilisés au Congo respectent les normes de l'OMS et sont "
            "surveillés. Les réactions graves sont extrêmement rares. Les adjuvants "
            "comme l'aluminium sont présents en quantités infimes, sans danger aux doses "
            "utilisées. Un enfant enrhumé ou légèrement malade peut généralement être "
            "vacciné ; parlez-en à l'agent de santé."
        ),
        "questions_fr": [
            "Est-ce que le vaccin est dangereux pour mon bébé ?",
            "Les vaccins contiennent-ils des produits toxiques ?",
            "Est-ce que le vaccin contient du mercure ?",
            "Y a-t-il de l'aluminium dans les vaccins ?",
            "Est-ce que le vaccin est sûr pour les nourrissons ?",
        ],
    },
    {
        "intention": "CALENDRIER_RDV",
        "answer_fr": (
            "Au Congo-Brazzaville, le calendrier prévoit : à la naissance le BCG et la "
            "polio ; à 6, 10 et 14 semaines les vaccins Penta, pneumocoque et polio ; à "
            "9 mois la rougeole et la fièvre jaune. Le carnet de vaccination indique la "
            "date du prochain rendez-vous ; conservez-le précieusement."
        ),
        "questions_fr": [
            "À quel âge mon bébé doit-il recevoir le premier vaccin ?",
            "Quel est le calendrier de vaccination des enfants au Congo ?",
            "Quand mon enfant doit-il recevoir le vaccin contre la rougeole ?",
            "À quel âge donne-t-on le vaccin Penta ?",
            "Quand a lieu le prochain rendez-vous de vaccination ?",
        ],
    },
    {
        "intention": "RETARD_RATTRAPAGE",
        "answer_fr": (
            "Un vaccin manqué peut toujours être rattrapé : il ne faut jamais recommencer "
            "tout le calendrier depuis le début. Apportez le carnet de vaccination au "
            "centre de santé, l'agent établira un plan de rattrapage adapté à l'âge de "
            "votre enfant."
        ),
        "questions_fr": [
            "Mon enfant a raté un rendez-vous de vaccination, que faire ?",
            "J'ai oublié le vaccin de neuf mois, est-ce trop tard ?",
            "Comment rattraper les vaccins en retard ?",
            "Mon enfant n'a pas été vacciné à la naissance, que faire ?",
            "Faut-il recommencer la vaccination depuis le début ?",
        ],
    },
    {
        "intention": "EFFET_SECONDAIRE",
        "answer_fr": (
            "Une fièvre légère, une douleur ou un gonflement au point d'injection, des "
            "pleurs et de la fatigue sont fréquents et disparaissent en 1 à 2 jours. "
            "Faites boire l'enfant et consultez l'agent de santé pour la fièvre. Si la "
            "fièvre est très élevée, si l'enfant convulse ou devient très faible, allez "
            "immédiatement au centre de santé."
        ),
        "questions_fr": [
            "Mon bébé a de la fièvre après le vaccin, est-ce normal ?",
            "Le bras de mon enfant est gonflé après le vaccin",
            "Que faire si mon bébé a de la fièvre après le vaccin ?",
            "Combien de temps dure la fièvre après le vaccin ?",
            "Mon enfant a une boule au point d'injection, est-ce normal ?",
        ],
    },
    {
        "intention": "RUMEUR_CROYANCE",
        "answer_fr": (
            "Non, les vaccins ne rendent pas stérile, ne contiennent ni poison ni puce, "
            "et ne sont pas un complot. Ces rumeurs circulent sans preuve scientifique. "
            "Les vaccins sauvent des vies. Discutez de vos craintes avec un agent de "
            "santé, il répondra à vos questions."
        ),
        "questions_fr": [
            "On dit que le vaccin rend les enfants stériles, est-ce vrai ?",
            "J'ai entendu que le vaccin empoisonne les enfants",
            "On dit que les vaccins sont faits pour réduire la population",
            "Est-ce vrai que le vaccin contient des puces pour nous contrôler ?",
            "On raconte que le vaccin donne le cancer, est-ce vrai ?",
        ],
    },
    {
        "intention": "LOCALISATION_ACCES",
        "answer_fr": (
            "La vaccination des enfants est gratuite dans les centres de santé intégrés "
            "et les hôpitaux publics du Congo. Munissez-vous du carnet de vaccination. "
            "Les séances ont souvent lieu en matinée ; renseignez-vous auprès du centre "
            "de votre quartier ou de la PMI la plus proche."
        ),
        "questions_fr": [
            "Où puis-je faire vacciner mon enfant ?",
            "Est-ce que la vaccination est gratuite ?",
            "Quels sont les horaires de vaccination au centre de santé ?",
            "Où se trouve le centre de vaccination le plus proche ?",
            "Quels papiers faut-il pour vacciner mon enfant ?",
        ],
    },
    {
        "intention": "HORS_DOMAINE_CLINIQUE",
        "answer_fr": (
            "Votre question concerne un problème médical urgent. Rendez-vous "
            "immédiatement au centre de santé le plus proche ou contactez un agent de "
            "santé. Ce service ne remplace pas une consultation médicale."
        ),
        "questions_fr": [
            "Mon bébé convulse, que faire ?",
            "Mon enfant a une forte fièvre et respire mal",
            "Mon bébé a la diarrhée depuis trois jours, que faire ?",
            "Mon enfant ne respire plus, que faire ?",
            "Mon enfant est inconscient, que faire ?",
        ],
    },
]

VALIDATION_STATUS = "à valider par un médecin"
FAQ_SOURCE = "redaction_PEV_OMS_a_valider"


def build_faq(translator=None) -> list[dict]:
    """Construit les 40 entrées FAQ (avec traduction lin/kt si traducteur fourni)."""
    entries: list[dict] = []
    for block in FAQ_INTENTS:
        intent = block["intention"]
        faq_id = INTENT_TO_FAQ[intent]
        answer_fr = block["answer_fr"]

        # Traductions éventuelles (une seule traduction par intention, réutilisée)
        answer_lin, answer_kt = "", ""
        if translator is not None:
            try:
                answer_lin = translator.translate([answer_fr], "fra_Latn", "lin_Latn")[0]
                answer_kt = translator.translate([answer_fr], "fra_Latn", "kon_Latn")[0]
            except Exception as exc:  # noqa: BLE001 — dégradation propre
                answer_lin, answer_kt = "", ""
                print(f"[FAQ] Traduction impossible ({exc}) : réponses lin/kt laissées vides.")

        for q in block["questions_fr"]:
            entries.append(
                {
                    "faq_id": faq_id,
                    "intention": intent,
                    "question_fr": q,
                    "reponse_fr": answer_fr,
                    "reponse_lin": answer_lin,
                    "reponse_kt": answer_kt,
                    "statut_validation": VALIDATION_STATUS,
                    "source": FAQ_SOURCE,
                }
            )
    return entries
