"""Banque seed v2 — questions supplémentaires ciblées (source : `seed_curated`).

Ces ~20 questions par intention S'AJOUTENT à la banque v1
(`seed_questions.SEED_QUESTIONS`) pour atteindre ~54 questions/intention, et
sont écrites selon les règles de `intent_rubric.md` : chaque question porte un
marqueur explicite de son intention (notamment pour départager
EFFET_SECONDAIRE / RUMEUR_CROYANCE / SECURITE_VACCIN, qui se confondaient en v1).

`SEED_QUESTIONS_V2` = v1 + ces ajouts. `build_seed_df()` renvoie le DataFrame
complet (labels humains, score=1.0).
"""

from __future__ import annotations

from .config import INTENTS
from .seed_questions import SEED_QUESTIONS

# --------------------------------------------------------------------------- #
# ~20 questions supplémentaires par intention (≈160 au total)
# --------------------------------------------------------------------------- #
SEED_QUESTIONS_V2_EXTRA: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- #
    # UTILITE_VACCIN — pourquoi / contre quoi (aucun marqueur de date)
    # ---------------------------------------------------------------- #
    ("À quoi sert le vaccin qu'on donne à la naissance ?", "UTILITE_VACCIN"),
    ("Quelles maladies mon enfant peut attraper s'il n'est pas vacciné ?", "UTILITE_VACCIN"),
    ("Le vaccin de la polio, il protège contre quoi au juste ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner les enfants contre des maladies qu'on ne voit plus ?", "UTILITE_VACCIN"),
    ("C'est quoi l'intérêt du BCG pour mon bébé ?", "UTILITE_VACCIN"),
    ("Le Penta sert à éviter quelles maladies ?", "UTILITE_VACCIN"),
    ("Est-ce que le vaccin évite la rougeole grave ?", "UTILITE_VACCIN"),
    ("Pourquoi donner un vaccin à un bébé qui vient de naître ?", "UTILITE_VACCIN"),
    ("Le vaccin contre la coqueluche protège mon enfant de quoi ?", "UTILITE_VACCIN"),
    ("Qu'est-ce que le vaccin apporte à la santé de mon enfant ?", "UTILITE_VACCIN"),
    ("Pourquoi faut-il plusieurs doses de vaccin ?", "UTILITE_VACCIN"),
    ("Le vaccin Rota sert à prévenir quoi ?", "UTILITE_VACCIN"),
    ("Quelle protection donne le vaccin contre le tétanos ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner les enfants contre la tuberculose ?", "UTILITE_VACCIN"),
    ("Le vaccin contre l'hépatite B protège de quoi ?", "UTILITE_VACCIN"),
    ("Est-ce que le vaccin peut empêcher la paralysie ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner mon enfant même s'il n'est jamais malade ?", "UTILITE_VACCIN"),
    ("Quel est le rôle du vaccin contre les épidémies de rougeole ?", "UTILITE_VACCIN"),
    ("Le vaccin pneumocoque évite quelle maladie ?", "UTILITE_VACCIN"),
    ("Pourquoi le vaccin est-il important dès le plus jeune âge ?", "UTILITE_VACCIN"),

    # ---------------------------------------------------------------- #
    # SECURITE_VACCIN — peur générique / danger / composition (1re pers.)
    # ---------------------------------------------------------------- #
    ("Est-ce que je peux faire confiance aux vaccins ?", "SECURITE_VACCIN"),
    ("Les vaccins sont-ils dangereux pour les petits ?", "SECURITE_VACCIN"),
    ("Que contient exactement le vaccin ?", "SECURITE_VACCIN"),
    ("Y a-t-il des produits dangereux dans le vaccin ?", "SECURITE_VACCIN"),
    ("J'ai peur des effets du vaccin sur mon bébé", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin est sans risque ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il blesser mon enfant ?", "SECURITE_VACCIN"),
    ("Est-ce que les adjuvants des vaccins sont dangereux ?", "SECURITE_VACCIN"),
    ("Les vaccins sont-ils contrôlés avant d'être utilisés ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin est fait avec des produits chimiques ?", "SECURITE_VACCIN"),
    ("J'hésite à faire vacciner mon enfant, est-ce vraiment sûr ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin peut rendre mon bébé faible ?", "SECURITE_VACCIN"),
    ("Les vaccins donnés ici sont-ils sûrs ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin contient des substances nocives ?", "SECURITE_VACCIN"),
    ("Puis-je vacciner mon enfant sans risque ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il avoir des effets graves ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin est dangereux pour un bébé de deux mois ?", "SECURITE_VACCIN"),
    ("Quels sont les risques si je fais vacciner mon enfant ?", "SECURITE_VACCIN"),
    ("Est-ce que les vaccins sont sûrs à cent pour cent ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il faire du mal à mon enfant ?", "SECURITE_VACCIN"),

    # ---------------------------------------------------------------- #
    # CALENDRIER_RDV — dates / âges / prochain RDV / doses (aucun retard)
    # ---------------------------------------------------------------- #
    ("Quel est l'âge exact pour le premier vaccin ?", "CALENDRIER_RDV"),
    ("Quand dois-je revenir pour la deuxième dose ?", "CALENDRIER_RDV"),
    ("À combien de mois se fait le vaccin de la rougeole ?", "CALENDRIER_RDV"),
    ("Quelle est la date du prochain vaccin de mon enfant ?", "CALENDRIER_RDV"),
    ("Mon enfant a trois mois, quels vaccins faut-il maintenant ?", "CALENDRIER_RDV"),
    ("Combien de visites de vaccination faut-il ?", "CALENDRIER_RDV"),
    ("À quel âge se fait le dernier vaccin du bébé ?", "CALENDRIER_RDV"),
    ("Quand faut-il faire le rappel ?", "CALENDRIER_RDV"),
    ("Quel vaccin à la naissance, à six semaines, à neuf mois ?", "CALENDRIER_RDV"),
    ("Quel vaccin doit recevoir mon enfant de sept mois ?", "CALENDRIER_RDV"),
    ("Quelle est la prochaine date de séance de vaccination ?", "CALENDRIER_RDV"),
    ("À quel âge donne-t-on le vaccin contre la fièvre jaune ?", "CALENDRIER_RDV"),
    ("Combien de doses de vaccin avant un an ?", "CALENDRIER_RDV"),
    ("Quand commence la vaccination des bébés ?", "CALENDRIER_RDV"),
    ("Mon bébé a six semaines, il faut quels vaccins ?", "CALENDRIER_RDV"),
    ("Quel âge pour le vaccin Penta ?", "CALENDRIER_RDV"),
    ("Quand faire le vaccin contre la polio ?", "CALENDRIER_RDV"),
    ("Le carnet indique de revenir à quelle date ?", "CALENDRIER_RDV"),
    ("Quels vaccins à dix semaines ?", "CALENDRIER_RDV"),
    ("À neuf mois, quel est le vaccin prévu ?", "CALENDRIER_RDV"),

    # ---------------------------------------------------------------- #
    # RETARD_RATTRAPAGE — manqué / oublié / perdu / interruption
    # ---------------------------------------------------------------- #
    ("Mon enfant a deux mois de retard sur ses vaccins, que faire ?", "RETARD_RATTRAPAGE"),
    ("J'ai oublié un vaccin, comment le rattraper ?", "RETARD_RATTRAPAGE"),
    ("Est-ce grave d'avoir raté le vaccin de six semaines ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant n'a reçu aucun vaccin, on peut commencer maintenant ?", "RETARD_RATTRAPAGE"),
    ("Comment faire si on a raté le rendez-vous de vaccination ?", "RETARD_RATTRAPAGE"),
    ("Le vaccin de naissance n'a pas été fait, c'est rattrapable ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté le Penta 1, comment rattraper ?", "RETARD_RATTRAPAGE"),
    ("Peut-on rattraper le vaccin de neuf mois à un an ?", "RETARD_RATTRAPAGE"),
    ("J'ai perdu le carnet, comment connaître les vaccins manqués ?", "RETARD_RATTRAPAGE"),
    ("Mon bébé était malade le jour du vaccin, que faire ?", "RETARD_RATTRAPAGE"),
    ("Comment reprendre la vaccination après une longue interruption ?", "RETARD_RATTRAPAGE"),
    ("Est-ce qu'on peut rattraper tous les vaccins d'un coup ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant de trois ans n'a jamais été vacciné, que faire ?", "RETARD_RATTRAPAGE"),
    ("Quel vaccin rattraper en premier ?", "RETARD_RATTRAPAGE"),
    ("Peut-on rattraper le BCG après la naissance ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté la troisième dose, faut-il tout recommencer ?", "RETARD_RATTRAPAGE"),
    ("Comment rattraper les vaccins si le centre était fermé ?", "RETARD_RATTRAPAGE"),
    ("Y a-t-il un programme spécial pour les enfants en retard ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a manqué la rougeole, jusqu'à quand rattraper ?", "RETARD_RATTRAPAGE"),
    ("Que faire si mon enfant a dépassé l'âge du vaccin ?", "RETARD_RATTRAPAGE"),

    # ---------------------------------------------------------------- #
    # EFFET_SECONDAIRE — bénin + lien vaccinal explicite
    # ---------------------------------------------------------------- #
    ("Mon bébé a 38 de fièvre après le vaccin, c'est normal ?", "EFFET_SECONDAIRE"),
    ("La jambe de mon enfant est rouge à l'endroit du vaccin", "EFFET_SECONDAIRE"),
    ("Mon enfant a mal là où on l'a piqué, que faire ?", "EFFET_SECONDAIRE"),
    ("Après le vaccin, mon bébé dort beaucoup", "EFFET_SECONDAIRE"),
    ("Mon enfant a une bosse au point d'injection", "EFFET_SECONDAIRE"),
    ("Le vaccin a donné un peu de fièvre à mon bébé, que faire ?", "EFFET_SECONDAIRE"),
    ("Mon enfant est grognon après la vaccination", "EFFET_SECONDAIRE"),
    ("Est-ce normal que mon bébé pleure après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Après le Penta, mon enfant a la diarrhée", "EFFET_SECONDAIRE"),
    ("Mon bébé mange moins après le vaccin, c'est grave ?", "EFFET_SECONDAIRE"),
    ("Le bras est gonflé deux jours après le vaccin", "EFFET_SECONDAIRE"),
    ("Mon enfant a vomi après le vaccin Rota", "EFFET_SECONDAIRE"),
    ("La peau est rouge autour de la piqûre", "EFFET_SECONDAIRE"),
    ("Mon enfant a une petite fièvre le soir du vaccin", "EFFET_SECONDAIRE"),
    ("Après le vaccin contre la rougeole, il a des boutons", "EFFET_SECONDAIRE"),
    ("Mon bébé a mal au bras et pleure, c'est à cause du vaccin ?", "EFFET_SECONDAIRE"),
    ("Combien de temps dure la douleur après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a de la fièvre et dort beaucoup depuis le vaccin", "EFFET_SECONDAIRE"),
    ("Est-ce que la boule sous la peau après le vaccin va disparaître ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a la tête chaude après la vaccination", "EFFET_SECONDAIRE"),

    # ---------------------------------------------------------------- #
    # RUMEUR_CROYANCE — source rapportée + contenu faux spécifique
    # ---------------------------------------------------------------- #
    ("On dit que le vaccin rend les filles stériles, c'est vrai ?", "RUMEUR_CROYANCE"),
    ("Ma voisine dit que le vaccin donne la malaria", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin contient du poison", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin est fait pour tuer les enfants", "RUMEUR_CROYANCE"),
    ("Le pasteur a dit que le vaccin est contre la religion", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin rend sourd", "RUMEUR_CROYANCE"),
    ("J'ai entendu dire que les vaccins rendent les femmes stériles", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin fait naître des enfants handicapés", "RUMEUR_CROYANCE"),
    ("Ma belle-mère dit que le vaccin attire les démons", "RUMEUR_CROYANCE"),
    ("On dit que les vaccins servent à contrôler les Africains", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin donne le cancer", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin empêche d'avoir des enfants", "RUMEUR_CROYANCE"),
    ("Des gens disent que le vaccin rend impuissant", "RUMEUR_CROYANCE"),
    ("On dit que les infirmières injectent du poison dans le vaccin", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin contient des puces électroniques", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin fait mourir les bébés", "RUMEUR_CROYANCE"),
    ("Ma famille croit que le vaccin apporte le mauvais sort", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin rend les enfants malades exprès", "RUMEUR_CROYANCE"),
    ("J'ai entendu que les vaccins sont inutiles et dangereux", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin rend les garçons stériles", "RUMEUR_CROYANCE"),

    # ---------------------------------------------------------------- #
    # LOCALISATION_ACCES — où / horaires / gratuité / documents
    # ---------------------------------------------------------------- #
    ("Où se fait la vaccination gratuite des enfants ?", "LOCALISATION_ACCES"),
    ("Le centre de santé vaccine quel jour ?", "LOCALISATION_ACCES"),
    ("Quels sont les horaires du centre de vaccination ?", "LOCALISATION_ACCES"),
    ("Est-ce que je dois payer pour vacciner mon enfant ?", "LOCALISATION_ACCES"),
    ("Où trouver un centre de vaccination ouvert le matin ?", "LOCALISATION_ACCES"),
    ("Comment aller au centre de santé pour le vaccin ?", "LOCALISATION_ACCES"),
    ("Est-ce que la PMI fait les vaccins gratuitement ?", "LOCALISATION_ACCES"),
    ("Y a-t-il une équipe de vaccination dans mon quartier ?", "LOCALISATION_ACCES"),
    ("Quel document apporter pour vacciner mon enfant ?", "LOCALISATION_ACCES"),
    ("Où faire le vaccin près de chez moi ?", "LOCALISATION_ACCES"),
    ("La vaccination est-elle gratuite à l'hôpital ?", "LOCALISATION_ACCES"),
    ("Quels jours de la semaine vaccine-t-on ?", "LOCALISATION_ACCES"),
    ("Est-ce que je peux venir vacciner mon enfant sans rendez-vous ?", "LOCALISATION_ACCES"),
    ("Où se trouve la maternité qui vaccine les nouveau-nés ?", "LOCALISATION_ACCES"),
    ("Le vaccin est-il disponible au dispensaire du quartier ?", "LOCALISATION_ACCES"),
    ("Combien coûte la vaccination d'un enfant ?", "LOCALISATION_ACCES"),
    ("Y a-t-il une campagne de vaccination gratuite bientôt ?", "LOCALISATION_ACCES"),
    ("Où récupérer le carnet de vaccination ?", "LOCALISATION_ACCES"),
    ("Est-ce que la vaccination est accessible le week-end ?", "LOCALISATION_ACCES"),
    ("Où emmener mon enfant pour son premier vaccin ?", "LOCALISATION_ACCES"),

    # ---------------------------------------------------------------- #
    # HORS_DOMAINE_CLINIQUE — urgence / gravité / sans lien vaccinal
    # ---------------------------------------------------------------- #
    ("Mon bébé a 40 de fièvre, où l'emmener en urgence ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a des convulsions, aidez-moi", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé ne tète plus depuis hier, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une plaie infectée à la jambe", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a le paludisme, quel traitement ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant est tombé et saigne de la tête", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé respire très vite, c'est grave ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a avalé un produit ménager", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la diarrhée avec du sang", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une fracture, où aller ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé est inconscient après une chute", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a très mal au ventre et vomit", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a les lèvres bleues", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a été mordu par un chien", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé ne réagit plus aux bruits", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une crise d'asthme grave", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la peau et les yeux jaunes", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant saigne du nez sans s'arrêter", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a une fièvre très élevée qui persiste malgré les médicaments", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a la rougeole et ne respire pas bien, que faire ?", "HORS_DOMAINE_CLINIQUE"),
]


def validate_seed_v2() -> None:
    """Vérifie la cohérence de la banque v2 (v1 + ajouts)."""
    texts = [t for t, _ in SEED_QUESTIONS_V2]
    intents = [i for _, i in SEED_QUESTIONS_V2]
    assert len(texts) == len(set(texts)), "Doublons exacts dans la banque seed v2"
    for intent in INTENTS:
        n = intents.count(intent)
        assert n >= 50, f"Couverture insuffisante pour {intent} : {n}"
    for t in texts:
        assert len(t.strip()) >= 10, f"Question trop courte : {t!r}"


def build_seed_df():
    """DataFrame pandas de la banque seed v2 (import pandas à la demande)."""
    import pandas as pd

    rows = [
        {
            "texte": texte,
            "intention": intention,
            "source": "seed_curated",
            "score": 1.0,  # étiquette humaine -> confiance maximale
            "is_seed": True,
        }
        for texte, intention in SEED_QUESTIONS_V2
    ]
    return pd.DataFrame(rows)


# Fusion v1 + v2 (une seule liste, ordonnée)
SEED_QUESTIONS_V2: list[tuple[str, str]] = SEED_QUESTIONS + SEED_QUESTIONS_V2_EXTRA
