"""Banque de questions seed en français (source : `seed_curated`).

Rôle : garantir la couverture des 8 intentions (notamment RUMEUR_CROYANCE et
HORS_DOMAINE_CLINIQUE, quasi absentes des datasets HF publics) et fournir la
voix réaliste de parents congolais (Brazzaville, Pointe-Noire…).

Les questions sont des tuples `(texte, intention)`. Elles sont rédigées à la
main, sans données personnelles, et passent ensuite par le même nettoyage que
les questions extraites de Hugging Face. `build_seed_df()` les convertit en
DataFrame (import pandas fait à la demande pour garder ce module importable
sans dépendances).
"""

from __future__ import annotations

from .config import INTENTS, RANDOM_SEED

# --------------------------------------------------------------------------- #
# ~34 questions par intention, soit ~272 questions.
# --------------------------------------------------------------------------- #
SEED_QUESTIONS: list[tuple[str, str]] = [
    # ------------------------------------------------------------------ #
    # UTILITE_VACCIN — pourquoi vacciner, contre quoi
    # ------------------------------------------------------------------ #
    ("À quoi servent les vaccins pour les bébés ?", "UTILITE_VACCIN"),
    ("Pourquoi faut-il vacciner mon enfant dès la naissance ?", "UTILITE_VACCIN"),
    ("Contre quelles maladies le vaccin protège-t-il mon bébé ?", "UTILITE_VACCIN"),
    ("Le BCG protège contre quelle maladie ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner contre la rougeole ?", "UTILITE_VACCIN"),
    ("Le vaccin Penta protège contre quoi exactement ?", "UTILITE_VACCIN"),
    ("Quelles maladies évite le vaccin contre la poliomyélite ?", "UTILITE_VACCIN"),
    ("Est-ce que le vaccin protège toute la vie ?", "UTILITE_VACCIN"),
    ("Pourquoi les vaccins sont-ils importants pour la santé de mon enfant ?", "UTILITE_VACCIN"),
    ("Mon bébé peut-il être malade sans vaccin ?", "UTILITE_VACCIN"),
    ("Le vaccin contre la fièvre jaune est-il obligatoire ?", "UTILITE_VACCIN"),
    ("Quel vaccin protège contre le tétanos ?", "UTILITE_VACCIN"),
    ("Le vaccin contre la rougeole évite quelles complications ?", "UTILITE_VACCIN"),
    ("Pourquoi donne-t-on le vaccin contre la polio en gouttes dans la bouche ?", "UTILITE_VACCIN"),
    ("Le vaccin pneumocoque protège contre quoi ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner mon enfant même s'il semble en bonne santé ?", "UTILITE_VACCIN"),
    ("Les vaccins aident-ils à protéger aussi les autres enfants ?", "UTILITE_VACCIN"),
    ("C'est quoi l'immunité que donne le vaccin ?", "UTILITE_VACCIN"),
    ("Le vaccin contre l'hépatite B est-il donné aux enfants ?", "UTILITE_VACCIN"),
    ("Pourquoi le vaccin contre la coqueluche est-il important pour les bébés ?", "UTILITE_VACCIN"),
    ("Est-ce que la vaccination peut faire disparaître une maladie ?", "UTILITE_VACCIN"),
    ("Contre quoi protège le vaccin Rota ?", "UTILITE_VACCIN"),
    ("Le vaccin contre la diphtérie protège de quoi ?", "UTILITE_VACCIN"),
    ("Pourquoi vacciner contre Haemophilus influenzae type b ?", "UTILITE_VACCIN"),
    ("Est-ce que mon enfant doit recevoir tous les vaccins du carnet ?", "UTILITE_VACCIN"),
    ("Quelle est l'importance du vaccin dans la prévention des maladies ?", "UTILITE_VACCIN"),
    ("Le vaccin protège-t-il contre la paralysie ?", "UTILITE_VACCIN"),
    ("Pourquoi le vaccin de la rougeole se donne à neuf mois ?", "UTILITE_VACCIN"),
    ("Quels vaccins protègent contre les maladies respiratoires ?", "UTILITE_VACCIN"),
    ("Le BCG évite quelle forme de tuberculose ?", "UTILITE_VACCIN"),
    ("Pourquoi les nouveau-nés reçoivent-ils un vaccin dès la naissance ?", "UTILITE_VACCIN"),
    ("Quel est le rôle des vaccins dans la lutte contre les épidémies ?", "UTILITE_VACCIN"),
    ("Le vaccin peut-il empêcher les épidémies dans mon quartier ?", "UTILITE_VACCIN"),
    ("Pourquoi le carnet recommande-t-il autant de vaccins ?", "UTILITE_VACCIN"),

    # ------------------------------------------------------------------ #
    # SECURITE_VACCIN — peurs, danger, composition
    # ------------------------------------------------------------------ #
    ("Est-ce que le vaccin est dangereux pour mon bébé ?", "SECURITE_VACCIN"),
    ("Les vaccins contiennent-ils des produits toxiques ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin contient du mercure ?", "SECURITE_VACCIN"),
    ("J'ai peur de faire vacciner mon enfant, est-ce que c'est sûr ?", "SECURITE_VACCIN"),
    ("Les vaccins peuvent-ils rendre mon enfant malade ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin est sans danger pour les nourrissons ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il causer la mort ?", "SECURITE_VACCIN"),
    ("Quels sont les composants du vaccin ?", "SECURITE_VACCIN"),
    ("Y a-t-il de l'aluminium dans les vaccins ?", "SECURITE_VACCIN"),
    ("Les vaccins sont-ils testés avant d'être donnés aux enfants ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin peut affaiblir mon bébé ?", "SECURITE_VACCIN"),
    ("Le vaccin Penta est-il sûr ?", "SECURITE_VACCIN"),
    ("Les vaccins donnés au Congo sont-ils de bonne qualité ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin fait grossir ou maigrir ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il donner des maladies à mon enfant ?", "SECURITE_VACCIN"),
    ("Les vaccins sont-ils sûrs pour les bébés prématurés ?", "SECURITE_VACCIN"),
    ("Est-ce que je peux refuser un vaccin si j'ai peur ?", "SECURITE_VACCIN"),
    ("Le vaccin contre la rougeole est-il dangereux ?", "SECURITE_VACCIN"),
    ("Est-ce que les vaccins contiennent des microbes vivants ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il brûler la peau de mon enfant ?", "SECURITE_VACCIN"),
    ("Pourquoi certains enfants tombent-ils malades après le vaccin ?", "SECURITE_VACCIN"),
    ("Les vaccins sont-ils fabriqués avec du sang ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin peut donner le sida ?", "SECURITE_VACCIN"),
    ("Le vaccin peut-il causer des malformations chez l'enfant ?", "SECURITE_VACCIN"),
    ("Y a-t-il des risques à vacciner un enfant enrhumé ?", "SECURITE_VACCIN"),
    ("Le vaccin est-il sûr si mon bébé a la diarrhée ?", "SECURITE_VACCIN"),
    ("Est-ce que recevoir trop de vaccins à la fois c'est dangereux ?", "SECURITE_VACCIN"),
    ("Quels sont les effets à long terme des vaccins ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin affaiblit les défenses de l'enfant ?", "SECURITE_VACCIN"),
    ("Les aiguilles utilisées pour vacciner sont-elles propres ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin peut donner une forte fièvre ?", "SECURITE_VACCIN"),
    ("Est-ce que le vaccin est sûr pour un enfant malnutri ?", "SECURITE_VACCIN"),
    ("Les vaccins peuvent-ils être périmés au centre de santé ?", "SECURITE_VACCIN"),
    ("Est-ce que les vaccins sont bien conservés au froid ?", "SECURITE_VACCIN"),

    # ------------------------------------------------------------------ #
    # CALENDRIER_RDV — dates, âges, prochain rendez-vous
    # ------------------------------------------------------------------ #
    ("À quel âge mon bébé doit-il recevoir le premier vaccin ?", "CALENDRIER_RDV"),
    ("Quel est le calendrier de vaccination des enfants au Congo ?", "CALENDRIER_RDV"),
    ("Quand faut-il donner le BCG ?", "CALENDRIER_RDV"),
    ("À quel âge donne-t-on le vaccin Penta ?", "CALENDRIER_RDV"),
    ("Quand mon enfant doit-il recevoir le vaccin contre la rougeole ?", "CALENDRIER_RDV"),
    ("À quel âge se fait le vaccin contre la fièvre jaune ?", "CALENDRIER_RDV"),
    ("Quand a lieu le prochain rendez-vous de vaccination ?", "CALENDRIER_RDV"),
    ("Mon bébé a six semaines, quels vaccins doit-il recevoir ?", "CALENDRIER_RDV"),
    ("À dix semaines, quels vaccins faut-il faire ?", "CALENDRIER_RDV"),
    ("Quel vaccin doit-on faire à quatorze semaines ?", "CALENDRIER_RDV"),
    ("Quand faire le rappel du vaccin ?", "CALENDRIER_RDV"),
    ("À neuf mois, quel vaccin faut-il donner ?", "CALENDRIER_RDV"),
    ("Combien de fois mon enfant doit-il être vacciné ?", "CALENDRIER_RDV"),
    ("Quelles sont les dates des séances de vaccination ?", "CALENDRIER_RDV"),
    ("Mon enfant de trois mois, quels vaccins lui manquent ?", "CALENDRIER_RDV"),
    ("Quand commence le calendrier vaccinal de l'enfant ?", "CALENDRIER_RDV"),
    ("À quel âge le vaccin de la polio est-il donné ?", "CALENDRIER_RDV"),
    ("Combien de doses de Penta faut-il ?", "CALENDRIER_RDV"),
    ("Quand donner la deuxième dose de rougeole ?", "CALENDRIER_RDV"),
    ("Mon bébé est né hier, quand faut-il le vacciner ?", "CALENDRIER_RDV"),
    ("Quels vaccins à deux mois ?", "CALENDRIER_RDV"),
    ("Le carnet de vaccination indique quelle prochaine date ?", "CALENDRIER_RDV"),
    ("Quand faire le vaccin contre le pneumocoque ?", "CALENDRIER_RDV"),
    ("À quel âge le vaccin Rota est-il donné ?", "CALENDRIER_RDV"),
    ("Quand revient-on après le BCG ?", "CALENDRIER_RDV"),
    ("Quel est le programme élargi de vaccination des enfants ?", "CALENDRIER_RDV"),
    ("À partir de quel âge un enfant est-il complètement vacciné ?", "CALENDRIER_RDV"),
    ("Quels vaccins faut-il faire avant l'âge d'un an ?", "CALENDRIER_RDV"),
    ("Quand donne-t-on la troisième dose de Penta ?", "CALENDRIER_RDV"),
    ("Mon enfant de cinq mois, quels vaccins lui donner ?", "CALENDRIER_RDV"),
    ("Quand se termine le calendrier vaccinal de l'enfant ?", "CALENDRIER_RDV"),
    ("Quel âge pour le vaccin antitétanique de l'enfant ?", "CALENDRIER_RDV"),
    ("Quels vaccins mon enfant doit-il recevoir avant son premier anniversaire ?", "CALENDRIER_RDV"),
    ("À quel âge donne-t-on le vaccin de rappel contre le tétanos ?", "CALENDRIER_RDV"),

    # ------------------------------------------------------------------ #
    # RETARD_RATTRAPAGE — que faire en cas de retard
    # ------------------------------------------------------------------ #
    ("Mon enfant a raté un rendez-vous de vaccination, que faire ?", "RETARD_RATTRAPAGE"),
    ("J'ai oublié le vaccin de neuf mois, est-ce trop tard ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant n'a pas été vacciné à la naissance, que faire ?", "RETARD_RATTRAPAGE"),
    ("Comment rattraper les vaccins en retard ?", "RETARD_RATTRAPAGE"),
    ("Est-ce qu'on peut encore vacciner un enfant en retard ?", "RETARD_RATTRAPAGE"),
    ("Mon bébé a raté le Penta 2, que faire ?", "RETARD_RATTRAPAGE"),
    ("Nous étions en voyage, l'enfant a raté ses vaccins", "RETARD_RATTRAPAGE"),
    ("Comment rattraper le vaccin contre la rougeole ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant de deux ans n'a jamais été vacciné, peut-on commencer ?", "RETARD_RATTRAPAGE"),
    ("Quel est le délai pour rattraper un vaccin manqué ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté le BCG à la naissance, peut-on le faire plus tard ?", "RETARD_RATTRAPAGE"),
    ("Que faire si j'ai perdu le carnet de vaccination ?", "RETARD_RATTRAPAGE"),
    ("Le centre était fermé, mon enfant a raté sa séance, que faire ?", "RETARD_RATTRAPAGE"),
    ("Faut-il recommencer la vaccination depuis le début ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté la deuxième dose de Penta, comment faire ?", "RETARD_RATTRAPAGE"),
    ("Peut-on rattraper deux vaccins en même temps ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a manqué la polio, peut-on la rattraper ?", "RETARD_RATTRAPAGE"),
    ("Quel est l'âge limite pour rattraper le vaccin contre la rougeole ?", "RETARD_RATTRAPAGE"),
    ("J'ai déménagé, comment continuer la vaccination de mon enfant ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté le vaccin de la fièvre jaune, que faire ?", "RETARD_RATTRAPAGE"),
    ("Peut-on rattraper les vaccins à l'école ?", "RETARD_RATTRAPAGE"),
    ("Est-ce grave si l'enfant a deux semaines de retard ?", "RETARD_RATTRAPAGE"),
    ("Comment rattraper le vaccin Penta 3 ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté le Rota, peut-on le donner plus tard ?", "RETARD_RATTRAPAGE"),
    ("Que faire si l'enfant a dépassé l'âge du vaccin BCG ?", "RETARD_RATTRAPAGE"),
    ("Est-ce que le retard annule les vaccins déjà reçus ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté sa vaccination à cause de la maladie, que faire ?", "RETARD_RATTRAPAGE"),
    ("Comment reprendre le calendrier après une interruption ?", "RETARD_RATTRAPAGE"),
    ("Peut-on rattraper les vaccins dans un autre centre ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant de dix-huit mois n'a reçu qu'un seul vaccin, que faire ?", "RETARD_RATTRAPAGE"),
    ("Quel est le programme de rattrapage recommandé ?", "RETARD_RATTRAPAGE"),
    ("Est-ce que je peux rattraper les vaccins le même mois ?", "RETARD_RATTRAPAGE"),
    ("Mon enfant a raté le BCG, jusqu'à quel âge peut-on le faire ?", "RETARD_RATTRAPAGE"),
    ("Comment rattraper les vaccins si le carnet est incomplet ?", "RETARD_RATTRAPAGE"),

    # ------------------------------------------------------------------ #
    # EFFET_SECONDAIRE — fièvre, gonflement, pleurs après vaccin
    # ------------------------------------------------------------------ #
    ("Mon bébé a de la fièvre après le vaccin, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Le bras de mon enfant est gonflé après le vaccin", "EFFET_SECONDAIRE"),
    ("Mon enfant pleure beaucoup après le vaccin, que faire ?", "EFFET_SECONDAIRE"),
    ("Que faire si mon bébé a de la fièvre après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a une boule au point d'injection, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Combien de temps dure la fièvre après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon bébé a la diarrhée après le vaccin Rota, que faire ?", "EFFET_SECONDAIRE"),
    ("Mon enfant dort beaucoup après le vaccin, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Est-ce normal que mon bébé refuse de manger après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon enfant vomit après le vaccin, que faire ?", "EFFET_SECONDAIRE"),
    ("La rougeur au point d'injection est-elle grave ?", "EFFET_SECONDAIRE"),
    ("Quel médicament donner contre la fièvre après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a une éruption cutanée après le vaccin contre la rougeole", "EFFET_SECONDAIRE"),
    ("Après le BCG, une plaie apparaît, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Mon bébé a mal à la jambe après le vaccin", "EFFET_SECONDAIRE"),
    ("Que faire contre la douleur au point d'injection ?", "EFFET_SECONDAIRE"),
    ("Mon enfant est irritable après le vaccin, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Après la vaccination, mon enfant a de la fièvre à 38, que faire ?", "EFFET_SECONDAIRE"),
    ("Le vaccin Penta donne-t-il de la fièvre ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a une petite boule sous la peau après le vaccin", "EFFET_SECONDAIRE"),
    ("Est-ce que la fièvre après le vaccin est dangereuse ?", "EFFET_SECONDAIRE"),
    ("Comment calmer les pleurs de mon bébé après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a la tête chaude après le vaccin, que faire ?", "EFFET_SECONDAIRE"),
    ("Combien de jours durent les effets du vaccin ?", "EFFET_SECONDAIRE"),
    ("Est-ce normal que l'enfant ait moins d'appétit après le vaccin ?", "EFFET_SECONDAIRE"),
    ("Après le vaccin rougeole, la fièvre arrive après combien de jours ?", "EFFET_SECONDAIRE"),
    ("Mon bébé a des convulsions après le vaccin, que faire ?", "EFFET_SECONDAIRE"),
    ("Peut-on mettre une compresse sur le point d'injection ?", "EFFET_SECONDAIRE"),
    ("Mon enfant tousse après le vaccin, est-ce lié ?", "EFFET_SECONDAIRE"),
    ("Le vaccin fait-il mal au bébé ?", "EFFET_SECONDAIRE"),
    ("Mon enfant a une forte fièvre le soir du vaccin, que faire ?", "EFFET_SECONDAIRE"),
    ("Quand faut-il s'inquiéter des effets du vaccin ?", "EFFET_SECONDAIRE"),
    ("Mon bébé a de la fièvre deux jours après le vaccin, est-ce normal ?", "EFFET_SECONDAIRE"),
    ("Peut-on donner du paracétamol avant le vaccin pour éviter la fièvre ?", "EFFET_SECONDAIRE"),

    # ------------------------------------------------------------------ #
    # RUMEUR_CROYANCE — rumeurs locales (stérilité, empoisonnement…)
    # ------------------------------------------------------------------ #
    ("On dit que le vaccin rend les enfants stériles, est-ce vrai ?", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin empoisonne les enfants", "RUMEUR_CROYANCE"),
    ("On dit que les vaccins sont faits pour réduire la population", "RUMEUR_CROYANCE"),
    ("Ma belle-mère dit que le vaccin attire les mauvais esprits", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin donne le cancer", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin est un complot des Blancs ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin rend la femme stérile, est-ce vrai ?", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin contient des puces pour nous contrôler", "RUMEUR_CROYANCE"),
    ("On dit que les enfants vaccinés meurent plus tard", "RUMEUR_CROYANCE"),
    ("Le vaccin est-il lié à la sorcellerie ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin rend les garçons impuissants", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin fait pousser des cornes aux enfants ?", "RUMEUR_CROYANCE"),
    ("Ma voisine dit qu'il ne faut pas vacciner les filles", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin stérilise les enfants dans le ventre", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin rend sourd ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin est fait avec du sang de serpent", "RUMEUR_CROYANCE"),
    ("Le pasteur a dit que le vaccin est la marque du diable", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin contient la maladie elle-même pour tuer", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que les infirmières donnent du poison dans le vaccin ?", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin fait grossir le ventre", "RUMEUR_CROYANCE"),
    ("Est-ce que le vaccin rend les enfants stupides ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin provoque des avortements", "RUMEUR_CROYANCE"),
    ("J'ai entendu que le vaccin donne le paludisme", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin est inutile, que c'est du business", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin contient des organes d'animaux ?", "RUMEUR_CROYANCE"),
    ("On raconte que le vaccin empêche les femmes d'accoucher", "RUMEUR_CROYANCE"),
    ("Ma famille dit que les vaccins tuent plus qu'ils ne protègent", "RUMEUR_CROYANCE"),
    ("Est-ce que le vaccin fait naître des enfants malformés ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin rend les enfants paresseux", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin donne la tuberculose ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin est réservé aux enfants pauvres", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que le vaccin rend les gens malades pour les tuer ?", "RUMEUR_CROYANCE"),
    ("On dit que le vaccin fait mourir les bébés dans leur sommeil, est-ce vrai ?", "RUMEUR_CROYANCE"),
    ("Est-ce vrai que les vaccins rendent les enfants obéissants comme des robots ?", "RUMEUR_CROYANCE"),

    # ------------------------------------------------------------------ #
    # LOCALISATION_ACCES — où aller, horaires, gratuité
    # ------------------------------------------------------------------ #
    ("Où puis-je faire vacciner mon enfant ?", "LOCALISATION_ACCES"),
    ("Où se trouve le centre de vaccination le plus proche ?", "LOCALISATION_ACCES"),
    ("Est-ce que la vaccination est gratuite ?", "LOCALISATION_ACCES"),
    ("Quels sont les horaires de vaccination au centre de santé ?", "LOCALISATION_ACCES"),
    ("Où faire le vaccin à Brazzaville ?", "LOCALISATION_ACCES"),
    ("Est-ce que je peux vacciner mon enfant à Pointe-Noire ?", "LOCALISATION_ACCES"),
    ("Quels papiers faut-il pour vacciner mon enfant ?", "LOCALISATION_ACCES"),
    ("La vaccination se fait-elle tous les jours ?", "LOCALISATION_ACCES"),
    ("Où acheter le carnet de vaccination ?", "LOCALISATION_ACCES"),
    ("Est-ce que le vaccin est disponible au dispensaire ?", "LOCALISATION_ACCES"),
    ("Comment trouver un centre de vaccination près de chez moi ?", "LOCALISATION_ACCES"),
    ("Y a-t-il des séances de vaccination le samedi ?", "LOCALISATION_ACCES"),
    ("Est-ce que la vaccination est payante ?", "LOCALISATION_ACCES"),
    ("Où faire le BCG à la naissance ?", "LOCALISATION_ACCES"),
    ("Est-ce que les vaccins sont disponibles à l'hôpital ?", "LOCALISATION_ACCES"),
    ("Quel centre de santé vaccine les enfants dans mon quartier ?", "LOCALISATION_ACCES"),
    ("Faut-il prendre rendez-vous pour la vaccination ?", "LOCALISATION_ACCES"),
    ("Où se trouve le centre de santé intégré le plus proche ?", "LOCALISATION_ACCES"),
    ("La vaccination est-elle gratuite au Congo ?", "LOCALISATION_ACCES"),
    ("Quels sont les jours de vaccination de masse ?", "LOCALISATION_ACCES"),
    ("Où faire le vaccin contre la rougeole ?", "LOCALISATION_ACCES"),
    ("Est-ce que je peux vacciner mon enfant à la PMI ?", "LOCALISATION_ACCES"),
    ("Le vaccin est-il disponible dans les villages ?", "LOCALISATION_ACCES"),
    ("Combien coûte le carnet de vaccination ?", "LOCALISATION_ACCES"),
    ("Où se procurer les vaccins du programme élargi ?", "LOCALISATION_ACCES"),
    ("Est-ce que les équipes passent dans les quartiers pour vacciner ?", "LOCALISATION_ACCES"),
    ("Quelle est l'adresse du centre de vaccination de Ouenzé ?", "LOCALISATION_ACCES"),
    ("Où faire vacciner mon enfant à Talangaï ?", "LOCALISATION_ACCES"),
    ("Faut-il une ordonnance pour vacciner mon enfant ?", "LOCALISATION_ACCES"),
    ("Y a-t-il une campagne de vaccination en ce moment ?", "LOCALISATION_ACCES"),
    ("Où faire le rappel de vaccination de mon enfant ?", "LOCALISATION_ACCES"),
    ("Est-ce que la vaccination est accessible aux personnes déplacées ?", "LOCALISATION_ACCES"),
    ("Est-ce que les centres de santé ouvrent le dimanche pour la vaccination ?", "LOCALISATION_ACCES"),
    ("Où se procurer un nouveau carnet de vaccination si on l'a perdu ?", "LOCALISATION_ACCES"),

    # ------------------------------------------------------------------ #
    # HORS_DOMAINE_CLINIQUE — urgences médicales à transférer à un agent
    # ------------------------------------------------------------------ #
    ("Mon bébé convulse, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une forte fièvre et respire mal", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la diarrhée depuis trois jours, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant saigne beaucoup, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé ne bouge plus, aidez-moi", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant est tombé et s'est cassé le bras", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a avalé un produit, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant ne respire plus, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a les yeux jaunes, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant vomit du sang", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a une forte fièvre qui ne descend pas", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a très mal au ventre", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la peau bleue, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant est inconscient, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé refuse de téter depuis ce matin", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une plaie qui s'infecte", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la malaria, quel traitement ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a bu de l'eau sale, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a des boutons partout sur le corps", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une toux qui ne s'arrête pas", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a de la fièvre et des convulsions", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a été mordu par un serpent", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la gorge enflée, il n'arrive pas à avaler", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a le palu grave, où l'emmener ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a perdu connaissance après une chute", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant urine du sang, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a très chaud et transpire beaucoup", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une crise d'asthme, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a une hernie qui grossit", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a la rougeole avec complications, que faire ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé ne réagit plus quand je l'appelle", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a une fracture, où aller en urgence ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon enfant a de la fièvre et des boutons, que faire en urgence ?", "HORS_DOMAINE_CLINIQUE"),
    ("Mon bébé a la fontanelle creuse, que faire ?", "HORS_DOMAINE_CLINIQUE"),
]


def validate_seed() -> None:
    """Vérifie la cohérence interne de la banque seed (utile en test)."""
    texts = [t for t, _ in SEED_QUESTIONS]
    intents = [i for _, i in SEED_QUESTIONS]
    assert len(texts) == len(set(texts)), "Doublons exacts dans la banque seed"
    for intent in INTENTS:
        n = intents.count(intent)
        assert n >= 30, f"Couverture insuffisante pour {intent} : {n}"
    # longueur minimale
    for t in texts:
        assert len(t.strip()) >= 10, f"Question trop courte : {t!r}"


def build_seed_df():
    """Construit le DataFrame pandas de la banque seed (import à la demande)."""
    import pandas as pd

    rows = [
        {
            "texte": texte,
            "intention": intention,
            "source": "seed_curated",
            "score": 1.0,  # étiquette humaine -> confiance maximale
            "is_seed": True,
        }
        for texte, intention in SEED_QUESTIONS
    ]
    return pd.DataFrame(rows)
