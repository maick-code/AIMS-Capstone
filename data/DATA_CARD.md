# DATA_CARD — VaxiMère-QA-CG

Dataset d'intentions multilingue (français, lingala, kituba/munukutuba) sur la
vaccination pédiatrique au Congo-Brazzaville, destiné au fine-tuning LoRA d'un
classifieur d'intentions (Gemma 3).

> ⚠️ **Avertissement** : ce dataset est une **aide à la recherche et au
> prototypage**. Il ne remplace ni un avis médical ni un protocole sanitaire.
> Toute réponse de la FAQ doit être validée par un professionnel de santé avant
> usage réel.

---

## 1. Sources

| Source | Type | Langue | Licence | Usage |
|---|---|---|---|---|
| `qanastek/frenchmedmcqa` | QCM médicaux (champ `question`) | fra | Apache-2.0 | Questions réelles FR, filtrées par mots-clés |
| `ANR-MALADES/MediQAl` (config `oeq`) | Questions-réponses ouvertes | fra | CC-BY-4.0 | Questions réelles FR, filtrées par mots-clés |
| `seed_curated` (banque interne) | ~270 questions rédigées à la main | fra | (création interne) | Socle garanti couvrant les 8 intentions |
| `masakhane/masakhanews` (config `lin`) | Actualités, catégorie `health` | lin | AFL-3.0 | **Optionnel** — non classé (pas de couverture zero-shot lingala) |

**Remplacement de `blinoff/medical_qa_fr`** : ce dataset n'existe pas sur Hugging
Face (vérifié : 404). Il est remplacé par les deux sources françaises réelles
ci-dessus, complétées par la banque seed.

**Tweets vaccinaux français** : les datasets publics (ex.
`webimmunization/COVID-19-vaccine-attitude-tweets`) sont soit en anglais, soit
des identifiants Twitter **sans texte** (réhydratation API nécessaire). Ils sont
donc **non utilisés** ; l'intention `RUMEUR_CROYANCE` est couverte par la banque
seed.

## 2. Méthode

1. **Extraction** — chargement des datasets HF via `datasets`, normalisation vers
   le schéma `[texte, source, langue]`, cache CSV.
2. **Filtrage** — liste de mots-clés robuste (vaccin, rougeole, polio, BCG, Penta,
   carnet vaccinal, fièvre après vaccin…), nettoyage (longueur 15–400 caractères),
   dédoublonnage exact + quasi-doublons (Jaccard ≥ 0.90).
3. **Classification zero-shot** — `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`,
   8 hypothèses en français, seuil de confiance **≥ 0.70**.
4. **Traduction** — `facebook/nllb-200-distilled-600M` :
   - lingala → `lin_Latn`
   - kituba → `kon_Latn` (kikongo, utilisé pour **simuler** le kituba : NLLB-200
     n'a pas de code dédié au kituba/munukutuba)
5. **Équilibrage** — ~30 questions FR maîtresses/intention (≈ 1/3 seed + 2/3
   sources réelles), chacune déclinée en 3 langues → ~90 exemples/intention.
6. **Sortie** — `vaximere_qa_cg_train.jsonl` + versions par langue + FAQ.

**Schéma d'un exemple** :
```json
{
  "query_id": "Q_001_FR",
  "texte": "À quel âge mon bébé doit-il recevoir le premier vaccin ?",
  "langue": "fra",
  "intention": "CALENDRIER_RDV",
  "faq_target_id": "FAQ_003",
  "source": "seed_curated",
  "score": 1.0
}
```

## 3. Biais linguistiques

- **Déséquilibre de la voix** : les questions lingala/kituba sont des
  **traductions automatiques** de questions françaises, pas des énoncés produits
  nativement par des locuteurs. Les tournures idiomatiques locales sont sous-représentées.
- **Approximation kituba** : `kon_Latn` (kikongo) n'est pas le kituba/munukutuba
  parlé à Brazzaville/Pointe-Noire ; il peut en différer lexicalement.
- **Domination du français** : les sources HF réelles sont francophones ; le
  lingala natif (`masakhanews`) est volontairement exclu du jeu d'entraînement
  (non classable par le modèle zero-shot).
- **Registre écrit** : le dataset est en français écrit ; l'oralité (whatsapp,
  sms, abréviations) est absente.

## 4. Limites

- La classification zero-shot est **automatique** : malgré le seuil 0.70, des
  erreurs d'étiquetage subsistent ; une relecture humaine est recommandée.
- Les questions seed (surtout `RUMEUR_CROYANCE` et `HORS_DOMAINE_CLINIQUE`) sont
  **synthétiques** : elles modélisent des situations plausibles, pas des
  verbatims collectés.
- Taille modeste (~700–800 exemples) : suffisant pour un LoRA, insuffisant pour
  une évaluation robuste de la généralisation inter-dialectes.
- La traduction NLLB n'est pas revue par un locuteur natif : erreurs possibles.
- Licence NLLB **CC-BY-NC-4.0** : les traductions dérivées sont à usage
  **non commercial**, sauf à re-traduire avec un modèle sous licence permissive.

## 5. Considérations éthiques

- **Santé publique** : les réponses de la FAQ sont alignées sur le PEV congolais
  et les recommandations OMS, mais portent la mention **« à valider par un
  médecin »**. Aucune réponse ne doit être servie sans validation clinique.
- **Rumeurs** : les énoncés de `RUMEUR_CROYANCE` reproduisent des rumeurs locales
  (stérilité, empoisonnement…) **à des fins de classification uniquement** ; ils
  sont systématiquement associés à une réponse qui les réfute.
- **Urgences** : `HORS_DOMAINE_CLINIQUE` doit déclencher un **transfert vers un
  agent de santé** ; le modèle ne doit jamais formuler d'avis clinique.
- **Pas de données personnelles** : aucune donnée de patient, identifiant ou
  information identifiable n'est incluse.
- **Consentement & représentativité** : les questions seed sont des créations
  originales ; toute collecte future de données réelles devra passer par un
  comité d'éthique et un consentement éclairé.
