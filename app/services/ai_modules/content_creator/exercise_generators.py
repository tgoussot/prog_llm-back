from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import retry_with_backoff, get_llm, valider_et_corriger_exercices, parse_json_from_text, fast_api_call, ultra_fast_api_call
from .theme_generator import generer_themes_aleatoires

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_comprehension_ecrite(langue, niveau_cible="", themes=None):
    """Génère uniquement la section compréhension écrite avec des QCM"""
    if themes is None:
        themes = generer_themes_aleatoires(langue, nombre=3, categorie="compréhension")
    
    llm = get_llm(temperature=0.8)  # Température élevée pour plus de créativité
    
    # Définir la complexité selon le niveau
    complexite_config = {
        "A1": {
            "longueur_texte": "100-150 mots",
            "types_questions": ["identification directe", "compréhension littérale", "reconnaissance de vocabulaire"],
            "difficulte": "questions dont les réponses sont explicitement mentionnées dans le texte"
        },
        "A2": {
            "longueur_texte": "150-200 mots", 
            "types_questions": ["compréhension littérale", "identification d'informations précises", "compréhension du contexte"],
            "difficulte": "questions nécessitant de faire des liens simples entre les informations"
        },
        "B1": {
            "longueur_texte": "200-250 mots",
            "types_questions": ["compréhension inférentielle", "analyse des intentions", "déduction logique", "compréhension implicite"],
            "difficulte": "questions nécessitant des inférences et la compréhension du sens implicite"
        },
        "B2": {
            "longueur_texte": "250-300 mots",
            "types_questions": ["analyse critique", "évaluation des arguments", "compréhension des nuances", "identification des biais"],
            "difficulte": "questions nécessitant une analyse critique et la compréhension des subtilités"
        },
        "C1": {
            "longueur_texte": "300-350 mots",
            "types_questions": ["analyse approfondie", "évaluation critique", "synthèse d'informations complexes", "identification des implications"],
            "difficulte": "questions nécessitant une compréhension sophistiquée et une analyse approfondie"
        },
        "C2": {
            "longueur_texte": "350-400 mots",
            "types_questions": ["analyse experte", "évaluation multidimensionnelle", "compréhension des nuances culturelles", "critique argumentative"],
            "difficulte": "questions nécessitant une maîtrise experte et une pensée critique avancée"
        }
    }
    
    niveau_final = niveau_cible if niveau_cible in complexite_config else "B1"
    config = complexite_config[niveau_final]
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues spécialisé dans la création d'exercices de compréhension avancée.

Crée EXACTEMENT 2 exercices de compréhension écrite en {langue} de niveau {niveau_cible} avec des QCM sophistiqués.

CONFIGURATION DE DIFFICULTÉ POUR NIVEAU {niveau_cible}:
- Longueur des textes: {longueur_texte}
- Types de questions: {types_questions}
- Niveau de difficulté: {difficulte}

THÈMES IMPOSÉS (développe-les de manière créative et originale):
1. Premier exercice: "{theme1}" 
2. Deuxième exercice: "{theme2}"

IMPORTANT - EXIGENCES DE QUALITÉ:
- Crée des textes AUTHENTIQUES et STIMULANTS intellectuellement
- Varie les genres: article analytique, essai, témoignage, critique, rapport d'expert...
- Les questions doivent tester différents niveaux cognitifs selon le niveau cible
- Évite les questions évidentes - privilégie l'analyse, l'inférence et l'évaluation critique
- Rends les distracteurs plausibles et sophistiqués

RÈGLES DE LANGUE:
- CONSIGNES en FRANÇAIS (langue d'interface)
- QUESTIONS en FRANÇAIS (pour tester la compréhension)
- OPTIONS en FRANÇAIS
- TEXTES PRINCIPAUX en {langue}

TYPES DE QUESTIONS SOPHISTIQUÉES À UTILISER:
- Compréhension inférentielle: "Que peut-on déduire de... ?"
- Analyse des intentions: "Quel est le but de l'auteur quand il... ?"
- Évaluation critique: "Quelle est la faiblesse de l'argument présenté ?"
- Identification des nuances: "Quel ton adopte l'auteur pour... ?"
- Synthèse d'informations: "Comment les différents éléments s'articulent-ils ?"
- Compréhension implicite: "Que suggère l'auteur sans le dire explicitement ?"

STRUCTURE OBLIGATOIRE:
[
  {{
    "consigne": "Lisez attentivement ce texte et répondez aux questions d'analyse en choisissant la meilleure réponse.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Compréhension écrite approfondie - {theme1}",
    "contenu": {{
      "texte_principal": "[TEXTE AUTHENTIQUE ET ENGAGEANT EN {langue} SUR {theme1} - {longueur_texte}]",
      "elements": [
        {{
          "id": 1, 
          "texte": "[QUESTION SOPHISTIQUÉE D'ANALYSE/INFÉRENCE - pas évidente]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Réponse correcte mais nuancée]", "est_correcte": true}},
            {{"id": "B", "texte": "[Distracteur plausible et intelligent]", "est_correcte": false}},
            {{"id": "C", "texte": "[Distracteur sophistiqué]", "est_correcte": false}},
            {{"id": "D", "texte": "[Distracteur trompeur mais logique]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[QUESTION DE COMPRÉHENSION IMPLICITE OU D'ÉVALUATION CRITIQUE]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Distracteur plausible]", "est_correcte": false}},
            {{"id": "B", "texte": "[Réponse correcte nécessitant réflexion]", "est_correcte": true}},
            {{"id": "C", "texte": "[Distracteur intelligent]", "est_correcte": false}},
            {{"id": "D", "texte": "[Distracteur sophistiqué]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[QUESTION D'ANALYSE AVANCÉE OU DE SYNTHÈSE]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Distracteur intelligent]", "est_correcte": false}},
            {{"id": "B", "texte": "[Distracteur plausible]", "est_correcte": false}},
            {{"id": "C", "texte": "[Réponse correcte mais subtile]", "est_correcte": true}},
            {{"id": "D", "texte": "[Distracteur sophistiqué]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }},
  {{
    "consigne": "Analysez ce document et répondez aux questions d'évaluation critique.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Compréhension écrite critique - {theme2}",
    "contenu": {{
      "texte_principal": "[TEXTE COMPLEXE ET NUANCÉ EN {langue} SUR {theme2} - {longueur_texte}]",
      "elements": [
        {{
          "id": 1, 
          "texte": "[QUESTION D'ANALYSE DES INTENTIONS OU DU TON]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Réponse correcte mais nuancée]", "est_correcte": true}},
            {{"id": "B", "texte": "[Distracteur intelligent]", "est_correcte": false}},
            {{"id": "C", "texte": "[Distracteur plausible]", "est_correcte": false}},
            {{"id": "D", "texte": "[Distracteur sophistiqué]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[QUESTION DE DÉDUCTION OU D'INFÉRENCE COMPLEXE]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Distracteur plausible]", "est_correcte": false}},
            {{"id": "B", "texte": "[Réponse correcte nécessitant logique]", "est_correcte": true}},
            {{"id": "C", "texte": "[Distracteur intelligent]", "est_correcte": false}},
            {{"id": "D", "texte": "[Distracteur trompeur]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[QUESTION D'ÉVALUATION CRITIQUE OU DE SYNTHÈSE]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Distracteur sophistiqué]", "est_correcte": false}},
            {{"id": "B", "texte": "[Distracteur plausible]", "est_correcte": false}},
            {{"id": "C", "texte": "[Réponse correcte mais subtile]", "est_correcte": true}},
            {{"id": "D", "texte": "[Distracteur intelligent]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }}
]

CRÉATIVITÉ OBLIGATOIRE:
- Varie les formats de questions pour chaque exercice
- Utilise des distracteurs intelligents qui testent vraiment la compréhension
- Créé des textes engageants avec du vocabulaire riche et varié
- Évite les formulations répétitives entre les exercices
- Adapte la complexité linguistique et cognitive au niveau {niveau_cible}"""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    niveau_cible_final = niveau_cible if niveau_cible else "B1"
    
    # Extraire les thèmes individuellement pour les passer au template
    theme1 = themes[0]
    theme2 = themes[1] if len(themes) > 1 else themes[0]
    theme3 = themes[2] if len(themes) > 2 else themes[0]
    
    config_niveau = complexite_config[niveau_cible_final]
    
    # Faire l'appel à l'API
    chain_text = prompt | llm | StrOutputParser()
    resultat = ultra_fast_api_call(lambda: chain_text.invoke({
        "langue": langue,
        "niveau_str": niveau_str,
        "niveau_cible": niveau_cible_final,
        "theme1": theme1,
        "theme2": theme2,
        "longueur_texte": config_niveau["longueur_texte"],
        "types_questions": ", ".join(config_niveau["types_questions"]),
        "difficulte": config_niveau["difficulte"]
    }))
    
    # Extraire et parser le JSON
    exercices_data = parse_json_from_text(resultat)
    if exercices_data:
        return valider_et_corriger_exercices(exercices_data, "QCM")
    else:
        return []

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_grammaire(langue, niveau_cible=""):
    """Génère uniquement la section grammaire avec des QCM"""
    llm = get_llm(temperature=0.8)
    
    # Configuration de complexité grammaticale par niveau
    complexite_grammaire = {
        "A1": {
            "structures": ["présent simple", "articles définis/indéfinis", "pluriels réguliers", "genre des noms"],
            "types_exercices": ["complétion directe", "identification forme correcte", "accord basic"],
            "complexite_phrases": "courtes et simples",
            "difficulte": "règles grammaticales élémentaires avec application directe"
        },
        "A2": {
            "structures": ["temps du passé", "futur proche", "comparatifs", "pronoms personnels", "adjectifs possessifs"],
            "types_exercices": ["transformation temporelle", "choix de pronoms", "accords complexes"],
            "complexite_phrases": "moyennes avec subordination simple",
            "difficulte": "règles grammaticales intermédiaires nécessitant réflexion"
        },
        "B1": {
            "structures": ["subjonctif", "gérondif", "voix passive", "pronoms relatifs", "expression de l'hypothèse"],
            "types_exercices": ["analyse de nuances", "transformation syntaxique", "détection d'erreurs subtiles"],
            "complexite_phrases": "complexes avec subordinations multiples",
            "difficulte": "maîtrise de structures avancées et subtilités grammaticales"
        },
        "B2": {
            "structures": ["concordance des temps", "style indirect", "participe présent", "propositions infinitives"],
            "types_exercices": ["analyse stylistique", "transformation complexe", "nuances de registre"],
            "complexite_phrases": "sophistiquées avec imbrications syntaxiques",
            "difficulte": "distinction de nuances grammaticales et stylistiques"
        },
        "C1": {
            "structures": ["subjontif imparfait", "gérondif composé", "participe absolu", "inversion stylistique"],
            "types_exercices": ["analyse syntaxique experte", "correction de style", "registres de langue"],
            "complexite_phrases": "très complexes avec structures littéraires",
            "difficulte": "maîtrise experte des subtilités et registres grammaticaux"
        },
        "C2": {
            "structures": ["archaïsmes grammaticaux", "structures littéraires", "variations dialectales", "registres spécialisés"],
            "types_exercices": ["analyse philologique", "correction experte", "identification de registres"],
            "complexite_phrases": "hautement sophistiquées avec références culturelles",
            "difficulte": "expertise grammaticale et connaissance des variations linguistiques"
        }
    }
    
    niveau_final = niveau_cible if niveau_cible in complexite_grammaire else "B1"
    config_gram = complexite_grammaire[niveau_final]
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues et en linguistique appliquée, spécialisé dans la création d'exercices grammaticaux sophistiqués.

Crée EXACTEMENT 3 exercices de grammaire en {langue} de niveau {niveau_cible} avec des QCM avancés.

CONFIGURATION GRAMMATICALE NIVEAU {niveau_cible}:
- Structures ciblées: {structures}
- Types d'exercices: {types_exercices}
- Complexité des phrases: {complexite_phrases}
- Niveau de difficulté: {difficulte}

EXIGENCES DE SOPHISTICATION:
- Évite les exercices mécaniques et répétitifs
- Teste la COMPRÉHENSION des règles, pas la mémorisation
- Intègre des contextes authentiques et variés
- Utilise des distracteurs linguistiquement plausibles
- Varie les genres textuels (formel, informel, littéraire, technique)

RÈGLES LINGUISTIQUES:
- CONSIGNES en FRANÇAIS (interface utilisateur)
- EXERCICES, QUESTIONS et OPTIONS en {langue}
- Contextualisations appropriées au niveau culturel

TYPOLOGIE D'EXERCICES SOPHISTIQUÉS:
1. ANALYSE CONTEXTUELLE: "Dans quel contexte cette forme est-elle appropriée ?"
2. TRANSFORMATION NUANCÉE: "Comment exprimer la même idée avec une autre structure ?"
3. DÉTECTION D'ERREURS SUBTILES: "Quelle phrase présente une nuance incorrecte ?"
4. CHOIX STYLISTIQUE: "Quelle formulation convient au registre demandé ?"
5. ANALYSE SÉMANTIQUE: "Quelle structure change le sens ?"

STRUCTURE OBLIGATOIRE:

[
  {{
    "consigne": "Analysez le contexte et choisissez la structure grammaticale la plus appropriée.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Grammaire contextuelle - {structures_principales}",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{
          "id": 1, 
          "texte": "[ANALYSE CONTEXTUELLE en {langue}: Phrase complexe avec choix grammatical nuancé]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Structure correcte avec justification contextuelle]", "est_correcte": true}},
            {{"id": "B", "texte": "[Structure grammaticalement valide mais inappropriée au contexte]", "est_correcte": false}},
            {{"id": "C", "texte": "[Structure avec erreur subtile de registre]", "est_correcte": false}},
            {{"id": "D", "texte": "[Structure formellement incorrecte]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[TRANSFORMATION STYLISTIQUE en {langue}: Comment reformuler cette phrase ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Transformation inadéquate]", "est_correcte": false}},
            {{"id": "B", "texte": "[Transformation correcte avec nuance appropriée]", "est_correcte": true}},
            {{"id": "C", "texte": "[Transformation avec changement de sens]", "est_correcte": false}},
            {{"id": "D", "texte": "[Transformation grammaticalement incorrecte]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[NUANCES GRAMMATICALES en {langue}: Quelle phrase exprime le mieux cette idée ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Expression avec registre inapproprié]", "est_correcte": false}},
            {{"id": "B", "texte": "[Expression ambiguë]", "est_correcte": false}},
            {{"id": "C", "texte": "[Expression précise et appropriée]", "est_correcte": true}},
            {{"id": "D", "texte": "[Expression avec erreur de nuance]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }},
  {{
    "consigne": "Évaluez la correction et l'appropriateness stylistique de ces structures.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Grammaire stylistique - Registres de langue",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{
          "id": 1, 
          "texte": "[ANALYSE DE REGISTRE en {langue}: Quelle formulation convient à un contexte formel ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Formulation appropriée au registre formel]", "est_correcte": true}},
            {{"id": "B", "texte": "[Formulation trop familière]", "est_correcte": false}},
            {{"id": "C", "texte": "[Formulation archaïque]", "est_correcte": false}},
            {{"id": "D", "texte": "[Formulation grammaticalement incorrecte]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[SUBTILITÉ SYNTAXIQUE en {langue}: Quelle structure change le sens de la phrase ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Structure équivalente sémantiquement]", "est_correcte": false}},
            {{"id": "B", "texte": "[Structure modifiant le sens de manière significative]", "est_correcte": true}},
            {{"id": "C", "texte": "[Structure synonyme]", "est_correcte": false}},
            {{"id": "D", "texte": "[Structure identique avec variante stylistique]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[COHÉRENCE TEXTUELLE en {langue}: Quelle phrase assure la meilleure cohésion ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Phrase avec rupture de cohérence]", "est_correcte": false}},
            {{"id": "B", "texte": "[Phrase répétitive]", "est_correcte": false}},
            {{"id": "C", "texte": "[Phrase assurant une transition fluide]", "est_correcte": true}},
            {{"id": "D", "texte": "[Phrase hors contexte]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }},
  {{
    "consigne": "Identifiez les erreurs grammaticales subtiles et les inadéquations stylistiques.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Correction experte - Détection d'erreurs avancées",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{
          "id": 1, 
          "texte": "[DÉTECTION AVANCÉE en {langue}: Quelle phrase contient une erreur subtile ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Phrase parfaitement correcte]", "est_correcte": false}},
            {{"id": "B", "texte": "[Phrase avec erreur subtile de concordance]", "est_correcte": true}},
            {{"id": "C", "texte": "[Phrase correcte mais style différent]", "est_correcte": false}},
            {{"id": "D", "texte": "[Phrase correcte avec registre approprié]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 2, 
          "texte": "[INADÉQUATION CONTEXTUELLE en {langue}: Quelle phrase est inappropriée au contexte ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Phrase adaptée au contexte]", "est_correcte": false}},
            {{"id": "B", "texte": "[Phrase neutre]", "est_correcte": false}},
            {{"id": "C", "texte": "[Phrase avec registre inadéquat au contexte]", "est_correcte": true}},
            {{"id": "D", "texte": "[Phrase parfaitement appropriée]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }},
        {{
          "id": 3, 
          "texte": "[SOPHISTICATION LINGUISTIQUE en {langue}: Quelle version est la plus raffinée ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Version basique correcte]", "est_correcte": false}},
            {{"id": "B", "texte": "[Version avec complexité artificielle]", "est_correcte": false}},
            {{"id": "C", "texte": "[Version élégante et sophistiquée]", "est_correcte": true}},
            {{"id": "D", "texte": "[Version verbale et lourde]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }}
]

CRÉATIVITÉ GRAMMATICALE OBLIGATOIRE:
- Varie les structures grammaticales ciblées selon le niveau
- Intègre des contextes professionnels, académiques, artistiques variés
- Teste la compréhension conceptuelle, pas seulement la règle
- Crée des distracteurs intelligents révélant les erreurs fréquentes
- Adapte la complexité syntaxique au niveau {niveau_cible}"""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    niveau_cible_final = niveau_cible if niveau_cible else "B1"
    
    # Faire l'appel à l'API
    chain_text = prompt | llm | StrOutputParser()
    resultat = fast_api_call(lambda: chain_text.invoke({
        "langue": langue,
        "niveau_str": niveau_str,
        "niveau_cible": niveau_cible_final,
        "structures": ", ".join(config_gram["structures"]),
        "types_exercices": ", ".join(config_gram["types_exercices"]),
        "complexite_phrases": config_gram["complexite_phrases"],
        "difficulte": config_gram["difficulte"],
        "structures_principales": config_gram["structures"][0] + " et " + config_gram["structures"][1] if len(config_gram["structures"]) > 1 else config_gram["structures"][0]
    }))
    
    # Extraire et parser le JSON
    exercices_data = parse_json_from_text(resultat)
    if exercices_data:
        return valider_et_corriger_exercices(exercices_data, "QCM")
    else:
        return []

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_vocabulaire(langue, niveau_cible="", domaines=None):
    """Génère uniquement la section vocabulaire avec des QCM"""
    if domaines is None:
        domaines = generer_themes_aleatoires(langue, nombre=3, categorie="domaines")
    
    llm = get_llm(temperature=0.8)
    
    # Configuration de complexité lexicale par niveau
    complexite_vocabulaire = {
        "A1": {
            "types_lexique": ["vocabulaire concret", "objets quotidiens", "actions basiques", "adjectifs simples"],
            "strategies": ["reconnaissance directe", "association image-mot", "définitions simples"],
            "complexite_context": "phrases courtes et contexte évident",
            "difficulte": "reconnaissance du vocabulaire de base avec contexte explicite"
        },
        "A2": {
            "types_lexique": ["vocabulaire thématique", "expressions courantes", "synonymes básicos", "famille de mots"],
            "strategies": ["déduction contextuelle", "associations logiques", "oppositions sémantiques"],
            "complexite_context": "phrases moyennes avec indices contextuels",
            "difficulte": "compréhension du vocabulaire avec aide contextuelle"
        },
        "B1": {
            "types_lexique": ["vocabulaire abstrait", "nuances sémantiques", "registres de langue", "expressions idiomatiques"],
            "strategies": ["inférence contextuelle", "analyse des nuances", "distinction de registres"],
            "complexite_context": "textes complexes nécessitant déduction",
            "difficulte": "maîtrise de vocabulaire nuancé et spécialisé"
        },
        "B2": {
            "types_lexique": ["terminologie spécialisée", "métaphores", "connotations", "vocabulaire technique"],
            "strategies": ["analyse sémantique fine", "compréhension implicite", "interprétation figurée"],
            "complexite_context": "textes sophistiqués avec implicites",
            "difficulte": "distinction de nuances subtiles et registres spécialisés"
        },
        "C1": {
            "types_lexique": ["vocabulaire savant", "archaïsmes", "néologismes", "jargons professionnels"],
            "strategies": ["analyse étymologique", "compréhension culturelle", "maîtrise stylistique"],
            "complexite_context": "textes littéraires et académiques complexes",
            "difficulte": "expertise lexicale et culturelle approfondie"
        },
        "C2": {
            "types_lexique": ["vocabulaire érudit", "variations dialectales", "références culturelles", "lexique historique"],
            "strategies": ["analyse philologique", "compréhension culturelle experte", "maîtrise diachronique"],
            "complexite_context": "textes hautement spécialisés et références implicites",
            "difficulte": "maîtrise experte du lexique dans toute sa richesse et complexité"
        }
    }
    
    niveau_final = niveau_cible if niveau_cible in complexite_vocabulaire else "B1"
    config_vocab = complexite_vocabulaire[niveau_final]
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues et en lexicologie, spécialisé dans la création d'exercices lexicaux sophistiqués.

Crée EXACTEMENT 2 exercices de vocabulaire en {langue} de niveau {niveau_cible} avec des QCM avancés.

CONFIGURATION LEXICALE NIVEAU {niveau_cible}:
- Types de vocabulaire: {types_lexique}
- Stratégies cognitives: {strategies}
- Complexité contextuelle: {complexite_context}
- Niveau de difficulté: {difficulte}

DOMAINES CIBLÉS:
- Premier exercice: "{domaine1}"
- Deuxième exercice: "{domaine2}"

EXIGENCES DE SOPHISTICATION LEXICALE:
- Évite le vocabulaire basique et prévisible
- Teste la COMPRÉHENSION sémantique, pas la mémorisation
- Intègre des contextes authentiques et culturellement riches
- Utilise des distracteurs sémantiquement plausibles
- Varie les champs lexicaux et registres de langue

RÈGLES LINGUISTIQUES:
- CONSIGNES en FRANÇAIS (interface utilisateur)
- EXERCICES, QUESTIONS et OPTIONS en {langue}
- Contextualisations culturellement appropriées

TYPOLOGIE D'EXERCICES LEXICAUX AVANCÉS:
1. ANALYSE CONTEXTUELLE: "Quel mot convient le mieux dans ce contexte spécialisé ?"
2. NUANCES SÉMANTIQUES: "Quelle différence de sens distingue ces termes ?"
3. REGISTRES DE LANGUE: "Quel terme convient au registre demandé ?"
4. CONNOTATIONS: "Quel mot porte la connotation appropriée ?"
5. COLLOCATIONS: "Quelle combinaison lexicale est idiomatique ?"

STRUCTURE OBLIGATOIRE:

[
  {{
    "consigne": "Analysez le vocabulaire spécialisé et choisissez le terme le plus approprié au contexte.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Vocabulaire spécialisé - {domaine1}",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{
          "id": 1, 
          "texte": "[ANALYSE CONTEXTUELLE en {langue}: Phrase complexe de {domaine1} avec choix lexical nuancé]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Terme spécialisé correct et précis]", "est_correcte": true}},
            {{"id": "B", "texte": "[Terme général mais plausible]", "est_correcte": false}},
            {{"id": "C", "texte": "[Terme proche mais nuance incorrecte]", "est_correcte": false}},
            {{"id": "D", "texte": "[Terme with registre inapproprié]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[NUANCES SÉMANTIQUES en {langue}: Quelle nuance de sens distingue ces termes de {domaine1} ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Terme avec connotation inappropriée]", "est_correcte": false}},
            {{"id": "B", "texte": "[Terme avec la nuance sémantique correcte]", "est_correcte": true}},
            {{"id": "C", "texte": "[Terme synonyme mais niveau inadéquat]", "est_correcte": false}},
            {{"id": "D", "texte": "[Terme avec registre incorrect]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[COLLOCATIONS SPÉCIALISÉES en {langue}: Quelle combinaison lexicale est idiomatique en {domaine1} ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Collocation artificielle]", "est_correcte": false}},
            {{"id": "B", "texte": "[Collocation calquée sur une autre langue]", "est_correcte": false}},
            {{"id": "C", "texte": "[Collocation authentique et idiomatique]", "est_correcte": true}},
            {{"id": "D", "texte": "[Combinaison grammaticale mais non idiomatique]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }},
  {{
    "consigne": "Évaluez les nuances lexicales et les registres de langue appropriés.",
    "niveau_cible": "{niveau_cible}",
    "competence": "Maîtrise lexicale avancée - {domaine2}",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{
          "id": 1, 
          "texte": "[REGISTRES DE LANGUE en {langue}: Quel terme convient au registre académique de {domaine2} ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Terme académique et précis]", "est_correcte": true}},
            {{"id": "B", "texte": "[Terme familier mais courant]", "est_correcte": false}},
            {{"id": "C", "texte": "[Terme technique mais hors contexte]", "est_correcte": false}},
            {{"id": "D", "texte": "[Terme vague et imprécis]", "est_correcte": false}}
          ],
          "reponse_correcte": "A"
        }},
        {{
          "id": 2, 
          "texte": "[CONNOTATIONS en {langue}: Quel terme porte la connotation appropriée au contexte de {domaine2} ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Terme à connotation négative inappropriée]", "est_correcte": false}},
            {{"id": "B", "texte": "[Terme avec connotation positive appropriée]", "est_correcte": true}},
            {{"id": "C", "texte": "[Terme neutre mais insuffisant]", "est_correcte": false}},
            {{"id": "D", "texte": "[Terme with connotation ambiguë]", "est_correcte": false}}
          ],
          "reponse_correcte": "B"
        }},
        {{
          "id": 3, 
          "texte": "[PRÉCISION LEXICALE en {langue}: Quel terme exprime le plus précisément cette idée en {domaine2} ?]", 
          "type": "QCM",
          "options": [
            {{"id": "A", "texte": "[Terme générique]", "est_correcte": false}},
            {{"id": "B", "texte": "[Terme approximatif]", "est_correcte": false}},
            {{"id": "C", "texte": "[Terme précis et spécialisé]", "est_correcte": true}},
            {{"id": "D", "texte": "[Terme technique mais inapproprié]", "est_correcte": false}}
          ],
          "reponse_correcte": "C"
        }}
      ]
    }}
  }}
]

CRÉATIVITÉ LEXICALE OBLIGATOIRE:
- Varie les champs sémantiques selon les domaines ciblés
- Intègre du vocabulaire culturellement authentique
- Teste la compréhension fine des nuances et registres
- Crée des distracteurs lexicalement sophistiqués
- Adapte la richesse lexicale au niveau {niveau_cible}"""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    niveau_cible_final = niveau_cible if niveau_cible else "B1"
    
    # Extraire les domaines individuellement pour les passer au template
    domaine1 = domaines[0]
    domaine2 = domaines[1] if len(domaines) > 1 else domaines[0]
    
    # Faire l'appel à l'API
    chain_text = prompt | llm | StrOutputParser()
    resultat = fast_api_call(lambda: chain_text.invoke({
        "langue": langue,
        "niveau_str": niveau_str,
        "niveau_cible": niveau_cible_final,
        "domaine1": domaine1,
        "domaine2": domaine2,
        "types_lexique": ", ".join(config_vocab["types_lexique"]),
        "strategies": ", ".join(config_vocab["strategies"]),
        "complexite_context": config_vocab["complexite_context"],
        "difficulte": config_vocab["difficulte"]
    }))
    
    # Extraire et parser le JSON
    exercices_data = parse_json_from_text(resultat)
    if exercices_data:
        return valider_et_corriger_exercices(exercices_data, "QCM")
    else:
        return [] 