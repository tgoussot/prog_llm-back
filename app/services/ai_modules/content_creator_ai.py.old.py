from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal
from langchain_core.output_parsers import StrOutputParser
from enum import Enum
import os
from dotenv import load_dotenv
import time
import random
import re
from functools import wraps

# Charger les variables d'environnement mais également définir une clé par défaut si absente
load_dotenv()

# Obtenir la clé API depuis les variables d'environnement ou utiliser la clé par défaut
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HWeyYbCcjhpjneQZmq5iszsQLil08omZ")

def retry_with_backoff(max_retries=3, base_delay=10):
    """Décorateur pour retry avec backoff exponentiel en cas d'erreur de rate limit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt) + random.uniform(1, 5)
                            print(f"Rate limit atteint, attente de {delay:.1f} secondes avant retry {attempt + 1}/{max_retries}")
                            time.sleep(delay)
                            continue
                    raise e
            return None
        return wrapper
    return decorator

def safe_api_call(func, *args, **kwargs):
    """Fonction utilitaire pour faire des appels API sécurisés avec délai"""
    try:
        # Délai aléatoire entre 2 et 5 secondes avant chaque appel
        delay = random.uniform(2, 5)
        print(f"Attente de {delay:.1f} secondes avant appel API...")
        time.sleep(delay)
        
        result = func(*args, **kwargs)
        
        # Délai après l'appel réussi
        post_delay = random.uniform(1, 3)
        print(f"Appel réussi, attente de {post_delay:.1f} secondes...")
        time.sleep(post_delay)
        
        return result
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"Rate limit détecté: {e}")
            # Attendre plus longtemps en cas de rate limit
            time.sleep(30)
        raise e

# Définition du type d'élément dans un exercice
class TypeElement(str, Enum):
    QUESTION = "QUESTION"
    PHRASE = "PHRASE"
    ITEM = "ITEM"
    CONSIGNE = "CONSIGNE"

# Définition des modèles de données
class Element(BaseModel):
    id: int = Field(..., description="Identifiant unique de l'élément")
    texte: str = Field(..., description="Texte de la question, de la phrase ou de l'item")
    type: TypeElement = Field(..., description="Type de l'élément (QUESTION, PHRASE, ITEM, CONSIGNE)")

class Contenu(BaseModel):
    texte_principal: str = Field("", description="Texte principal (peut être vide pour grammaire/vocabulaire)")
    elements: List[Element] = Field(..., description="Liste des éléments (questions, phrases, items) de l'exercice")

class Exercice(BaseModel):
    consigne: str = Field(..., description="Consigne claire de l'exercice")
    contenu: Contenu = Field(..., description="Contenu structuré de l'exercice")
    niveau_cible: str = Field(..., description="Niveau CECRL ciblé (A1-C2)")
    competence: str = Field(..., description="Compétence précise évaluée")

class TestDeCompetence(BaseModel):
    comprehension_ecrite: List[Exercice] = Field(default_factory=list)
    expression_ecrite: List[Exercice] = Field(default_factory=list)
    grammaire: List[Exercice] = Field(default_factory=list)
    vocabulaire: List[Exercice] = Field(default_factory=list)

class ExercicePersonnalise(BaseModel):
    consigne: str
    contenu: Contenu
    niveau_cible: str
    competence: str
    lacune_ciblee: str = Field(..., description="Lacune spécifique ciblée par l'exercice")

class TermesTraduction(BaseModel):
    comprehension_ecrite: str = Field(..., description="Traduction de 'Compréhension écrite'")
    expression_ecrite: str = Field(..., description="Traduction de 'Expression écrite'")
    grammaire: str = Field(..., description="Traduction de 'Grammaire'")
    vocabulaire: str = Field(..., description="Traduction de 'Vocabulaire'")
    consigne: str = Field(..., description="Traduction de 'Consigne'")
    contenu: str = Field(..., description="Traduction de 'Contenu'")
    niveau_cible: str = Field(..., description="Traduction de 'Niveau cible'")
    competence: str = Field(..., description="Traduction de 'Compétence'")
    question: str = Field(..., description="Traduction de 'Question'")
    phrase: str = Field(..., description="Traduction de 'Phrase'")
    item: str = Field(..., description="Traduction de 'Item'")
    texte_principal: str = Field(..., description="Traduction de 'Texte principal'")

def traduire_termes_techniques(langue_cible):
    """Traduit les termes techniques du système dans la langue cible."""
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1,
        api_key=MISTRAL_API_KEY
    )
    
    # Si la langue cible est déjà le français, retourner les termes en français
    if langue_cible.lower() == "français":
        return TermesTraduction(
            comprehension_ecrite="Compréhension écrite",
            expression_ecrite="Expression écrite",
            grammaire="Grammaire",
            vocabulaire="Vocabulaire",
            consigne="Consigne",
            contenu="Contenu",
            niveau_cible="Niveau cible",
            competence="Compétence",
            question="Question",
            phrase="Phrase",
            item="Item",
            texte_principal="Texte principal"
        )
        
    # Traduire les termes vers la langue cible
    prompt = ChatPromptTemplate.from_template(
        """Tu es un traducteur technique spécialisé dans la didactique des langues.
        Traduis précisément les termes suivants du français vers {langue_cible}.
        
        Termes à traduire:
        - Compréhension écrite
        - Expression écrite
        - Grammaire
        - Vocabulaire
        - Consigne
        - Contenu
        - Niveau cible
        - Compétence
        - Question
        - Phrase
        - Item
        - Texte principal
        
        Fournis tes traductions au format structuré uniquement, sans explications.
        """
    )
    
    chain = prompt | llm.with_structured_output(TermesTraduction)
    return chain.invoke({"langue_cible": langue_cible})

def traduire_prompt(prompt_texte, termes, langue_cible):
    """Traduit un prompt vers la langue cible."""
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1,
        api_key=MISTRAL_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un traducteur expert dans la didactique des langues.
        Traduis précisément le texte suivant du français vers {langue_cible}.
        
        Voici les traductions des termes techniques à utiliser:
        {termes}
        
        Texte à traduire:
        {texte}
        
        La traduction doit être fidèle et naturelle dans la langue cible.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "texte": prompt_texte, 
        "langue_cible": langue_cible,
        "termes": "\n".join([f"- {k}: {v}" for k, v in termes.__dict__.items() if not k.startswith("_")])
    })

@retry_with_backoff(max_retries=3, base_delay=15)
def generer_themes_aleatoires(langue="français", nombre=2, categorie="compréhension"):
    """Génère des thèmes aléatoires à l'aide de l'IA plutôt que d'utiliser des listes prédéfinies"""
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=1.0,  # Température élevée pour maximiser la créativité
        api_key=MISTRAL_API_KEY
    )
    
    # Thèmes à éviter car surreprésentés
    themes_a_eviter = ["gastronomie", "cuisine", "nourriture", "plats", "alimentation", 
                       "traditions culinaires", "spécialités régionales", "marchés de Noël"]
    
    # Descriptions en français qui seront traduites si nécessaire
    descriptions_fr = {
        "compréhension": "thèmes variés pour des textes de compréhension écrite",
        "domaines": "domaines lexicaux pour des exercices de vocabulaire"
    }
    
    # Obtenir la description en français ou utiliser une description générique si la catégorie n'existe pas
    description_fr = descriptions_fr.get(categorie, "thèmes variés pour des exercices de langue")
    
    # Traduire la description si nécessaire (sauf pour le français)
    if langue.lower() == "français":
        description = description_fr
    else:
        # Utiliser l'IA pour traduire la description
        prompt_traduction = ChatPromptTemplate.from_template(
            """Traduis précisément cette phrase du français vers {langue}:
            
            Phrase: "{texte}"
            
            Donne UNIQUEMENT la traduction sans aucune autre explication."""
        )
        
        chaine_traduction = prompt_traduction | llm | StrOutputParser()
        description = safe_api_call(
            chaine_traduction.invoke,
            {"langue": langue, "texte": description_fr}
        ).strip()
    
    # Thèmes de secours variés en cas d'échec de l'IA
    themes_secours = [
        "les nouvelles technologies", "le changement climatique", "l'urbanisation", 
        "les sports extrêmes", "le tourisme responsable", "l'art contemporain",
        "l'intelligence artificielle", "les transports du futur", "la biodiversité", 
        "l'apprentissage des langues", "la psychologie positive", "le télétravail",
        "les réseaux sociaux", "la littérature fantastique", "l'exploration spatiale",
        "les métiers d'avenir", "la musique électronique", "l'architecture moderne",
        "le développement personnel", "les énergies renouvelables", "la médecine préventive"
    ]
    
    # Création du prompt pour générer des thèmes
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Tu es un expert en didactique des langues spécialisé dans la création de matériel pédagogique 
        en {langue}. Ta tâche est de générer {nombre} {description} qui sont originaux, 
        spécifiques et adaptés à un contexte d'apprentissage."""),
        ("human", f"""Génère {nombre} {description} originaux et variés.

        Ces thèmes doivent:
        1. Être authentiques et représentatifs de la culture des locuteurs de {langue}
        2. Être suffisamment spécifiques et non génériques
        3. Couvrir différents domaines de connaissances (art, science, société, technologie, etc.)
        4. Être adaptés à l'enseignement des langues
        5. IMPORTANT: Évite ABSOLUMENT les thèmes suivants qui sont surreprésentés: {', '.join(themes_a_eviter)}
        
        Réponds UNIQUEMENT avec une liste de {nombre} thèmes TRÈS DIFFÉRENTS les uns des autres, un par ligne, sans numérotation ni autres explications.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({})
    
    # Nettoyer le résultat et le transformer en liste
    themes = [theme.strip() for theme in result.strip().split('\n') if theme.strip()]
    
    # Vérifier si nous avons assez de thèmes
    if len(themes) < nombre:
        # Utiliser des thèmes de secours si nécessaire
        random.shuffle(themes_secours)
        themes.extend(themes_secours[:nombre - len(themes)])
    
    # Limiter au nombre demandé au cas où l'IA en génère plus
    return themes[:nombre]

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_comprehension_ecrite(langue, niveau_cible="", themes=None):
    """Génère uniquement la section compréhension écrite"""
    if themes is None:
        themes = generer_themes_aleatoires(langue, nombre=2, categorie="compréhension")
    
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=1.0,  # Température augmentée à 1.0 pour maximiser la variété
        api_key=MISTRAL_API_KEY
    )
    
    # Pour simplifier, on crée une classe représentant les exercices de compréhension écrite
    class ExercicesComprehension(BaseModel):
        exercices: List[Exercice] = Field(..., description="Liste des exercices de compréhension écrite")
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues.

Crée 2 exercices de compréhension écrite en {langue}{niveau_str}.

IMPORTANT - RÈGLES DE LANGUE:
- Les CONSIGNES doivent être en FRANÇAIS (langue d'interface)
- Les QUESTIONS doivent être en FRANÇAIS (pour tester la compréhension)
- Seuls les TEXTES PRINCIPAUX doivent être en {langue}
- Les COMPÉTENCES peuvent être en français

THÈMES IMPOSÉS (respecte-les strictement):
1. Premier exercice: texte informatif/journalistique sur "{theme1}" - RESPECTE STRICTEMENT CE THÈME!
2. Deuxième exercice: dialogue ou texte narratif sur "{theme2}" - RESPECTE STRICTEMENT CE THÈME!

IMPORTANT - CHAQUE ÉLÉMENT DOIT AVOIR OBLIGATOIREMENT LE CHAMP "type":
- Pour les questions: "type": "QUESTION"
- Pour les phrases: "type": "PHRASE" 
- Pour les items: "type": "ITEM"

Voici EXACTEMENT le format à suivre (tu dois créer 2 exercices qui suivent STRICTEMENT cette structure JSON) :
```
[
  {{
    "consigne": "Lisez le texte suivant et répondez aux questions.",
    "niveau_cible": "B1",
    "competence": "Compréhension écrite - Texte informatif",
    "contenu": {{
      "texte_principal": "[TEXTE EN {langue} SUR LE THÈME {theme1}]",
      "elements": [
        {{"id": 1, "texte": "Quel est le sujet principal du texte ?", "type": "QUESTION"}},
        {{"id": 2, "texte": "Quels éléments spécifiques sont mentionnés ?", "type": "QUESTION"}},
        {{"id": 3, "texte": "Quelle est la conclusion du texte ?", "type": "QUESTION"}}
      ]
    }}
  }},
  {{
    "consigne": "Lisez ce dialogue et répondez aux questions.",
    "niveau_cible": "B1",
    "competence": "Compréhension écrite - Dialogue",
    "contenu": {{
      "texte_principal": "[DIALOGUE EN {langue} SUR LE THÈME {theme2}]",
      "elements": [
        {{"id": 1, "texte": "Où se déroule cette conversation ?", "type": "QUESTION"}},
        {{"id": 2, "texte": "Que veulent faire les personnages ?", "type": "QUESTION"}},
        {{"id": 3, "texte": "Quel est le résultat de la conversation ?", "type": "QUESTION"}}
      ]
    }}
  }}
]
```

VÉRIFICATION FINALE OBLIGATOIRE:
- Es-tu CERTAIN que les CONSIGNES sont en FRANÇAIS ?
- Es-tu CERTAIN que les QUESTIONS sont en FRANÇAIS ?
- Es-tu CERTAIN que seuls les TEXTES PRINCIPAUX sont en {langue} ?
- Es-tu CERTAIN que CHAQUE élément a un champ "type" avec une valeur "QUESTION", "PHRASE" ou "ITEM"?
- Es-tu CERTAIN d'avoir respecté STRICTEMENT les thèmes "{theme1}" et "{theme2}"?
- As-tu gardé EXACTEMENT la même structure avec les mêmes champs?

Retourne UNIQUEMENT les exercices au format structuré demandé, rien de plus."""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    
    # Extraire les thèmes individuellement pour les passer au template
    theme1 = themes[0]
    theme2 = themes[1] if len(themes) > 1 else themes[0]
    
    try:
        # Utiliser structured_output directement
        structured_llm = llm.with_structured_output(ExercicesComprehension)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "langue": langue,
            "niveau_str": niveau_str,
            "theme1": theme1,
            "theme2": theme2
        })
        
        return result.exercices
        
    except Exception as e:
        print(f"Erreur lors de la génération structurée: {e}")
        print("Tentative de génération avec validation manuelle...")
        
        # Fallback: générer du texte brut et le parser manuellement
        chain_text = prompt | llm | StrOutputParser()
        result_text = chain_text.invoke({
            "langue": langue,
            "niveau_str": niveau_str,
            "theme1": theme1,
            "theme2": theme2
        })
        
        # Essayer de parser le JSON manuellement
        try:
            import json
            import re
            
            # Extraire le JSON du texte
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                exercices_data = json.loads(json_str)
                
                # Valider et corriger les exercices
                exercices_corriges = valider_et_corriger_exercices(exercices_data, "QUESTION")
                return exercices_corriges
            else:
                print("Impossible d'extraire le JSON du résultat")
                return []
                
        except Exception as parse_error:
            print(f"Erreur lors du parsing manuel: {parse_error}")
            return []

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_grammaire(langue, niveau_cible=""):
    """Génère uniquement la section grammaire"""
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.9,
        api_key=MISTRAL_API_KEY
    )
    
    # Pour simplifier, on crée une classe représentant les exercices de grammaire
    class ExercicesGrammaire(BaseModel):
        exercices: List[Exercice] = Field(..., description="Liste des exercices de grammaire")
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues.

Crée 3 exercices de grammaire en {langue}{niveau_str}.

IMPORTANT - RÈGLES DE LANGUE:
- Les CONSIGNES doivent être en FRANÇAIS (langue d'interface)
- Les COMPÉTENCES peuvent être en français
- Les contenus des exercices (phrases à transformer, corriger, etc.) doivent être dans la langue {langue}

IMPORTANT - CHAQUE ÉLÉMENT DOIT AVOIR OBLIGATOIREMENT LE CHAMP "type":
- Pour les questions: "type": "QUESTION"
- Pour les phrases: "type": "PHRASE" 
- Pour les items: "type": "ITEM"

Consignes:
1. Premier exercice: conjugaison de verbes
2. Deuxième exercice: transformation de phrases
3. Troisième exercice: correction d'erreurs

Voici EXACTEMENT le format à suivre (tu dois créer 3 exercices qui suivent STRICTEMENT cette structure JSON) :
```
[
  {{
    "consigne": "Conjuguez les verbes entre parenthèses au présent de l'indicatif.",
    "niveau_cible": "B1",
    "competence": "Conjugaison - Présent",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{"id": 1, "texte": "[PHRASE EN {langue} AVEC VERBE À CONJUGUER]", "type": "PHRASE"}},
        {{"id": 2, "texte": "[PHRASE EN {langue} AVEC VERBE À CONJUGUER]", "type": "PHRASE"}},
        {{"id": 3, "texte": "[PHRASE EN {langue} AVEC VERBE À CONJUGUER]", "type": "PHRASE"}}
      ]
    }}
  }},
  {{
    "consigne": "Transformez les phrases suivantes au passé composé.",
    "niveau_cible": "B1",
    "competence": "Grammaire - Passé composé",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{"id": 1, "texte": "[PHRASE EN {langue} À TRANSFORMER] → _____", "type": "PHRASE"}},
        {{"id": 2, "texte": "[PHRASE EN {langue} À TRANSFORMER] → _____", "type": "PHRASE"}},
        {{"id": 3, "texte": "[PHRASE EN {langue} À TRANSFORMER] → _____", "type": "PHRASE"}}
      ]
    }}
  }},
  {{
    "consigne": "Corrigez les erreurs dans les phrases suivantes.",
    "niveau_cible": "B1",
    "competence": "Grammaire - Correction d'erreurs",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{"id": 1, "texte": "[PHRASE EN {langue} AVEC ERREUR]", "type": "PHRASE"}},
        {{"id": 2, "texte": "[PHRASE EN {langue} AVEC ERREUR]", "type": "PHRASE"}},
        {{"id": 3, "texte": "[PHRASE EN {langue} AVEC ERREUR]", "type": "PHRASE"}}
      ]
    }}
  }}
]
```

VÉRIFICATION FINALE OBLIGATOIRE:
- Es-tu CERTAIN que les CONSIGNES sont en FRANÇAIS ?
- Es-tu CERTAIN que CHAQUE élément a un champ "type" avec une valeur "QUESTION", "PHRASE" ou "ITEM"?
- As-tu gardé EXACTEMENT la même structure avec les mêmes champs?
- As-tu inclus tous les champs: consigne, niveau_cible, competence, contenu (avec texte_principal et elements)?

Retourne UNIQUEMENT les exercices au format structuré demandé, rien de plus."""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    
    try:
        # Utiliser structured_output directement
        structured_llm = llm.with_structured_output(ExercicesGrammaire)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "langue": langue,
            "niveau_str": niveau_str
        })
        
        return result.exercices
        
    except Exception as e:
        print(f"Erreur lors de la génération structurée (grammaire): {e}")
        print("Tentative de génération avec validation manuelle...")
        
        # Fallback: générer du texte brut et le parser manuellement
        chain_text = prompt | llm | StrOutputParser()
        result_text = chain_text.invoke({
            "langue": langue,
            "niveau_str": niveau_str
        })
        
        # Essayer de parser le JSON manuellement
        try:
            import json
            import re
            
            # Extraire le JSON du texte
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                exercices_data = json.loads(json_str)
                
                # Valider et corriger les exercices (type par défaut PHRASE pour grammaire)
                exercices_corriges = valider_et_corriger_exercices(exercices_data, "PHRASE")
                return exercices_corriges
            else:
                print("Impossible d'extraire le JSON du résultat")
                return []
                
        except Exception as parse_error:
            print(f"Erreur lors du parsing manuel: {parse_error}")
            return []

@retry_with_backoff(max_retries=3, base_delay=10)
def generer_vocabulaire(langue, niveau_cible="", domaines=None):
    """Génère uniquement la section vocabulaire"""
    if domaines is None:
        domaines = generer_themes_aleatoires(langue, nombre=2, categorie="domaines")
    
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.9,
        api_key=MISTRAL_API_KEY
    )
    
    # Pour simplifier, on crée une classe représentant les exercices de vocabulaire
    class ExercicesVocabulaire(BaseModel):
        exercices: List[Exercice] = Field(..., description="Liste des exercices de vocabulaire")
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un expert en didactique des langues.

Crée 2 exercices de vocabulaire en {langue}{niveau_str}.

IMPORTANT - RÈGLES DE LANGUE:
- Les CONSIGNES doivent être en FRANÇAIS (langue d'interface)
- Les COMPÉTENCES peuvent être en français
- Les contenus des exercices (mots, phrases à compléter) doivent être adaptés à la langue {langue}

IMPORTANT - CHAQUE ÉLÉMENT DOIT AVOIR OBLIGATOIREMENT LE CHAMP "type":
- Pour les questions: "type": "QUESTION"
- Pour les phrases: "type": "PHRASE" 
- Pour les items: "type": "ITEM"

Consignes:
1. Premier exercice: vocabulaire lié à "{domaine1}"
2. Deuxième exercice: synonymes ou antonymes

Voici EXACTEMENT le format à suivre (tu dois créer 2 exercices qui suivent STRICTEMENT cette structure JSON) :
```
[
  {{
    "consigne": "Complétez les phrases avec le mot approprié du vocabulaire de {domaine1}.",
    "niveau_cible": "B1",
    "competence": "Vocabulaire - {domaine1}",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{"id": 1, "texte": "[PHRASE À COMPLÉTER EN {langue}]", "type": "ITEM"}},
        {{"id": 2, "texte": "[PHRASE À COMPLÉTER EN {langue}]", "type": "ITEM"}},
        {{"id": 3, "texte": "[PHRASE À COMPLÉTER EN {langue}]", "type": "ITEM"}}
      ]
    }}
  }},
  {{
    "consigne": "Trouvez un synonyme pour chacun des mots suivants.",
    "niveau_cible": "B1",
    "competence": "Vocabulaire - Synonymes",
    "contenu": {{
      "texte_principal": "",
      "elements": [
        {{"id": 1, "texte": "[MOT EN {langue}] - _____", "type": "ITEM"}},
        {{"id": 2, "texte": "[MOT EN {langue}] - _____", "type": "ITEM"}},
        {{"id": 3, "texte": "[MOT EN {langue}] - _____", "type": "ITEM"}}
      ]
    }}
  }}
]
```

VÉRIFICATION FINALE OBLIGATOIRE:
- Es-tu CERTAIN que les CONSIGNES sont en FRANÇAIS ?
- Es-tu CERTAIN que CHAQUE élément a un champ "type" avec une valeur "QUESTION", "PHRASE" ou "ITEM"?
- Es-tu CERTAIN d'avoir adapté le premier exercice au domaine "{domaine1}"?
- As-tu gardé EXACTEMENT la même structure avec les mêmes champs?

Retourne UNIQUEMENT les exercices au format structuré demandé, rien de plus."""
    )
    
    niveau_str = f" de niveau {niveau_cible}" if niveau_cible else ""
    
    # Extraire les domaines individuellement pour les passer au template
    domaine1 = domaines[0]
    
    try:
        # Utiliser structured_output directement
        structured_llm = llm.with_structured_output(ExercicesVocabulaire)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "langue": langue,
            "niveau_str": niveau_str,
            "domaine1": domaine1
        })
        
        return result.exercices
        
    except Exception as e:
        print(f"Erreur lors de la génération structurée (vocabulaire): {e}")
        print("Tentative de génération avec validation manuelle...")
        
        # Fallback: générer du texte brut et le parser manuellement
        chain_text = prompt | llm | StrOutputParser()
        result_text = chain_text.invoke({
            "langue": langue,
            "niveau_str": niveau_str,
            "domaine1": domaine1
        })
        
        # Essayer de parser le JSON manuellement
        try:
            import json
            import re
            
            # Extraire le JSON du texte
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                exercices_data = json.loads(json_str)
                
                # Valider et corriger les exercices (type par défaut ITEM pour vocabulaire)
                exercices_corriges = valider_et_corriger_exercices(exercices_data, "ITEM")
                return exercices_corriges
            else:
                print("Impossible d'extraire le JSON du résultat")
                return []
                
        except Exception as parse_error:
            print(f"Erreur lors du parsing manuel: {parse_error}")
            return []

def generer_test_initial(langue="français", niveau_cible="", domaines=None):
    """Génère un test initial pour évaluer le niveau de l'apprenant en utilisant une approche modulaire"""
    print(f"Génération d'un test pour la langue: {langue}, niveau: {niveau_cible}")
    
    try:
        # Générer les sections indépendamment avec un délai plus important entre chaque pour éviter les erreurs de rate limit
        print("Génération de la section compréhension écrite...")
        comprehension_ecrite = safe_api_call(generer_comprehension_ecrite, langue, niveau_cible)
        
        # Délai plus important avant de continuer pour éviter les erreurs de rate limit
        print("Attente de 15 secondes avant de continuer...")
        time.sleep(15)
        
        print("Génération de la section grammaire...")
        grammaire = safe_api_call(generer_grammaire, langue, niveau_cible)
        
        # Délai plus important avant de continuer
        print("Attente de 15 secondes avant de continuer...")
        time.sleep(15)
        
        print("Génération de la section vocabulaire...")
        vocabulaire = safe_api_call(generer_vocabulaire, langue, niveau_cible, domaines)
        
        # Assembler le test complet
        test_complet = TestDeCompetence(
            comprehension_ecrite=comprehension_ecrite,
            grammaire=grammaire,
            vocabulaire=vocabulaire
        )
        
        print("Test complet généré avec succès!")
        return test_complet
        
    except Exception as e:
        print(f"Erreur lors de la génération du test: {e}")
        # En cas d'erreur, retourner un test minimal
        print("Génération d'un test de secours minimal...")
        return TestDeCompetence(
            comprehension_ecrite=[],
            grammaire=[],
            vocabulaire=[]
        )

def generer_test_simplifie(langue="français", niveau_cible=""):
    """Version simplifiée qui génère un test avec moins d'appels API pour éviter les rate limits"""
    print(f"Génération d'un test simplifié pour la langue: {langue}, niveau: {niveau_cible}")
    
    # Utiliser des thèmes prédéfinis au lieu de les générer via l'IA
    themes_predefinies = {
        "français": ["la technologie", "l'environnement"],
        "anglais": ["technology", "environment"], 
        "espagnol": ["la tecnología", "el medio ambiente"],
        "breton": ["an teknologiezh", "an endro"]
    }
    
    domaines_predefinis = {
        "français": ["les transports", "la famille"],
        "anglais": ["transportation", "family"],
        "espagnol": ["el transporte", "la familia"], 
        "breton": ["an treuzdougen", "an tiegezh"]
    }
    
    themes = themes_predefinies.get(langue.lower(), ["thème général", "culture"])
    domaines = domaines_predefinis.get(langue.lower(), ["domaine général", "société"])
    
    try:
        print("Génération simplifiée - compréhension écrite...")
        comprehension_ecrite = safe_api_call(generer_comprehension_ecrite, langue, niveau_cible, themes)
        
        print("Attente de 20 secondes...")
        time.sleep(20)
        
        print("Génération simplifiée - grammaire...")
        grammaire = safe_api_call(generer_grammaire, langue, niveau_cible)
        
        print("Attente de 20 secondes...")
        time.sleep(20)
        
        print("Génération simplifiée - vocabulaire...")
        vocabulaire = safe_api_call(generer_vocabulaire, langue, niveau_cible, domaines)
        
        test_complet = TestDeCompetence(
            comprehension_ecrite=comprehension_ecrite,
            grammaire=grammaire,
            vocabulaire=vocabulaire
        )
        
        print("Test simplifié généré avec succès!")
        return test_complet
        
    except Exception as e:
        print(f"Erreur lors de la génération simplifiée: {e}")
        # Retourner un test minimal en cas d'erreur
        return TestDeCompetence(
            comprehension_ecrite=[],
            grammaire=[],
            vocabulaire=[]
        )

def valider_et_corriger_exercices(exercices_data, type_defaut="QUESTION"):
    """Valide et corrige les exercices en ajoutant les champs manquants"""
    if not isinstance(exercices_data, list):
        return []
    
    exercices_corriges = []
    for exercice_data in exercices_data:
        try:
            # Vérifier et corriger la structure de l'exercice
            if not isinstance(exercice_data, dict):
                continue
                
            # S'assurer que tous les champs requis sont présents
            exercice_corrige = {
                "consigne": exercice_data.get("consigne", "Exercice"),
                "niveau_cible": exercice_data.get("niveau_cible", "B1"),
                "competence": exercice_data.get("competence", "Compétence générale"),
                "contenu": exercice_data.get("contenu", {})
            }
            
            # Corriger le contenu
            contenu = exercice_corrige["contenu"]
            if not isinstance(contenu, dict):
                contenu = {}
                
            contenu_corrige = {
                "texte_principal": contenu.get("texte_principal", ""),
                "elements": contenu.get("elements", [])
            }
            
            # Corriger les éléments
            elements_corriges = []
            for i, element in enumerate(contenu_corrige["elements"]):
                if not isinstance(element, dict):
                    continue
                    
                element_corrige = {
                    "id": element.get("id", i + 1),
                    "texte": element.get("texte", f"Élément {i + 1}"),
                    "type": element.get("type", type_defaut)  # Ajouter le type manquant
                }
                elements_corriges.append(element_corrige)
            
            contenu_corrige["elements"] = elements_corriges
            exercice_corrige["contenu"] = contenu_corrige
            
            # Créer l'objet Exercice validé
            exercice_valide = Exercice(**exercice_corrige)
            exercices_corriges.append(exercice_valide)
            
        except Exception as e:
            print(f"Erreur lors de la correction d'un exercice: {e}")
            continue
    
    return exercices_corriges