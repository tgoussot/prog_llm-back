from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union, Literal
import os
from dotenv import load_dotenv

# Charger les variables d'environnement mais également définir une clé par défaut si absente
load_dotenv()

# Obtenir la clé API depuis les variables d'environnement ou utiliser la clé par défaut
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "HWeyYbCcjhpjneQZmq5iszsQLil08omZ")

# Définition des modèles d'évaluation
class Erreur(BaseModel):
    type: str = Field(..., description="Type d'erreur (grammaire, vocabulaire, etc.)")
    description: str = Field(..., description="Description précise de l'erreur")
    correction: str = Field(..., description="Correction suggérée")
    explication: str = Field(..., description="Explication pédagogique de la correction")

class ValidationReponse(BaseModel):
    est_correct: bool = Field(..., description="Si la réponse est correcte ou non")
    confiance: float = Field(..., description="Niveau de confiance de l'évaluation (0-1)")
    explication: Optional[str] = Field(None, description="Courte explication si nécessaire")

class AnalyseErreur(BaseModel):
    type_erreur: str = Field(..., description="Type d'erreur (grammaire, vocabulaire, compréhension, etc.)")
    description: str = Field(..., description="Description détaillée de l'erreur")
    correction: str = Field(..., description="Correction suggérée")
    explication: str = Field(..., description="Explication pédagogique de la correction")
    suggestion: str = Field(..., description="Suggestion pour améliorer cette compétence")

class ResultatQuestion(BaseModel):
    id_question: int = Field(..., description="ID de la question")
    texte_question: str = Field(..., description="Texte de la question")
    reponse_utilisateur: str = Field(..., description="Réponse donnée par l'utilisateur")
    est_correct: bool = Field(..., description="Si la réponse est correcte")
    analyse: Optional[AnalyseErreur] = Field(None, description="Analyse détaillée si la réponse est incorrecte")

class Evaluation(BaseModel):
    note: float = Field(..., description="Note sur 10")
    niveau_estime: str = Field(..., description="Niveau CECRL estimé pour cette compétence")
    commentaire_general: str = Field(..., description="Commentaire global sur la performance")
    points_forts: List[str] = Field(..., description="Points forts identifiés")
    points_faibles: List[str] = Field(..., description="Points faibles à améliorer")
    erreurs: List[Erreur] = Field(..., description="Liste détaillée des erreurs")
    suggestions: List[str] = Field(..., description="Suggestions d'amélioration")
    resultats_questions: Optional[List[Any]] = Field(None, description="Résultats détaillés par question")

class BilanCompetences(BaseModel):
    niveau_global: str = Field(..., description="Niveau global estimé (A1-C2)")
    comprehension_ecrite: str = Field(..., description="Niveau en compréhension écrite")
    expression_ecrite: str = Field(..., description="Niveau en expression écrite")
    grammaire: str = Field(..., description="Niveau en grammaire")
    vocabulaire: str = Field(..., description="Niveau en vocabulaire")
    lacunes_identifiees: List[str] = Field(..., description="Principales lacunes identifiées")
    recommandations: List[str] = Field(..., description="Recommandations d'apprentissage")

def valider_reponse(question: Dict, reponse_utilisateur: str, langue: str, texte_original: str = None) -> ValidationReponse:
    """Première IA: vérifie simplement si la réponse est correcte (true/false)"""
    # Configuration du modèle - on utilise un modèle plus léger
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1,
        api_key=MISTRAL_API_KEY
    )
    
    # Extraire les informations de la question
    id_question = question.get("id", 0)
    texte_question = question.get("texte", "")
    type_question = question.get("type", "QUESTION")
    
    # Détecter si c'est un exercice de compréhension écrite
    est_comprehension_ecrite = type_question.upper() == "QUESTION"
    
    # Traiter le cas où l'utilisateur n'a pas répondu
    if not reponse_utilisateur or reponse_utilisateur.strip() == "":
        return ValidationReponse(
            est_correct=False,
            confiance=1.0,
            explication="Aucune réponse fournie."
        )
    
    # Vérifie si l'utilisateur indique que l'information n'est pas dans le texte
    reponse_absence_info = any(phrase in reponse_utilisateur.lower() for phrase in [
        "pas indiqué", "pas mentionné", "n'est pas indiqué", "n'est pas mentionné", 
        "n'est pas précisé", "pas précisé", "on ne sait pas", "je ne sais pas",
        "ne dit pas", "pas dit", "n'est pas dit"
    ])
    
    # Construire le prompt pour la validation
    system_prompt = f"""Tu es un évaluateur objectif pour les exercices de langue {langue}.
    Ta SEULE tâche est de déterminer si la réponse de l'utilisateur est CORRECTE ou INCORRECTE.
    Réponds seulement par TRUE (correct) ou FALSE (incorrect), avec un niveau de confiance.
    Sois très strict sur la précision du contenu, mais ne pénalise pas les fautes d'orthographe mineures.
    """
    
    # Ajouter des instructions spécifiques pour la compréhension écrite
    if est_comprehension_ecrite:
        system_prompt += f"""
        RÈGLES CRUCIALES POUR LA COMPRÉHENSION ÉCRITE:
        - IGNORE TOTALEMENT les erreurs d'orthographe, de grammaire et de style
        - Évalue UNIQUEMENT si l'utilisateur a compris le contenu et l'information demandée
        - Considère la réponse comme correcte (TRUE) même si elle contient des fautes d'orthographe
        - Une réponse est correcte si elle contient l'information principale demandée, quelle que soit sa formulation
        - Les erreurs d'orthographe n'ont AUCUNE importance pour cette évaluation
        
        RÈGLES SPÉCIALES POUR LES RÉPONSES DU TYPE "NON MENTIONNÉ DANS LE TEXTE":
        - Si l'utilisateur répond que l'information n'est pas dans le texte, vérifie ATTENTIVEMENT si:
          1. La question demande des DÉTAILS SPÉCIFIQUES qui ne sont effectivement PAS présents
          2. Ou si la question demande une information générale qui EST présente
        - Exemple: Si le texte dit "il y a des activités pour enfants" sans préciser lesquelles, et 
          que la question demande "quelles activités spécifiques", alors "non précisé dans le texte" est CORRECT
        - IMPORTANT: Ne confonds pas une mention générale avec des détails spécifiques
        """
    
    human_prompt = f"""Question ({type_question}): {texte_question}
    
    Réponse de l'utilisateur: {reponse_utilisateur}
    """
    
    # Ajouter le texte original pour les exercices de compréhension
    if est_comprehension_ecrite and texte_original:
        human_prompt += f"""
        TEXTE ORIGINAL SUR LEQUEL PORTE LA QUESTION:
        {texte_original}
        
        CRUCIAL: Vérifie TOUJOURS si les informations demandées sont bien présentes dans ce texte.
        """
        
        # Ajouter des instructions spécifiques pour le cas où l'utilisateur dit que l'info n'est pas présente
        if reponse_absence_info:
            human_prompt += f"""
            ATTENTION: L'utilisateur a répondu que l'information n'est pas présente dans le texte.
            Vérifie MÉTICULEUSEMENT si:
            1. La question demande une information générale qui EST présente dans le texte -> Réponse INCORRECTE
            2. La question demande des détails spécifiques qui ne sont effectivement PAS explicitement mentionnés -> Réponse CORRECTE
            
            Exemple: Si le texte dit "il y a des activités pour enfants" sans préciser lesquelles,
            et que la question est "Quelles activités spécifiques sont proposées pour les enfants?",
            alors répondre "les activités spécifiques ne sont pas mentionnées" est CORRECT.
            """
    
    human_prompt += """
    Cette réponse est-elle correcte sur le fond (indépendamment de petites fautes d'orthographe ou de grammaire) ?
    Donne ton verdict avec un niveau de confiance entre 0 et 1.
    """
    
    # Création et exécution de la chaîne
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(ValidationReponse)
    
    try:
        return chain.invoke({})
    except Exception as e:
        print(f"Erreur lors de la validation de la réponse: {e}")
        # Valeur par défaut en cas d'erreur
        return ValidationReponse(
            est_correct=False,
            confiance=0.5,
            explication="Impossible de valider la réponse en raison d'une erreur technique."
        )

def analyser_erreur(question: Dict, reponse_utilisateur: str, langue: str, texte_original: str = None) -> AnalyseErreur:
    """Deuxième IA: analyse détaillée uniquement pour les réponses incorrectes"""
    # Configuration du modèle - on utilise un modèle plus puissant pour l'analyse
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.2,
        api_key=MISTRAL_API_KEY
    )
    
    # Extraire les informations de la question
    id_question = question.get("id", 0)
    texte_question = question.get("texte", "")
    type_question = question.get("type", "QUESTION")
    
    # Détecter si c'est un exercice de compréhension écrite
    est_comprehension_ecrite = type_question.upper() == "QUESTION"
    
    # Traiter le cas où l'utilisateur n'a pas répondu
    if not reponse_utilisateur or reponse_utilisateur.strip() == "":
        return AnalyseErreur(
            type_erreur="Absence de réponse",
            description="Aucune réponse fournie pour cette question.",
            correction="N/A",
            explication="Il est important de répondre à toutes les questions.",
            suggestion="Essayez de répondre à chaque question, même si vous n'êtes pas sûr."
        )
    
    # Vérifie si l'utilisateur indique que l'information n'est pas dans le texte
    reponse_absence_info = any(phrase in reponse_utilisateur.lower() for phrase in [
        "pas indiqué", "pas mentionné", "n'est pas indiqué", "n'est pas mentionné", 
        "n'est pas précisé", "pas précisé", "on ne sait pas", "je ne sais pas",
        "ne dit pas", "pas dit", "n'est pas dit"
    ])
    
    # Construire le prompt pour l'analyse détaillée
    system_prompt = f"""Tu es un professeur de {langue} expérimenté spécialisé dans l'analyse précise des erreurs.
    Ta mission est d'analyser en profondeur la réponse incorrecte d'un apprenant pour identifier:
    1. Le type exact d'erreur (compréhension, vocabulaire, grammaire, etc.)
    2. Une description précise de ce qui est incorrect
    3. La correction appropriée
    4. Une explication pédagogique de l'erreur
    5. Une suggestion personnalisée pour améliorer cette compétence spécifique
    """
    
    # Ajouter des instructions spécifiques pour la compréhension écrite
    if est_comprehension_ecrite:
        system_prompt += f"""
        RÈGLES STRICTES POUR LA COMPRÉHENSION ÉCRITE:
        - Tu DOIS IGNORER COMPLÈTEMENT les erreurs d'orthographe, de grammaire et de style
        - Ton évaluation doit porter UNIQUEMENT sur la compréhension du contenu
        - Tu ne dois PAS mentionner les erreurs d'orthographe ou de grammaire dans ton analyse
        - Si la réponse est correcte sur le fond malgré des fautes d'orthographe, considère-la comme TOTALEMENT CORRECTE
        - NE SIGNALE PAS les erreurs orthographiques comme des erreurs - elles sont SANS IMPORTANCE pour la compréhension écrite
        
        RÈGLES SPÉCIALES POUR LES RÉPONSES DU TYPE "NON MENTIONNÉ DANS LE TEXTE":
        - Fais la distinction entre:
          1. Une information générale qui EST présente dans le texte
          2. Des détails spécifiques qui ne sont effectivement PAS explicitement mentionnés
        - Si le texte mentionne une catégorie générale (ex: "activités pour enfants") sans préciser les 
          détails spécifiques, et que la question demande ces détails spécifiques, alors dire que 
          "les détails ne sont pas mentionnés" est CORRECT.
        - Vérifie MÉTICULEUSEMENT le texte avant de juger une réponse incorrecte
        """
    
    human_prompt = f"""Question ({type_question}): {texte_question}
    
    Réponse incorrecte de l'utilisateur: {reponse_utilisateur}
    """
    
    # Ajouter le texte original pour les exercices de compréhension
    if est_comprehension_ecrite and texte_original:
        human_prompt += f"""
        TEXTE ORIGINAL SUR LEQUEL PORTE LA QUESTION:
        {texte_original}
        
        CRUCIAL: Vérifie TOUJOURS si les informations demandées sont bien présentes dans ce texte.
        Ne corrige la réponse que si elle contredit réellement le texte original.
        N'INVENTE PAS d'informations qui ne sont pas dans le texte.
        """
        
        # Ajouter des instructions spécifiques pour le cas où l'utilisateur dit que l'info n'est pas présente
        if reponse_absence_info:
            human_prompt += f"""
            ATTENTION PARTICULIÈRE: L'utilisateur a répondu que l'information demandée n'est pas dans le texte.
            
            Vérifie SCRUPULEUSEMENT si cette affirmation est vraie:
            - Si la question demande des détails spécifiques qui ne sont effectivement PAS mentionnés explicitement
              dans le texte, alors sa réponse "l'information n'est pas dans le texte" pourrait être CORRECTE.
            
            - Si par exemple le texte mentionne "des activités pour enfants" sans préciser lesquelles, et que la
              question demande "quelles activités spécifiques", alors dire "elles ne sont pas mentionnées" 
              devrait être considéré comme CORRECT.
            
            - Sois très prudent avant de juger cette réponse comme incorrecte.
            """

    human_prompt += """
    Analyse l'erreur en détail pour aider l'apprenant à progresser.
    """
    
    # Création et exécution de la chaîne
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(AnalyseErreur)
    
    try:
        return chain.invoke({})
    except Exception as e:
        print(f"Erreur lors de l'analyse de l'erreur: {e}")
        # Valeur par défaut en cas d'erreur
        return AnalyseErreur(
            type_erreur="Indéterminé",
            description="Impossible d'analyser l'erreur en détail en raison d'une erreur technique.",
            correction="N/A",
            explication="Veuillez consulter un enseignant pour une analyse précise.",
            suggestion="Continuez à pratiquer et demandez de l'aide à un enseignant."
        )

def compiler_evaluation(resultats_questions: List[ResultatQuestion], exercice: Dict, langue: str) -> Evaluation:
    """Compile les résultats des questions individuelles en une évaluation globale"""
    # Configuration du modèle
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.3,
        api_key=MISTRAL_API_KEY
    )
    
    # Extraction des informations de l'exercice
    consigne = exercice.get("consigne", "")
    niveau_cible = exercice.get("niveau_cible", "")
    competence = exercice.get("competence", "")
    
    # Déterminer si c'est un exercice de compréhension écrite
    est_comprehension_ecrite = "compréhension" in competence.lower() or any(
        resultat.texte_question and "texte" in resultat.texte_question.lower() 
        for resultat in resultats_questions if hasattr(resultat, "texte_question")
    )
    
    # Vérifier si des réponses indiquent que l'information n'est pas dans le texte
    questions_avec_info_absente = []
    for resultat in resultats_questions:
        if not resultat.est_correct and hasattr(resultat, "reponse_utilisateur") and resultat.reponse_utilisateur:
            reponse = resultat.reponse_utilisateur.lower()
            if any(phrase in reponse for phrase in [
                "pas indiqué", "pas mentionné", "n'est pas indiqué", "n'est pas mentionné", 
                "n'est pas précisé", "pas précisé", "on ne sait pas", "je ne sais pas",
                "ne dit pas", "pas dit", "n'est pas dit"
            ]):
                questions_avec_info_absente.append(resultat.id_question)
    
    # Compiler les résultats des questions en format lisible
    resultats_texte = ""
    for resultat in resultats_questions:
        resultats_texte += f"Question {resultat.id_question}: {resultat.texte_question}\n"
        resultats_texte += f"Réponse: {resultat.reponse_utilisateur}\n"
        resultats_texte += f"Correct: {resultat.est_correct}\n"
        if not resultat.est_correct and resultat.analyse:
            resultats_texte += f"Type d'erreur: {resultat.analyse.type_erreur}\n"
            resultats_texte += f"Description: {resultat.analyse.description}\n"
            resultats_texte += f"Correction: {resultat.analyse.correction}\n"
        resultats_texte += "\n"
    
    # Construire le prompt pour l'évaluation globale
    system_prompt = f"""Tu es un professeur de {langue} expert dans l'évaluation des compétences linguistiques.
    Ta mission est de compiler les résultats d'un exercice et de produire une évaluation globale et formative.
    """
    
    # Ajouter des instructions spécifiques pour la compréhension écrite
    if est_comprehension_ecrite:
        system_prompt += f"""
        INSTRUCTIONS STRICTES POUR LA COMPRÉHENSION ÉCRITE:
        1. Tu DOIS IGNORER COMPLÈTEMENT les erreurs d'orthographe, de grammaire et de style
        2. Si les réponses montrent une bonne compréhension du CONTENU, la note doit être de 10/10
        3. Ne mentionne PAS du tout les erreurs d'orthographe dans ton évaluation
        4. Ne liste AUCUNE erreur orthographique ou grammaticale dans la section "erreurs détectées"
        5. SUPPRIME toute référence aux erreurs orthographiques des points faibles
        6. Si l'information demandée est présente dans la réponse, elle doit être considérée comme PARFAITE
        7. L'évaluation doit porter UNIQUEMENT sur la compréhension du texte, pas sur la forme
        
        RÈGLES SUPPLÉMENTAIRES POUR LES RÉPONSES "NON MENTIONNÉ DANS LE TEXTE":
        - Si l'apprenant a répondu que certaines informations ne sont pas dans le texte, revérifie SCRUPULEUSEMENT.
        - Si la question demande des DÉTAILS SPÉCIFIQUES qui ne sont PAS mentionnés explicitement, 
          alors dire que "ce n'est pas précisé dans le texte" est CORRECT.
        - Par exemple, si le texte mentionne "des activités pour enfants" sans détailler lesquelles,
          et que la question demande "quelles activités spécifiques", dire qu'elles ne sont pas 
          mentionnées est une réponse CORRECTE.
        - Ne pénalise PAS l'apprenant pour avoir correctement identifié qu'une information 
          spécifique n'est pas explicitement mentionnée dans le texte.
        """
    
    human_prompt = f"""Consigne de l'exercice: {consigne}
    Niveau cible: {niveau_cible}
    Compétence évaluée: {competence}
    
    Résultats détaillés des questions:
    {resultats_texte}
    
    Génère une évaluation globale incluant:
    1. Une note sur 10
    2. Un niveau CECRL estimé
    3. Un commentaire général sur la performance
    4. Les points forts identifiés
    5. Les points faibles à améliorer
    6. Une liste détaillée des erreurs (type, description, correction, explication)
    7. Des suggestions concrètes d'amélioration
    """
    
    # Pour la compréhension écrite, ajouter un rappel explicite
    if est_comprehension_ecrite:
        human_prompt += f"""
        RAPPEL CRUCIAL: Pour cet exercice de compréhension écrite, tu DOIS:
        - Ignorer COMPLÈTEMENT les erreurs d'orthographe
        - Attribuer une note de 10/10 si toutes les informations essentielles sont comprises
        - Ne PAS mentionner d'erreurs orthographiques ou grammaticales dans ton évaluation
        - Supprimer toute erreur orthographique de la liste des erreurs détectées
        - Se concentrer UNIQUEMENT sur la compréhension du contenu
        """
        
        # Ajouter des informations sur les questions avec info absente
        if questions_avec_info_absente:
            human_prompt += f"""
            ATTENTION PARTICULIÈRE:
            L'apprenant a indiqué que certaines informations ne sont pas dans le texte pour les questions: {', '.join(map(str, questions_avec_info_absente))}.
            
            Vérifie MINUTIEUSEMENT le texte original pour déterminer si:
            1. Ces informations sont EXPLICITEMENT mentionnées (réponse incorrecte)
            2. Ces informations ne sont PAS EXPLICITEMENT mentionnées (réponse correcte)
            
            IMPORTANTE DISTINCTION:
            - Si le texte mentionne une catégorie générale sans détails spécifiques, et que la question
              demande ces détails spécifiques, alors dire que "les détails ne sont pas mentionnés" doit
              être considéré comme CORRECT.
            """
    
    # Création et exécution de la chaîne
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    chain = prompt | llm.with_structured_output(Evaluation)
    
    try:
        return chain.invoke({})
    except Exception as e:
        print(f"Erreur lors de la compilation de l'évaluation: {e}")
        # Évaluation par défaut en cas d'erreur
        return Evaluation(
            note=5.0,
            niveau_estime="Non déterminé",
            commentaire_general="Une erreur est survenue lors de l'évaluation automatique.",
            points_forts=["Non déterminé en raison d'une erreur technique"],
            points_faibles=["Non déterminé en raison d'une erreur technique"],
            erreurs=[],
            suggestions=["Consulter un enseignant pour une évaluation détaillée"]
        )

def evaluer_reponse(exercice, reponses_utilisateur, langue="français"):
    """Évalue les réponses de l'utilisateur avec l'architecture à deux IA"""
    try:
        # Préparation des données
        contenu_obj = exercice.get("contenu", {})
        
        # Vérifier si le contenu est au nouveau format (avec texte_principal et elements)
        if isinstance(contenu_obj, dict) and "elements" in contenu_obj:
            elements = contenu_obj.get("elements", [])
            texte_principal = contenu_obj.get("texte_principal", "")
        else:
            # Si ancien format, créer une structure factice compatible
            print("Format d'exercice non compatible avec l'évaluation à deux niveaux.")
            # Utiliser l'ancienne méthode d'évaluation
            return evaluer_reponse_legacy(exercice, reponses_utilisateur, langue)
        
        # Analyser les réponses de l'utilisateur
        # Format attendu: "Question 1: réponse1\nQuestion 2: réponse2"
        reponses_dict = {}
        
        # Parser les réponses ligne par ligne
        for ligne in reponses_utilisateur.split('\n'):
            # Chercher les patterns "Question X:" ou "Phrase X:" ou "Item X:"
            for prefix in ["Question", "Phrase", "Item"]:
                if f"{prefix} " in ligne:
                    parts = ligne.split(':', 1)
                    if len(parts) == 2:
                        # Extraire l'ID de la question (ex: "Question 1" -> 1)
                        id_str = parts[0].replace(f"{prefix} ", "").strip()
                        try:
                            id_question = int(id_str)
                            reponse = parts[1].strip()
                            reponses_dict[id_question] = reponse
                        except ValueError:
                            pass
        
        # Évaluer chaque question
        resultats_questions = []
        
        for element in elements:
            id_question = element.get("id", 0)
            texte_question = element.get("texte", "")
            
            # Récupérer la réponse de l'utilisateur pour cette question
            reponse_utilisateur = reponses_dict.get(id_question, "")
            
            if reponse_utilisateur:
                # Première étape: validation simple (correct/incorrect)
                validation = valider_reponse_avec_retry(element, reponse_utilisateur, langue, texte_principal)
                
                if validation.est_correct:
                    # Si correct, pas besoin d'analyse détaillée
                    resultats_questions.append(ResultatQuestion(
                        id_question=id_question,
                        texte_question=texte_question,
                        reponse_utilisateur=reponse_utilisateur,
                        est_correct=True,
                        analyse=None
                    ))
                else:
                    # Si incorrect, faire une analyse détaillée avec le texte original
                    analyse = analyser_erreur_avec_retry(element, reponse_utilisateur, langue, texte_principal)
                    resultats_questions.append(ResultatQuestion(
                        id_question=id_question,
                        texte_question=texte_question,
                        reponse_utilisateur=reponse_utilisateur,
                        est_correct=False,
                        analyse=analyse
                    ))
            else:
                # Pas de réponse pour cette question
                resultats_questions.append(ResultatQuestion(
                    id_question=id_question,
                    texte_question=texte_question,
                    reponse_utilisateur="[Pas de réponse]",
                    est_correct=False,
                    analyse=AnalyseErreur(
                        type_erreur="Absence de réponse",
                        description="Aucune réponse fournie pour cette question.",
                        correction="N/A",
                        explication="Il est important de répondre à toutes les questions.",
                        suggestion="Essayez de répondre à chaque question, même si vous n'êtes pas sûr."
                    )
                ))
        
        # Compiler les résultats en une évaluation globale avec le texte original
        evaluation = compiler_evaluation_avec_retry(resultats_questions, exercice, langue)
        
        # Ajouter les résultats détaillés par question à l'évaluation
        evaluation.resultats_questions = resultats_questions
        
        return evaluation
    
    except Exception as e:
        print(f"Erreur générale lors de l'évaluation: {e}")
        # En cas d'erreur, revenir à l'ancienne méthode
        return evaluer_reponse_legacy(exercice, reponses_utilisateur, langue)

# Fonctions avec système de réessai pour gérer les erreurs 429 (rate limit)
def valider_reponse_avec_retry(question, reponse_utilisateur, langue, texte_original=None, max_retries=3, delay=2):
    """Version avec réessai de la fonction valider_reponse pour gérer les erreurs 429"""
    import time
    
    for attempt in range(max_retries):
        try:
            return valider_reponse(question, reponse_utilisateur, langue, texte_original)
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "rate limit" in error_message:
                print(f"Rate limit atteint, nouvel essai {attempt+1}/{max_retries} dans {delay} secondes...")
                time.sleep(delay)
                # Augmenter le délai exponentiellement à chaque tentative
                delay *= 2
            else:
                # Erreur non liée au rate limit, la remonter
                raise e
    
    # Si on arrive ici, tous les essais ont échoué
    print("Tous les essais de validation ont échoué - retour valeur par défaut")
    return ValidationReponse(
        est_correct=True,  # Par défaut, considérer comme correct en cas d'incertitude
        confiance=0.5,
        explication="Impossible de valider la réponse en raison de limitations techniques."
    )

def analyser_erreur_avec_retry(question, reponse_utilisateur, langue, texte_original=None, max_retries=3, delay=2):
    """Version avec réessai de la fonction analyser_erreur pour gérer les erreurs 429"""
    import time
    
    for attempt in range(max_retries):
        try:
            return analyser_erreur(question, reponse_utilisateur, langue, texte_original)
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "rate limit" in error_message:
                print(f"Rate limit atteint, nouvel essai {attempt+1}/{max_retries} dans {delay} secondes...")
                time.sleep(delay)
                # Augmenter le délai exponentiellement à chaque tentative
                delay *= 2
            else:
                # Erreur non liée au rate limit, la remonter
                raise e
    
    # Si on arrive ici, tous les essais ont échoué
    print("Tous les essais d'analyse ont échoué - retour valeur par défaut")
    return AnalyseErreur(
        type_erreur="Indéterminé",
        description="Impossible d'analyser la réponse en raison de limitations techniques.",
        correction="N/A",
        explication="Veuillez consulter un enseignant pour une analyse détaillée.",
        suggestion="Réessayez ultérieurement lorsque le système sera moins sollicité."
    )

def compiler_evaluation_avec_retry(resultats_questions, exercice, langue, max_retries=3, delay=2):
    """Version avec réessai de la fonction compiler_evaluation pour gérer les erreurs 429"""
    import time
    
    for attempt in range(max_retries):
        try:
            return compiler_evaluation(resultats_questions, exercice, langue)
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "rate limit" in error_message:
                print(f"Rate limit atteint, nouvel essai {attempt+1}/{max_retries} dans {delay} secondes...")
                time.sleep(delay)
                # Augmenter le délai exponentiellement à chaque tentative
                delay *= 2
            else:
                # Erreur non liée au rate limit, la remonter
                raise e
    
    # Si on arrive ici, tous les essais ont échoué
    print("Tous les essais de compilation ont échoué - génération d'évaluation par défaut")
    
    # Déterminer le nombre de réponses correctes pour une note approximative
    nb_questions = len(resultats_questions)
    nb_corrects = sum(1 for r in resultats_questions if r.est_correct)
    note_approx = 10.0 if nb_questions == 0 else (10.0 * nb_corrects / nb_questions)
    
    # Générer une évaluation par défaut
    return Evaluation(
        note=note_approx,
        niveau_estime="Indéterminé",
        commentaire_general=f"Évaluation automatique limitée en raison de contraintes techniques. {nb_corrects}/{nb_questions} réponses semblent correctes.",
        points_forts=["La plateforme n'a pas pu générer une évaluation détaillée de vos points forts."],
        points_faibles=["La plateforme n'a pas pu générer une évaluation détaillée de vos points à améliorer."],
        erreurs=[],
        suggestions=["Réessayez l'évaluation ultérieurement lorsque le système sera moins sollicité."]
    )

def generer_resultats_questions_pour_legacy(exercice, reponse_utilisateur, evaluation):
    """Génère des résultats par question pour la fonction evaluer_reponse_legacy."""
    
    # Extraire les informations de l'exercice
    contenu_obj = exercice.get("contenu", {})
    
    # Vérifier si le contenu est au nouveau format (avec texte_principal et elements)
    if isinstance(contenu_obj, dict) and "elements" in contenu_obj:
        elements = contenu_obj.get("elements", [])
    else:
        # Si ancien format, pas d'éléments disponibles
        return None
    
    # Analyser les réponses de l'utilisateur
    # Format attendu: "Question 1: réponse1\nQuestion 2: réponse2"
    reponses_dict = {}
    
    # Parser les réponses ligne par ligne
    for ligne in reponse_utilisateur.split('\n'):
        # Chercher les patterns "Question X:" ou "Phrase X:" ou "Item X:"
        for prefix in ["Question", "Phrase", "Item"]:
            if f"{prefix} " in ligne:
                parts = ligne.split(':', 1)
                if len(parts) == 2:
                    # Extraire l'ID de la question (ex: "Question 1" -> 1)
                    id_str = parts[0].replace(f"{prefix} ", "").strip()
                    try:
                        id_question = int(id_str)
                        reponse = parts[1].strip()
                        reponses_dict[id_question] = reponse
                    except ValueError:
                        pass
    
    # Les erreurs dans l'évaluation
    erreurs_dict = {}
    for erreur in evaluation.erreurs:
        if hasattr(erreur, "description") and "question" in erreur.description.lower():
            # Essayer de trouver l'ID de la question dans la description
            for i in range(1, 10):  # Supposons qu'il n'y a pas plus de 9 questions
                if f"question {i}" in erreur.description.lower():
                    erreurs_dict[i] = erreur
                    break
    
    # Générer les résultats par question
    resultats = []
    
    for element in elements:
        if isinstance(element, dict):
            id_question = element.get("id", 0)
            texte_question = element.get("texte", "")
            type_element = element.get("type", "")
        else:
            id_question = getattr(element, "id", 0)
            texte_question = getattr(element, "texte", "")
            type_element = getattr(element, "type", "")
        
        # Ignorer les éléments qui ne sont pas des questions
        if type_element.upper() != "QUESTION" and type_element.upper() != "PHRASE" and type_element.upper() != "ITEM":
            continue
        
        # Récupérer la réponse de l'utilisateur pour cette question
        reponse_utilisateur_question = reponses_dict.get(id_question, "[Pas de réponse]")
        
        # Déterminer si la réponse est correcte en se basant sur les erreurs détectées
        est_correct = id_question not in erreurs_dict
        
        # Créer l'analyse de l'erreur si nécessaire
        analyse = None
        if not est_correct and id_question in erreurs_dict:
            erreur = erreurs_dict[id_question]
            analyse = AnalyseErreur(
                type_erreur=erreur.type,
                description=erreur.description,
                correction=erreur.correction,
                explication=erreur.explication,
                suggestion="Revoyez cette partie du cours."
            )
        
        # Ajouter le résultat pour cette question
        resultats.append(ResultatQuestion(
            id_question=id_question,
            texte_question=texte_question,
            reponse_utilisateur=reponse_utilisateur_question,
            est_correct=est_correct,
            analyse=analyse
        ))
    
    return resultats

# Garder l'ancienne fonction pour la compatibilité
def evaluer_reponse_legacy(exercice, reponse_utilisateur, langue="français"):
    """Ancienne méthode d'évaluation pour la compatibilité"""
    import time
    
    # Nombre maximum d'essais et délai initial entre essais
    max_retries = 3
    delay = 2
    
    for attempt in range(max_retries):
        try:
            # Configuration du modèle
            llm = ChatMistralAI(
                model="mistral-large-latest", 
                temperature=0.1,
                api_key=MISTRAL_API_KEY
            )
            
            # Extraction des informations de l'exercice
            consigne = exercice.get("consigne", "")
            contenu_obj = exercice.get("contenu", {})
            niveau_cible = exercice.get("niveau_cible", "")
            competence = exercice.get("competence", "")
            
            # Vérifier si le contenu est au nouveau format (avec texte_principal et elements)
            if isinstance(contenu_obj, dict) and "texte_principal" in contenu_obj:
                texte_principal = contenu_obj.get("texte_principal", "")
                elements = contenu_obj.get("elements", [])
                
                # Formater le contenu pour l'évaluation
                contenu_formate = f"Texte principal:\n{texte_principal}\n\n"
                if elements:
                    contenu_formate += "Éléments:\n"
                    for elem in elements:
                        elem_id = elem.get("id", 0)
                        elem_texte = elem.get("texte", "")
                        elem_type = elem.get("type", "")
                        contenu_formate += f"{elem_id}. [{elem_type}] {elem_texte}\n"
                
                contenu = contenu_formate
            else:
                # Ancien format: contenu est une simple chaîne
                contenu = contenu_obj if isinstance(contenu_obj, str) else str(contenu_obj)
            
            # Conserver le texte original et les questions pour la vérification de cohérence
            texte_original = texte_principal if isinstance(contenu_obj, dict) and "texte_principal" in contenu_obj else ""
            questions_originales = [(e.get("id", 0), e.get("texte", "")) for e in elements] if isinstance(contenu_obj, dict) and "elements" in contenu_obj else []
            
            # Déterminer le type d'exercice
            est_comprehension_ecrite = False
            if "compréhension" in competence.lower() or "compréhension" in consigne.lower():
                est_comprehension_ecrite = True
            
            # Construction du prompt d'évaluation selon la langue
            if langue == "espagnol":
                if est_comprehension_ecrite:
                    # Instructions spécifiques pour les exercices de compréhension écrite
                    system_message = f"""Tu es un professeur d'espagnol expert dans l'évaluation précise des compétences de compréhension écrite.
                    Ton rôle est d'analyser si l'apprenant a correctement compris le texte et s'il a saisi les informations demandées.
                    
                    INSTRUCTIONS CRITIQUES POUR LA COMPRÉHENSION ÉCRITE:
                    - Tu DOIS IGNORER TOTALEMENT les erreurs d'orthographe, de grammaire ou de style
                    - L'objectif UNIQUE est d'évaluer la compréhension du texte, PAS la qualité de l'expression écrite
                    
                    RÈGLES D'ÉVALUATION STRICTES ET NON NÉGOCIABLES:
                    - Si TOUTES les réponses contiennent les informations principales correctes → Note de 10/10 OBLIGATOIRE
                    - Les fautes d'orthographe ne doivent PAS apparaître dans l'évaluation
                    - NE MENTIONNE JAMAIS les erreurs orthographiques dans ton évaluation
                    - NE PÉNALISE JAMAIS une réponse pour des erreurs orthographiques ou grammaticales
                    - OBLIGATION DE CONCENTRER ton évaluation UNIQUEMENT sur la compréhension du contenu
                    - Si l'information clé est présente dans la réponse, même mal orthographiée, elle est CORRECTE
                    - INTERDICTION ABSOLUE de baisser la note pour des raisons orthographiques
                    - N'INCLUS PAS d'erreurs orthographiques dans la liste des erreurs
                    
                    RÈGLES CRUCIALES POUR LES RÉPONSES "NON MENTIONNÉ DANS LE TEXTE":
                    - Si l'apprenant répond que certaines informations ne sont pas dans le texte, vérifie RIGOUREUSEMENT
                    - Si la question demande des DÉTAILS SPÉCIFIQUES qui ne sont effectivement PAS explicitement mentionnés 
                      dans le texte, alors la réponse "ce n'est pas précisé dans le texte" est CORRECTE
                    - Par exemple, si le texte dit "il y a des activités pour enfants" sans préciser lesquelles, 
                      et que la question demande "quelles activités spécifiques", alors dire qu'elles 
                      ne sont pas mentionnées est une réponse CORRECTE
                    - Ne pénalise JAMAIS l'apprenant pour avoir correctement identifié qu'une information 
                      spécifique n'est pas explicitement mentionnée dans le texte
                    """
                    
                    human_message = f"""Voici un exercice d'espagnol de niveau {niveau_cible} qui évalue la compétence: {competence}
                    
                    Consigne de l'exercice: {consigne}
                    
                    Contenu de l'exercice:
                    {contenu}
                    
                    Réponse de l'apprenant:
                    {reponse_utilisateur}
                    
                    Applique les règles d'évaluation STRICTES pour noter cette réponse:
                    - Note de 10/10 si toutes les informations essentielles sont correctement identifiées
                    - IGNORE TOTALEMENT l'orthographe, la grammaire et le style
                    - Évalue UNIQUEMENT la compréhension du contenu
                    - SUPPRIME toute mention d'erreurs orthographiques ou grammaticales
                    - NE BAISSE JAMAIS la note pour des fautes d'orthographe ou de grammaire
                    """
                else:
                    # Instructions standard pour les autres types d'exercices
                    system_message = f"""Tu es un professeur d'espagnol expert dans l'évaluation précise des compétences linguistiques.
                    Ton rôle est d'analyser en détail la réponse d'un apprenant et de fournir une évaluation complète et formative.
                    Tu évalues des exercices en espagnol, mais tes commentaires et ton analyse doivent être en français."""
                    
                    human_message = f"""Voici un exercice d'espagnol de niveau {niveau_cible} qui évalue la compétence: {competence}
                    
                    Consigne de l'exercice: {consigne}
                    
                    Contenu de l'exercice:
                    {contenu}
                    
                    Réponse de l'apprenant:
                    {reponse_utilisateur}
                    
                    Évalue cette réponse de façon détaillée en identifiant:
                    1. Les erreurs précises en espagnol (avec type, description, correction et explication en français)
                    2. Les points forts et points faibles (en français)
                    3. Une estimation du niveau CECRL pour cette compétence spécifique
                    4. Des suggestions concrètes d'amélioration (en français)"""
            else:
                if est_comprehension_ecrite:
                    # Instructions spécifiques pour les exercices de compréhension écrite en français
                    system_message = f"""Tu es un professeur de français expert dans l'évaluation précise des compétences de compréhension écrite.
                    Ton rôle est d'analyser si l'apprenant a correctement compris le texte et s'il a saisi les informations demandées.
                    
                    INSTRUCTIONS CRITIQUES POUR LA COMPRÉHENSION ÉCRITE:
                    - Tu DOIS IGNORER TOTALEMENT les erreurs d'orthographe, de grammaire ou de style
                    - L'objectif UNIQUE est d'évaluer la compréhension du texte, PAS la qualité de l'expression écrite
                    
                    RÈGLES D'ÉVALUATION STRICTES ET NON NÉGOCIABLES:
                    - Si TOUTES les réponses contiennent les informations principales correctes → Note de 10/10 OBLIGATOIRE
                    - Les fautes d'orthographe ne doivent PAS apparaître dans l'évaluation
                    - NE MENTIONNE JAMAIS les erreurs orthographiques dans ton évaluation
                    - NE PÉNALISE JAMAIS une réponse pour des erreurs orthographiques ou grammaticales
                    - OBLIGATION DE CONCENTRER ton évaluation UNIQUEMENT sur la compréhension du contenu
                    - Si l'information clé est présente dans la réponse, même mal orthographiée, elle est CORRECTE
                    - INTERDICTION ABSOLUE de baisser la note pour des raisons orthographiques
                    - N'INCLUS PAS d'erreurs orthographiques dans la liste des erreurs
                    
                    RÈGLES CRUCIALES POUR LES RÉPONSES "NON MENTIONNÉ DANS LE TEXTE":
                    - Si l'apprenant répond que certaines informations ne sont pas dans le texte, vérifie RIGOUREUSEMENT
                    - Si la question demande des DÉTAILS SPÉCIFIQUES qui ne sont effectivement PAS explicitement mentionnés 
                      dans le texte, alors la réponse "ce n'est pas précisé dans le texte" est CORRECTE
                    - Par exemple, si le texte dit "il y a des activités pour enfants" sans préciser lesquelles, 
                      et que la question demande "quelles activités spécifiques", alors dire qu'elles 
                      ne sont pas mentionnées est une réponse CORRECTE
                    - Ne pénalise JAMAIS l'apprenant pour avoir correctement identifié qu'une information 
                      spécifique n'est pas explicitement mentionnée dans le texte
                    """
                    
                    human_message = f"""Voici un exercice de niveau {niveau_cible} qui évalue la compétence: {competence}
                    
                    Consigne de l'exercice: {consigne}
                    
                    Contenu de l'exercice:
                    {contenu}
                    
                    Réponse de l'apprenant:
                    {reponse_utilisateur}
                    
                    Applique les règles d'évaluation STRICTES pour noter cette réponse:
                    - Note de 10/10 si toutes les informations essentielles sont correctement identifiées
                    - IGNORE TOTALEMENT l'orthographe, la grammaire et le style
                    - Évalue UNIQUEMENT la compréhension du contenu
                    - SUPPRIME toute mention d'erreurs orthographiques ou grammaticales
                    - NE BAISSE JAMAIS la note pour des fautes d'orthographe ou de grammaire
                    """
                else:
                    # Instructions standard pour les autres types d'exercices
                    system_message = f"""Tu es un professeur de {langue} expert dans l'évaluation précise des compétences linguistiques.
                    Ton rôle est d'analyser en détail la réponse d'un apprenant et de fournir une évaluation complète et formative."""
                    
                    human_message = f"""Voici un exercice de niveau {niveau_cible} qui évalue la compétence: {competence}
                    
                    Consigne de l'exercice: {consigne}
                    
                    Contenu de l'exercice:
                    {contenu}
                    
                    Réponse de l'apprenant:
                    {reponse_utilisateur}
                    
                    Évalue cette réponse de façon détaillée en identifiant:
                    1. Les erreurs précises (avec type, description, correction et explication)
                    2. Les points forts et points faibles
                    3. Une estimation du niveau CECRL pour cette compétence spécifique
                    4. Des suggestions concrètes d'amélioration"""
            
            # Instruction pour assurer la cohérence avec le texte original
            if est_comprehension_ecrite and texte_original:
                verification_coherence = f"""
                TRÈS IMPORTANT - VÉRIFICATION DE COHÉRENCE:
                Avant de finaliser ton évaluation, vérifie que tes corrections correspondent au texte original.
                Texte original: "{texte_original[:200]}..."
                
                Questions du texte:
                {chr(10).join([f"{id}. {q}" for id, q in questions_originales])}
                
                Assure-toi que tes corrections ne contredisent PAS le texte original.
                Si tu remarques une incohérence, corrige ton évaluation pour qu'elle soit parfaitement alignée avec le texte original.
                """
                
                # Ajouter la vérification de cohérence au message humain
                human_message += verification_coherence
            
            # Construction du prompt final
            prompt_content = [
                ("system", system_message),
                ("human", human_message)
            ]
            
            # Création et exécution de la chaîne
            prompt = ChatPromptTemplate.from_messages(prompt_content)
            chain = prompt | llm.with_structured_output(Evaluation)
            evaluation = chain.invoke({})
            
            # Générer les résultats par question
            resultats_questions = generer_resultats_questions_pour_legacy(exercice, reponse_utilisateur, evaluation)
            if resultats_questions:
                evaluation.resultats_questions = resultats_questions
            
            return evaluation
            
        except Exception as e:
            error_message = str(e).lower()
            if "429" in error_message or "rate limit" in error_message:
                print(f"Rate limit atteint, nouvel essai {attempt+1}/{max_retries} dans {delay} secondes...")
                time.sleep(delay)
                # Augmenter le délai exponentiellement à chaque tentative
                delay *= 2
            elif attempt < max_retries - 1:
                print(f"Erreur lors de l'évaluation: {e}, tentative {attempt+1}/{max_retries}")
                time.sleep(delay)
            else:
                # Dernière tentative échouée, créer une évaluation par défaut
                print(f"Erreur lors de l'évaluation après {max_retries} tentatives: {e}")
                break
    
    # Si toutes les tentatives ont échoué, retourner une évaluation par défaut
    return Evaluation(
        note=7.5,  # Note moyenne par défaut pour ne pas pénaliser l'apprenant
        niveau_estime="Indéterminé",
        commentaire_general="Une erreur est survenue lors de l'évaluation automatique. Veuillez consulter un enseignant pour une évaluation manuelle.",
        points_forts=["Les réponses semblent globalement correctes, mais le système ne peut pas fournir une analyse détaillée."],
        points_faibles=["Le système d'évaluation automatique a rencontré une limitation technique."],
        erreurs=[],
        suggestions=["Consultez un enseignant pour une évaluation détaillée", 
                    "Réessayer ultérieurement lorsque le système sera moins sollicité."]
    )

# Fonction de bilan global des compétences
def generer_bilan_competences(resultats_test, langue="français"):
    """Génère un bilan global des compétences à partir des résultats du test complet"""
    
    # Configuration du modèle
    llm = ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0.1,
        api_key=MISTRAL_API_KEY
    )
    
    # Préparation des résultats pour le prompt
    # Extraction des données importantes pour éviter la sérialisation d'objets complexes
    resultats_simplifies = {}
    
    try:
        # Vérifier si nous avons le nouveau format de résultats (dictionnaire par catégorie)
        if isinstance(resultats_test, dict) and any(cat in resultats_test for cat in ["comprehension_ecrite", "expression_ecrite", "grammaire", "vocabulaire"]):
            for categorie, exercices in resultats_test.items():
                if isinstance(exercices, list):
                    resultats_simplifies[categorie] = []
                    for exercice_data in exercices:
                        # Extraire les informations essentielles
                        eval_data = exercice_data.get("evaluation", {})
                        if isinstance(eval_data, dict):
                            resultats_simplifies[categorie].append({
                                "competence": categorie,
                                "note": eval_data.get("note", 5.0),
                                "niveau": eval_data.get("niveau_estime", "B1"),
                                "points_forts": eval_data.get("points_forts", []),
                                "points_faibles": eval_data.get("points_faibles", [])
                            })
        # Format de données plus ancien ou différent
        else:
            # Créer un minimum de données pour permettre l'analyse
            resultats_simplifies = {
                "donnees_generales": {
                    "langue": langue,
                    "resultats_bruts": str(resultats_test)[:1000]  # Limiter la longueur
                }
            }
        
        # Formater les données pour le prompt
        resultats_format = str(resultats_simplifies)
    except Exception as e:
        print(f"Erreur lors de la préparation des données pour le bilan: {e}")
        # Créer un minimum de données en cas d'erreur
        resultats_format = f"""{{
            "erreur": "Les données n'ont pas pu être correctement analysées",
            "langue": "{langue}"
        }}"""
    
    # Construction du prompt pour le bilan selon la langue
    if langue == "espagnol":
        prompt_content = [
            ("system", f"""Tu es un expert en didactique des langues spécialisé dans l'évaluation 
            des compétences linguistiques en espagnol selon le Cadre Européen Commun de Référence pour les Langues (CECRL).
            Ta mission est d'analyser les résultats d'un test complet d'espagnol et de produire un bilan détaillé en français."""),
            ("human", f"""Voici les résultats simplifiés d'un test de compétence en espagnol:
            
            {resultats_format}
            
            Analyse ces résultats et génère un bilan complet des compétences en français incluant:
            1. Une estimation du niveau global selon le CECRL (A1-C2)
            2. Une estimation du niveau pour chaque compétence (compréhension écrite, expression écrite, grammaire, vocabulaire)
            3. Une identification précise des principales lacunes en espagnol
            4. Des recommandations personnalisées pour progresser""")
        ]
    else:
        prompt_content = [
            ("system", f"""Tu es un expert en didactique des langues spécialisé dans l'évaluation 
            des compétences linguistiques selon le Cadre Européen Commun de Référence pour les Langues (CECRL).
            Ta mission est d'analyser les résultats d'un test complet et de produire un bilan détaillé."""),
            ("human", f"""Voici les résultats simplifiés d'un test de compétence en {langue}:
            
            {resultats_format}
            
            Analyse ces résultats et génère un bilan complet des compétences incluant:
            1. Une estimation du niveau global selon le CECRL (A1-C2)
            2. Une estimation du niveau pour chaque compétence (compréhension écrite, expression écrite, grammaire, vocabulaire)
            3. Une identification précise des principales lacunes
            4. Des recommandations personnalisées pour progresser""")
        ]
    
    # Création et exécution de la chaîne
    try:
        prompt = ChatPromptTemplate.from_messages(prompt_content)
        chain = prompt | llm.with_structured_output(BilanCompetences)
        return chain.invoke({})
    except Exception as e:
        # En cas d'erreur, créer un bilan par défaut avec des informations sur l'erreur
        print(f"Erreur lors de la génération du bilan: {e}")
        import traceback
        trace = traceback.format_exc()
        print(trace)
        
        return BilanCompetences(
            niveau_global="B1",
            comprehension_ecrite="B1",
            expression_ecrite="B1",
            grammaire="B1",
            vocabulaire="B1",
            lacunes_identifiees=[
                f"Impossible de déterminer les lacunes en raison d'une erreur technique: {str(e)}",
                "Veuillez consulter un enseignant pour une analyse détaillée"
            ],
            recommandations=[
                "Consultez un enseignant pour une évaluation personnalisée",
                "Continuez à pratiquer régulièrement dans toutes les compétences linguistiques"
            ]
        )

# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple d'exercice
    exercice_exemple = {
        "consigne": "Conjuguez les verbes entre parenthèses au passé composé.",
        "contenu": "1. Je (aller) au cinéma hier soir.\n2. Nous (manger) au restaurant.\n3. Ils (prendre) le train.",
        "niveau_cible": "A2",
        "competence": "Conjugaison des verbes au passé composé"
    }
    
    # Exemple de réponse
    reponse_exemple = "1. Je suis allé au cinéma hier soir.\n2. Nous avons mangé au restaurant.\n3. Ils ont pris le train."
    
    # Évaluation de la réponse
    evaluation = evaluer_reponse(exercice_exemple, reponse_exemple)
    print("Évaluation de la réponse:")
    print(f"Note: {evaluation.note}/10")
    print(f"Niveau estimé: {evaluation.niveau_estime}")
    print(f"Points forts: {', '.join(evaluation.points_forts)}")
    print(f"Points faibles: {', '.join(evaluation.points_faibles)}")
    
    # Affichage des erreurs détaillées
    print("\nErreurs détaillées:")
    for i, erreur in enumerate(evaluation.erreurs, 1):
        print(f"\nErreur {i}:")
        print(f"Type: {erreur.type}")
        print(f"Description: {erreur.description}")
        print(f"Correction: {erreur.correction}")
        print(f"Explication: {erreur.explication}") 