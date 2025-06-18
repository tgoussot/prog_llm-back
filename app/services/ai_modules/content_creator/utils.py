import os
import time
import random
import re
import json
from functools import wraps
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from app.schemas.language_test import Exercice, Element, TypeElement, OptionQCM

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

def fast_api_call(func, *args, **kwargs):
    """Fonction utilitaire pour faire des appels API rapides avec délais optimisés"""
    try:
        # Délai réduit entre 0.5 et 1.5 secondes avant chaque appel
        delay = random.uniform(0.5, 1.5)
        print(f"Appel API optimisé - délai: {delay:.1f}s")
        time.sleep(delay)
        
        result = func(*args, **kwargs)
        
        # Délai réduit après l'appel réussi
        post_delay = random.uniform(0.3, 0.8)
        print(f"Appel API terminé - délai post-traitement: {post_delay:.1f}s")
        time.sleep(post_delay)
        
        return result
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"Rate limit détecté en mode optimisé: {e}")
            print("Basculement vers le mode sécurisé")
            # En cas de rate limit, basculer vers la méthode sécurisée
            time.sleep(10)  # Attendre 10s puis utiliser safe_api_call
            return safe_api_call(func, *args, **kwargs)
        raise e

def ultra_fast_api_call(func, *args, **kwargs):
    """Fonction utilitaire pour faire des appels API ultra-rapides avec délais minimaux"""
    try:
        # Délai minimal entre 0.2 et 0.8 secondes avant chaque appel
        delay = random.uniform(0.2, 0.8)
        print(f"Appel API haute performance - délai: {delay:.1f}s")
        time.sleep(delay)
        
        result = func(*args, **kwargs)
        
        # Délai minimal après l'appel réussi
        post_delay = random.uniform(0.2, 0.5)
        print(f"Appel API terminé - délai post-traitement: {post_delay:.1f}s")
        time.sleep(post_delay)
        
        return result
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"Rate limit détecté en mode haute performance: {e}")
            print("Basculement vers le mode optimisé")
            # En cas de rate limit, basculer vers fast_api_call
            time.sleep(5)  # Attendre 5s puis utiliser fast_api_call
            return fast_api_call(func, *args, **kwargs)
        raise e

def get_llm(temperature=0.7):
    """Retourne une instance configurée du modèle LLM"""
    return ChatMistralAI(
        model="mistral-large-latest", 
        temperature=temperature,
        api_key=MISTRAL_API_KEY
    )

def valider_et_corriger_exercices(exercices_data, type_defaut="QCM"):
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
                    "type": element.get("type", type_defaut)
                }
                
                # Pour les QCM, convertir les options en objets OptionQCM
                if element_corrige["type"] == "QCM":
                    if "options" in element and isinstance(element["options"], list):
                        options_converties = []
                        for option_data in element["options"]:
                            if isinstance(option_data, dict):
                                option_qcm = OptionQCM(
                                    id=option_data.get("id", "A"),
                                    texte=option_data.get("texte", "Option"),
                                    est_correcte=option_data.get("est_correcte", False)
                                )
                                options_converties.append(option_qcm)
                        element_corrige["options"] = options_converties
                        
                    if "reponse_correcte" in element:
                        element_corrige["reponse_correcte"] = element["reponse_correcte"]
                
                elements_corriges.append(element_corrige)
            
            contenu_corrige["elements"] = elements_corriges
            exercice_corrige["contenu"] = contenu_corrige
            
            # Créer l'objet Exercice validé
            exercice_valide = Exercice(**exercice_corrige)
            
            # Validation finale des éléments QCM (logs supprimés pour nettoyer la sortie)
            
            exercices_corriges.append(exercice_valide)
            
        except Exception as e:
            print(f"Erreur lors de la correction d'un exercice: {e}")
            continue
    
    return exercices_corriges

def parse_json_from_text(text):
    """Extrait et parse le JSON d'un texte"""
    try:
        # Extraire le JSON du texte
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print("Impossible d'extraire le JSON du résultat")
            return None
    except Exception as e:
        print(f"Erreur lors du parsing JSON: {e}")
        return None 