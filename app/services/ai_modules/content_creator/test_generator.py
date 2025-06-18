import time
import random
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.schemas.language_test import TestComplet
from .utils import safe_api_call, fast_api_call
from .exercise_generators import generer_comprehension_ecrite, generer_grammaire, generer_vocabulaire
from .theme_generator import generer_themes_aleatoires

def ultra_fast_api_call(func, *args, **kwargs):
    """Fonction utilitaire pour faire des appels API ultra-rapides"""
    try:
        # Délai ultra-réduit entre 0.2 et 0.8 secondes avant chaque appel
        delay = random.uniform(0.2, 0.8)
        print(f"Appel API haute performance - délai: {delay:.1f}s")
        time.sleep(delay)
        
        result = func(*args, **kwargs)
        
        # Délai ultra-réduit après l'appel réussi
        post_delay = random.uniform(0.1, 0.5)
        print(f"Appel API terminé - délai post-traitement: {post_delay:.1f}s")
        time.sleep(post_delay)
        
        return result
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            print(f"Rate limit détecté en mode haute performance: {e}")
            print("Basculement vers le mode optimisé")
            # En cas de rate limit, basculer vers fast_api_call
            time.sleep(5)
            return fast_api_call(func, *args, **kwargs)
        raise e

def generer_test_parallele(langue="français", niveau_cible="", domaines=None):
    """Génère un test avec génération en parallèle pour une vitesse maximale"""
    print(f"Génération parallèle d'un test - langue: {langue}, niveau: {niveau_cible}")
    
    try:
        # Étape 1: Générer compréhension écrite d'abord (car elle peut influencer les thèmes)
        print("Génération section compréhension écrite...")
        comprehension_ecrite = ultra_fast_api_call(generer_comprehension_ecrite, langue, niveau_cible)
        
        # Délai minimal avant de lancer les tâches parallèles
        delay = random.uniform(1, 2)
        print(f"Préparation génération parallèle - délai: {delay:.1f}s")
        time.sleep(delay)
        
        # Étape 2: Générer grammaire et vocabulaire EN PARALLÈLE
        print("Lancement génération parallèle: grammaire + vocabulaire")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Soumettre les deux tâches en parallèle
            future_grammaire = executor.submit(ultra_fast_api_call, generer_grammaire, langue, niveau_cible)
            future_vocabulaire = executor.submit(ultra_fast_api_call, generer_vocabulaire, langue, niveau_cible, domaines)
            
            # Attendre que les deux tâches se terminent
            grammaire = None
            vocabulaire = None
            
            for future in as_completed([future_grammaire, future_vocabulaire]):
                if future == future_grammaire:
                    grammaire = future.result()
                    print("Section grammaire générée")
                elif future == future_vocabulaire:
                    vocabulaire = future.result()
                    print("Section vocabulaire générée")
        
        # Assembler le test complet
        test_complet = TestComplet(
            comprehension_ecrite=comprehension_ecrite,
            grammaire=grammaire,
            vocabulaire=vocabulaire
        )
        
        print("Test généré avec succès en mode parallèle")
        return test_complet
        
    except Exception as e:
        print(f"Erreur lors de la génération parallèle: {e}")
        print("Basculement vers génération optimisée")
        # En cas d'erreur, basculer vers la méthode optimisée
        return generer_test_optimise(langue, niveau_cible, domaines)

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
        
        # Assembler le test complet - maintenant directement compatible avec l'API
        test_complet = TestComplet(
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
        return TestComplet(
            comprehension_ecrite=[],
            grammaire=[],
            vocabulaire=[]
        )

def generer_test_simplifie(langue="français", niveau_cible=""):
    """Version simplifiée qui génère un test avec moins d'appels API pour éviter les rate limits"""
    print(f"Génération d'un test simplifié pour la langue: {langue}, niveau: {niveau_cible}")
    
    # Utiliser des thèmes générés au lieu de thèmes prédéfinis
    themes = generer_themes_aleatoires(langue, nombre=3, categorie="compréhension")
    domaines = generer_themes_aleatoires(langue, nombre=3, categorie="domaines")
    
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
        
        test_complet = TestComplet(
            comprehension_ecrite=comprehension_ecrite,
            grammaire=grammaire,
            vocabulaire=vocabulaire
        )
        
        print("Test simplifié généré avec succès!")
        return test_complet
        
    except Exception as e:
        print(f"Erreur lors de la génération simplifiée: {e}")
        # Retourner un test minimal en cas d'erreur
        return TestComplet(
            comprehension_ecrite=[],
            grammaire=[],
            vocabulaire=[]
        )

def generer_test_optimise(langue="français", niveau_cible="", domaines=None):
    """Génère un test avec des délais optimisés pour une génération plus rapide"""
    print(f"Génération optimisée d'un test - langue: {langue}, niveau: {niveau_cible}")
    
    try:
        # Générer les sections avec des délais réduits
        print("Génération section compréhension écrite...")
        comprehension_ecrite = fast_api_call(generer_comprehension_ecrite, langue, niveau_cible)
        
        # Délai réduit entre les sections : 3-5 secondes au lieu de 15
        delay = random.uniform(3, 5)
        print(f"Délai entre sections: {delay:.1f}s")
        time.sleep(delay)
        
        print("Génération section grammaire...")
        grammaire = fast_api_call(generer_grammaire, langue, niveau_cible)
        
        # Délai réduit entre les sections
        delay = random.uniform(3, 5)
        print(f"Délai entre sections: {delay:.1f}s")
        time.sleep(delay)
        
        print("Génération section vocabulaire...")
        vocabulaire = fast_api_call(generer_vocabulaire, langue, niveau_cible, domaines)
        
        # Assembler le test complet
        test_complet = TestComplet(
            comprehension_ecrite=comprehension_ecrite,
            grammaire=grammaire,
            vocabulaire=vocabulaire
        )
        
        print("Test généré avec succès en mode optimisé")
        return test_complet
        
    except Exception as e:
        print(f"Erreur lors de la génération optimisée: {e}")
        print("Basculement vers génération standard")
        # En cas d'erreur, basculer vers la méthode standard
        return generer_test_initial(langue, niveau_cible, domaines) 