"""
Module de création de contenu pédagogique pour l'apprentissage des langues.

Ce module fournit des outils pour générer automatiquement des exercices
et des tests de compétences linguistiques en utilisant l'intelligence artificielle.
"""

# Import des modèles principaux depuis les schémas unifiés
from app.schemas.language_test import (
    TypeElement,
    Element,
    Contenu,
    Exercice,
    TestComplet,
    OptionQCM
)

# Import des fonctions principales
from .test_generator import generer_test_initial, generer_test_simplifie, generer_test_optimise, generer_test_parallele
from .exercise_generators import generer_comprehension_ecrite, generer_grammaire, generer_vocabulaire
from .theme_generator import generer_themes_aleatoires
from .translation import traduire_termes_techniques, traduire_prompt
from .utils import retry_with_backoff, safe_api_call, get_llm, valider_et_corriger_exercices

# Définir les exports publics
__all__ = [
    # Modèles
    'TypeElement',
    'Element', 
    'Contenu',
    'Exercice',
    'TestComplet',
    'OptionQCM',
    
    # Fonctions principales
    'generer_test_initial',
    'generer_test_simplifie',
    'generer_test_optimise',
    'generer_test_parallele',
    'generer_comprehension_ecrite',
    'generer_grammaire', 
    'generer_vocabulaire',
    'generer_themes_aleatoires',
    'traduire_termes_techniques',
    'traduire_prompt',
    
    # Utilitaires
    'retry_with_backoff',
    'safe_api_call',
    'get_llm',
    'valider_et_corriger_exercices'
] 