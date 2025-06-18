"""
Module de création de contenu pédagogique - Interface de compatibilité

Ce fichier maintient la compatibilité avec l'ancienne interface tout en 
utilisant la nouvelle architecture modulaire du package content_creator.
"""

# Import de toutes les fonctionnalités du nouveau package modulaire
from .content_creator import *

# Maintenir la compatibilité avec l'ancienne interface
# Toutes les fonctions et classes sont maintenant importées du package content_creator
# qui est organisé de manière plus modulaire et maintenable.

# Les fonctions principales disponibles sont :
# - generer_test_initial()
# - generer_test_simplifie() 
# - generer_comprehension_ecrite()
# - generer_grammaire()
# - generer_vocabulaire()
# - generer_themes_aleatoires()
# - traduire_termes_techniques()
# - traduire_prompt()
# - valider_et_corriger_exercices()

# Les modèles de données disponibles sont :
# - TypeElement, Element, Contenu, Exercice
# - TestDeCompetence, ExercicePersonnalise
# - TermesTraduction

# Les utilitaires disponibles sont :
# - retry_with_backoff, safe_api_call, get_llm