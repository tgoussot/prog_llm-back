"""
Routes API pour les tests de langue
"""
from fastapi import APIRouter, HTTPException, status
from app.schemas.language_test import LanguageTestRequest, LanguageTestResponse
from app.services.language_test_service import generate_language_test
from app.schemas.message import MessageResponse
import sys

router = APIRouter()

# Vérification de l'import du module
CONTENT_CREATOR_IMPORTED = False
try:
    from app.services.ai_modules.content_creator_ai import generer_test_initial
    CONTENT_CREATOR_IMPORTED = True
except ImportError as e:
    import_error = str(e)

@router.post("/", response_model=LanguageTestResponse, status_code=status.HTTP_201_CREATED)
async def create_language_test(request: LanguageTestRequest):
    """
    Génère un nouveau test de langue initial
    
    - **langue**: Langue du test (français, anglais, espagnol, etc.)
    - **niveau_cible**: Niveau CECRL ciblé (A1-C2, optionnel)
    
    Retourne un test complet avec exercices de compréhension écrite, expression écrite,
    grammaire et vocabulaire.
    """
    try:
        return await generate_language_test(request)
    except Exception as e:
        # En production, utilisez un logger pour enregistrer l'erreur
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la génération du test: {str(e)}"
        )

@router.get("/import-check", response_model=MessageResponse)
async def check_imports():
    """
    Vérifie si l'import du module content_creator_ai fonctionne correctement
    """
    if CONTENT_CREATOR_IMPORTED:
        return MessageResponse(message="Le module content_creator_ai est correctement importé")
    else:
        return MessageResponse(message=f"Erreur d'import: {import_error}. Chemins Python: {sys.path}") 