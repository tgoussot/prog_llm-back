from fastapi import APIRouter, HTTPException
from app.schemas.message import MessageResponse
from app.api.endpoints.language_tests import router as language_tests_router

router = APIRouter()

# Inclure le routeur pour les tests de langue
router.include_router(
    language_tests_router,
    prefix="/language-tests",
    tags=["language-tests"]
)

@router.get("/hello", response_model=MessageResponse)
async def hello():
    """
    Endpoint simple qui renvoie un message hello world
    """
    return MessageResponse(message="Hello World API") 