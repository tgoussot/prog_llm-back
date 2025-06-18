from fastapi import APIRouter, HTTPException, Query, Depends
import json
from pathlib import Path
from typing import Dict, List, Optional
from app.schemas.language import LanguageCreate, Language
from app.services.language import create_language, get_languages
from app.db.session import get_db
from sqlalchemy.orm import Session
import logging

router = APIRouter()

# Charger le fichier JSON des langues (anglais)
try:
    with open(Path("app/data/languages.json"), "r", encoding="utf-8") as f:
        languages_data = json.load(f)
except FileNotFoundError:
    languages_data = {}

# Charger le fichier JSON des langues (français)
try:
    with open(Path("app/data/languages_fr.json"), "r", encoding="utf-8") as f:
        languages_fr_data = json.load(f)
except FileNotFoundError:
    languages_fr_data = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.info")

@router.get("/languages/", response_model=List[Language])
async def get_saved_languages(db: Session = Depends(get_db)):
    """Récupère toutes les langues sauvegardées dans la base de données"""
    return await get_languages(db)

@router.post("/languages/", response_model=Language)
async def add_language(language: LanguageCreate, db: Session = Depends(get_db)):
    """Ajoute une nouvelle langue à la base de données"""
    return create_language(db, language)

@router.get("/languages/search")
def search_languages(query: str, lang: str = "en", limit: int = 20):
    """Recherche des langues par nom"""
    query = query.lower()
    results = []
    
    if lang.lower() == "fr":
        data_source = languages_fr_data
        search_field = "name_fr"
    else:
        data_source = languages_data
        search_field = "name"
    
    for code, lang_info in data_source.items():
        if search_field in lang_info and query in lang_info[search_field].lower():
            # Créer un objet simplifié avec uniquement les champs requis
            simplified_lang = {
                "code": code,
                "name": lang_info.get("name_fr", lang_info.get("name", "")),
                "country_code": lang_info.get("country_code", ""),
                "country_name": lang_info.get("country_name", "")
            }
            
            # Utiliser le nom français si disponible
            if lang.lower() == "fr" and "name_fr" in lang_info:
                simplified_lang["name"] = lang_info["name_fr"]
            
            results.append(simplified_lang)
            
            if len(results) >= limit:
                break
    
    return results 