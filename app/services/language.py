from sqlalchemy.orm import Session
from app.models.language import Language as LanguageModel
from app.schemas.language import LanguageCreate
from typing import List

def create_language(db: Session, language: LanguageCreate):
    """Crée une nouvelle langue dans la base de données"""
    db_language = LanguageModel(
        name=language.name,
        country_code=language.country_code,
        country_name=language.country_name
    )
    db.add(db_language)
    db.commit()
    db.refresh(db_language)
    return db_language

async def get_languages(db: Session, skip: int = 0, limit: int = 100):
    """Récupère toutes les langues de la base de données"""
    return db.query(LanguageModel).offset(skip).limit(limit).all() 