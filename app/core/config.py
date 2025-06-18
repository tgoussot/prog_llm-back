from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """
    Configuration de l'application
    """
    PROJECT_NAME: str = "Language Learning App"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173"]
    DATABASE_URL: str = "sqlite:///./app.db"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"  # Permet les champs supplémentaires
    }

settings = Settings() 