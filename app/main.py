from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import languages, language_tests
from app.db import create_tables

# Créer les tables de la base de données
create_tables()

app = FastAPI(title=settings.PROJECT_NAME)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(languages.router, prefix="/api", tags=["languages"])
app.include_router(language_tests.router, prefix="/api/tests", tags=["tests"])

@app.get("/")
async def root():
    """
    Route racine de l'application
    """
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 