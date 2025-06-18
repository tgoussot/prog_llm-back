# FastAPI Hello World
## Installation

1. Cloner le dépôt
2. Créer un environnement virtuel :
   ```
   python3 -m venv venv
   ```
   
3. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Lancement de l'application

Pour lancer l'application en mode développement :

```
uvicorn app.main:app --reload
```

L'API sera disponible à l'adresse : http://localhost:8000

La documentation interactive Swagger UI : http://localhost:8000/docs

## Structure du projet

```
backend/
├── app/
│   ├── api/             # Routes API
│   ├── core/            # Configurations et utilitaires
│   ├── models/          # Modèles de données
│   ├── schemas/         # Schémas Pydantic
│   └── main.py          # Point d'entrée de l'application
├── venv/                # Environnement virtuel
└── requirements.txt     # Dépendances
``` 
