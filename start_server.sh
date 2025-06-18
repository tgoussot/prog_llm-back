#!/bin/bash

# Régler les problèmes de dépendances
pip install -r requirements.txt

# Définir la variable d'environnement PYTHONPATH pour inclure les dossiers parents
export PYTHONPATH="$PYTHONPATH:$(dirname $(pwd))"

# Lancer le serveur FastAPI
echo "Démarrage du serveur FastAPI..."
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 