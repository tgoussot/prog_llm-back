from .base_class import Base
from .session import SessionLocal, engine

# Créer toutes les tables
def create_tables():
    Base.metadata.create_all(bind=engine) 