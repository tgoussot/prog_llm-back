from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class LanguageTestRequest(BaseModel):
    """
    Requête pour générer un test de langue initial
    """
    langue: str = Field(..., description="Langue du test (français, anglais, espagnol, etc.)")
    niveau_cible: Optional[str] = Field("", description="Niveau CECRL ciblé (A1-C2), vide pour auto-détection")

# Définition du type d'élément dans un exercice
class TypeElement(str, Enum):
    QUESTION = "QUESTION"
    PHRASE = "PHRASE"
    ITEM = "ITEM"
    CONSIGNE = "CONSIGNE"
    QCM = "QCM"  # Type pour les questions à choix multiples

# Modèle pour les options de QCM
class OptionQCM(BaseModel):
    id: str = Field(..., description="Identifiant de l'option (A, B, C, D)")
    texte: str = Field(..., description="Texte de l'option de réponse")
    est_correcte: bool = Field(False, description="Indique si cette option est la bonne réponse")
    
    class Config:
        # Forcer la sérialisation de tous les champs
        exclude_none = False

class Element(BaseModel):
    """
    Élément dans un exercice (question, phrase, item, consigne, QCM)
    """
    id: int = Field(..., description="Identifiant unique de l'élément")
    texte: str = Field(..., description="Texte de la question, de la phrase ou de l'item")
    type: TypeElement = Field(..., description="Type de l'élément (QUESTION, PHRASE, ITEM, CONSIGNE, QCM)")
    
    # Champs spécifiques aux QCM
    options: Optional[List[OptionQCM]] = Field(None, description="Options de réponse pour les QCM")
    reponse_correcte: Optional[str] = Field(None, description="ID de la réponse correcte (A, B, C, D)")
    
    class Config:
        # Forcer la sérialisation de tous les champs, même None
        exclude_none = False

class Contenu(BaseModel):
    """
    Contenu d'un exercice
    """
    texte_principal: str = Field("", description="Texte principal (peut être vide pour grammaire/vocabulaire)")
    elements: List[Element] = Field(..., description="Liste des éléments de l'exercice")

class Exercice(BaseModel):
    """
    Exercice de langue complet
    """
    consigne: str = Field(..., description="Consigne claire de l'exercice")
    contenu: Contenu = Field(..., description="Contenu structuré de l'exercice")
    niveau_cible: str = Field(..., description="Niveau CECRL ciblé (A1-C2)")
    competence: str = Field(..., description="Compétence précise évaluée")

class TestComplet(BaseModel):
    """
    Test de langue complet avec tous les exercices
    """
    comprehension_ecrite: List[Exercice] = Field(default_factory=list)
    grammaire: List[Exercice] = Field(default_factory=list)
    vocabulaire: List[Exercice] = Field(default_factory=list)

class LanguageTestResponse(BaseModel):
    """
    Réponse contenant le test de langue généré
    """
    id: str = Field(..., description="Identifiant unique du test")
    langue: str = Field(..., description="Langue du test")
    niveau_cible: str = Field("", description="Niveau CECRL ciblé ou auto-détection")
    test: TestComplet = Field(..., description="Test complet généré") 