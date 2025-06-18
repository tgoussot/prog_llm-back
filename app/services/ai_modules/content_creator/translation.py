from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import get_llm
from pydantic import BaseModel, Field

# Modèle simple pour les traductions (localement défini)
class TermesTraduction(BaseModel):
    comprehension_ecrite: str = Field(..., description="Traduction de 'Compréhension écrite'")
    expression_ecrite: str = Field(..., description="Traduction de 'Expression écrite'")
    grammaire: str = Field(..., description="Traduction de 'Grammaire'")
    vocabulaire: str = Field(..., description="Traduction de 'Vocabulaire'")
    consigne: str = Field(..., description="Traduction de 'Consigne'")
    contenu: str = Field(..., description="Traduction de 'Contenu'")
    niveau_cible: str = Field(..., description="Traduction de 'Niveau cible'")
    competence: str = Field(..., description="Traduction de 'Compétence'")
    question: str = Field(..., description="Traduction de 'Question'")
    phrase: str = Field(..., description="Traduction de 'Phrase'")
    item: str = Field(..., description="Traduction de 'Item'")
    texte_principal: str = Field(..., description="Traduction de 'Texte principal'")

def traduire_termes_techniques(langue_cible):
    """Traduit les termes techniques du système dans la langue cible."""
    llm = get_llm(temperature=0.1)
    
    # Si la langue cible est déjà le français, retourner les termes en français
    if langue_cible.lower() == "français":
        return TermesTraduction(
            comprehension_ecrite="Compréhension écrite",
            expression_ecrite="Expression écrite",
            grammaire="Grammaire",
            vocabulaire="Vocabulaire",
            consigne="Consigne",
            contenu="Contenu",
            niveau_cible="Niveau cible",
            competence="Compétence",
            question="Question",
            phrase="Phrase",
            item="Item",
            texte_principal="Texte principal"
        )
        
    # Traduire les termes vers la langue cible
    prompt = ChatPromptTemplate.from_template(
        """Tu es un traducteur technique spécialisé dans la didactique des langues.
        Traduis précisément les termes suivants du français vers {langue_cible}.
        
        Termes à traduire:
        - Compréhension écrite
        - Expression écrite
        - Grammaire
        - Vocabulaire
        - Consigne
        - Contenu
        - Niveau cible
        - Compétence
        - Question
        - Phrase
        - Item
        - Texte principal
        
        Fournis tes traductions au format structuré uniquement, sans explications.
        """
    )
    
    chain = prompt | llm.with_structured_output(TermesTraduction)
    return chain.invoke({"langue_cible": langue_cible})

def traduire_prompt(prompt_texte, termes, langue_cible):
    """Traduit un prompt vers la langue cible."""
    llm = get_llm(temperature=0.1)
    
    prompt = ChatPromptTemplate.from_template(
        """Tu es un traducteur expert dans la didactique des langues.
        Traduis précisément le texte suivant du français vers {langue_cible}.
        
        Voici les traductions des termes techniques à utiliser:
        {termes}
        
        Texte à traduire:
        {texte}
        
        La traduction doit être fidèle et naturelle dans la langue cible.
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "texte": prompt_texte, 
        "langue_cible": langue_cible,
        "termes": "\n".join([f"- {k}: {v}" for k, v in termes.__dict__.items() if not k.startswith("_")])
    }) 