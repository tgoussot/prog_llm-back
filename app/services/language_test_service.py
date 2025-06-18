"""
Service pour la génération de tests de langue via l'IA
"""
import uuid
import sys
from app.schemas.language_test import (
    LanguageTestRequest, 
    LanguageTestResponse, 
    TestComplet
)

async def generate_language_test(request: LanguageTestRequest) -> LanguageTestResponse:
    """
    Génère un test de langue à partir des paramètres fournis
    
    Args:
        request: Requête contenant la langue et éventuellement le niveau cible
        
    Returns:
        Un test de langue complet
    """
    # Générer un identifiant unique pour ce test
    test_id = str(uuid.uuid4())
    
    # Gérer le cas où niveau_cible est None
    niveau_cible_str = request.niveau_cible if request.niveau_cible is not None else ""
    
    print(f"Génération d'un test pour la langue: {request.langue}, niveau: {niveau_cible_str}")
    
    # Import local pour éviter les imports circulaires
    try:
        from app.services.ai_modules.content_creator_ai import generer_test_parallele
        print("Module content_creator_ai importé avec succès")
        
        # Appeler l'IA de génération de test en mode parallèle (haute performance)
        test_result = generer_test_parallele(
            langue=request.langue,
            niveau_cible=niveau_cible_str
        )
        
        print("Test généré par l'IA avec succès")
        print(f"Type de résultat: {type(test_result)}")
        
    except ImportError as e:
        print(f"Erreur d'import: {str(e)}")
        print(f"Chemins de recherche Python: {sys.path}")
        print("Utilisation du mode fallback")
        
        # Fallback pour les environnements de test ou de développement
        from app.schemas.language_test import Exercice, Contenu, Element, TypeElement, OptionQCM
        
        # Créer un test minimal pour le développement avec QCM
        test_result = TestComplet(
            comprehension_ecrite=[
                Exercice(
                    consigne="Lisez le texte et répondez aux questions.",
                    contenu=Contenu(
                        texte_principal="Ceci est un texte d'exemple pour le développement.",
                        elements=[
                            Element(
                                id=1, 
                                texte="Quelle est la première phrase du texte?", 
                                type=TypeElement.QCM,
                                options=[
                                    OptionQCM(id="A", texte="Ceci est un texte d'exemple", est_correcte=True),
                                    OptionQCM(id="B", texte="Une autre réponse", est_correcte=False),
                                    OptionQCM(id="C", texte="Encore une autre", est_correcte=False),
                                    OptionQCM(id="D", texte="La dernière option", est_correcte=False)
                                ],
                                reponse_correcte="A"
                            )
                        ]
                    ),
                    niveau_cible="B1",
                    competence="Compréhension de texte informatif"
                )
            ],
            grammaire=[
                Exercice(
                    consigne="Complétez les phrases avec la forme correcte du verbe.",
                    contenu=Contenu(
                        texte_principal="",
                        elements=[
                            Element(
                                id=1, 
                                texte="Je ___ (aller) au cinéma hier soir.", 
                                type=TypeElement.QCM,
                                options=[
                                    OptionQCM(id="A", texte="vais", est_correcte=False),
                                    OptionQCM(id="B", texte="suis allé", est_correcte=True),
                                    OptionQCM(id="C", texte="irai", est_correcte=False),
                                    OptionQCM(id="D", texte="allais", est_correcte=False)
                                ],
                                reponse_correcte="B"
                            )
                        ]
                    ),
                    niveau_cible="B1",
                    competence="Conjugaison des verbes"
                )
            ],
            vocabulaire=[
                Exercice(
                    consigne="Choisissez le mot qui convient le mieux.",
                    contenu=Contenu(
                        texte_principal="",
                        elements=[
                            Element(
                                id=1, 
                                texte="Il fait très ___ aujourd'hui.", 
                                type=TypeElement.QCM,
                                options=[
                                    OptionQCM(id="A", texte="chaud", est_correcte=True),
                                    OptionQCM(id="B", texte="froid", est_correcte=False),
                                    OptionQCM(id="C", texte="humide", est_correcte=False),
                                    OptionQCM(id="D", texte="venteux", est_correcte=False)
                                ],
                                reponse_correcte="A"
                            )
                        ]
                    ),
                    niveau_cible="A2",
                    competence="Vocabulaire météorologique"
                )
            ]
        )
    except Exception as e:
        print(f"Erreur lors de la génération du test: {e}")
        raise
    
    # Créer la réponse avec le test généré
    response = LanguageTestResponse(
        id=test_id,
        langue=request.langue,
        niveau_cible=niveau_cible_str,
        test=test_result
    )
    
    return response 