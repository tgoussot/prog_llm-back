import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .utils import retry_with_backoff, safe_api_call, get_llm

@retry_with_backoff(max_retries=3, base_delay=15)
def generer_themes_aleatoires(langue="français", nombre=2, categorie="compréhension"):
    """Génère des thèmes aléatoires à l'aide de l'IA plutôt que d'utiliser des listes prédéfinies"""
    llm = get_llm(temperature=1.0)  # Température maximale pour maximiser la créativité et la diversité
    
    # Thèmes à éviter car surreprésentés
    themes_a_eviter = ["gastronomie", "cuisine", "nourriture", "plats", "alimentation", 
                       "traditions culinaires", "spécialités régionales", "marchés de Noël",
                       "vacances", "famille", "école", "travail quotidien", "shopping",
                       "météo", "sport basique", "animaux domestiques", "couleurs"]
    
    # Descriptions en français qui seront traduites si nécessaire
    descriptions_fr = {
        "compréhension": "thèmes intellectuellement stimulants et contemporains pour des textes de compréhension écrite avancée",
        "domaines": "domaines lexicaux spécialisés et sophistiqués pour des exercices de vocabulaire expert"
    }
    
    # Obtenir la description en français ou utiliser une description générique si la catégorie n'existe pas
    description_fr = descriptions_fr.get(categorie, "thèmes contemporains et stimulants pour des exercices de langue avancés")
    
    # Traduire la description si nécessaire (sauf pour le français)
    if langue.lower() == "français":
        description = description_fr
    else:
        # Utiliser l'IA pour traduire la description
        prompt_traduction = ChatPromptTemplate.from_template(
            """Traduis précisément cette phrase du français vers {langue}:
            
            Phrase: "{texte}"
            
            Donne UNIQUEMENT la traduction sans aucune autre explication."""
        )
        
        chaine_traduction = prompt_traduction | llm | StrOutputParser()
        description = safe_api_call(
            chaine_traduction.invoke,
            {"langue": langue, "texte": description_fr}
        ).strip()
    
    # Thèmes de secours variés et sophistiqués
    themes_secours = [
        "l'éthique de l'intelligence artificielle", "la bioéthique moderne", "l'économie circulaire", 
        "la géopolitique énergétique", "l'urbanisme durable", "la neuroscience cognitive",
        "la philosophie environnementale", "l'innovation sociale", "la cryptomonnaie et société", 
        "l'architecture bioclimatique", "la sociologie numérique", "l'anthropologie culturelle",
        "la médecine personnalisée", "la diplomatie culturelle", "l'ingénierie génétique",
        "la psychologie comportementale", "l'astrophysique contemporaine", "l'économie collaborative",
        "la critique d'art moderne", "la linguistique computationnelle", "la cybersécurité éthique",
        "l'écologie industrielle", "la philosophie des sciences", "l'innovation pédagogique",
        "la sociologie urbaine", "l'anthropologie digitale", "la physique quantique appliquée",
        "l'éthique médicale", "la géographie humaine", "l'histoire des mentalités",
        "la sémiologie des médias", "l'épistémologie moderne", "la théorie des systèmes complexes"
    ]
    
    # Domaines thématiques sophistiqués pour inspiration
    domaines_inspiration = [
        "sciences cognitives", "géopolitique contemporaine", "innovations technologiques",
        "enjeux environnementaux", "transformations sociétales", "philosophie appliquée",
        "économie comportementale", "anthropologie moderne", "éthique professionnelle",
        "histoire des idées", "sociologie des organisations", "psychologie sociale",
        "épistémologie des sciences", "théories de la communication", "critique culturelle"
    ]
    
    # Création du prompt sophistiqué pour générer des thèmes
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Tu es un expert en didactique des langues et un intellectuel spécialisé dans la création 
        de matériel pédagogique sophistiqué en {langue}. Tu maîtrises parfaitement les enjeux contemporains, 
        les débats intellectuels actuels, et les domaines de recherche avancés.
        
        Ta mission: créer {nombre} {description} qui défient intellectuellement les apprenants 
        et les exposent à des problématiques contemporaines riches et nuancées."""),
        
        ("human", f"""Génère {nombre} {description} ORIGINAUX, SOPHISTIQUÉS et INTELLECTUELLEMENT STIMULANTS.

        EXIGENCES DE QUALITÉ MAXIMALE:
        1. ORIGINALITÉ: Évite absolument les thèmes banals comme {', '.join(themes_a_eviter[:10])}
        2. SOPHISTICATION: Privilégie les sujets qui nécessitent réflexion, analyse et esprit critique
        3. CONTEMPORANÉITÉ: Intègre les enjeux actuels de société, science, technologie, culture
        4. DIVERSITÉ: Couvre différents domaines intellectuels: {', '.join(domaines_inspiration[:8])}
        5. SPÉCIFICITÉ: Chaque thème doit être précis, pas générique
        6. AUTHENTICITÉ: Pertinents pour la culture des locuteurs de {langue}
        
        TYPOLOGIE DE THÈMES RECHERCHÉS:
        - Débats éthiques contemporains (IA, biotechnologies, environnement...)
        - Enjeux géopolitiques actuels
        - Innovations scientifiques et leurs implications
        - Transformations sociétales et culturelles
        - Philosophie appliquée aux problèmes modernes
        - Économie comportementale et nouveaux modèles
        - Psychologie sociale et cognitive
        - Critique culturelle et artistique contemporaine
        - Anthropologie des sociétés modernes
        - Épistémologie et théorie de la connaissance
        
        INTERDICTIONS STRICTES:
        - Aucun thème de la liste: {', '.join(themes_a_eviter)}
        - Évite les sujets trop généraux ou évidents
        - Pas de thèmes répétitifs ou similaires entre eux
        - Bannir les clichés et stéréotypes culturels
        
        FORMAT DE RÉPONSE:
        Réponds UNIQUEMENT avec {nombre} thèmes TRÈS DIFFÉRENTS les uns des autres, 
        un par ligne, sans numérotation ni explications. Chaque thème doit être formulé 
        de manière précise et engageante.""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({})
    
    # Nettoyer le résultat et le transformer en liste
    themes = [theme.strip() for theme in result.strip().split('\n') if theme.strip()]
    
    # Filtrer pour éviter les thèmes interdits
    themes_filtres = []
    for theme in themes:
        theme_lower = theme.lower()
        # Vérifier qu'aucun mot interdit n'est présent
        if not any(mot_interdit in theme_lower for mot_interdit in themes_a_eviter):
            themes_filtres.append(theme)
    
    # Vérifier si nous avons assez de thèmes après filtrage
    if len(themes_filtres) < nombre:
        # Mélanger et utiliser des thèmes de secours sophistiqués si nécessaire
        random.shuffle(themes_secours)
        themes_filtres.extend(themes_secours[:nombre - len(themes_filtres)])
    
    # Limiter au nombre demandé au cas où l'IA en génère plus
    return themes_filtres[:nombre]

