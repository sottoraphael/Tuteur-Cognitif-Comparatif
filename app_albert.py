import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import PyPDF2
import time

# ==========================================
# CONFIGURATION DE LA PAGE & CSS
# ==========================================
st.set_page_config(page_title="Réviser avec les sciences cognitives", page_icon="🦉", layout="centered")

st.markdown("""
    <style>
    .stApp { transition: all 0.1s ease-in-out; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button { width: 100%; border-radius: 15px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

MAX_HISTORIQUE_MESSAGES = 6
# Modèle Albert API cible
MODELE_ALBERT = "openai/gpt-oss-120b"
# Point d'accès (Endpoint) officiel Albert API
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"

# ==========================================
# GESTION DE L'ÉTAT DE SESSION (State)
# ==========================================
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "texte_cours_integral" not in st.session_state:
    st.session_state.texte_cours_integral = ""
if "tutoriel_vu" not in st.session_state:
    st.session_state.tutoriel_vu = False

# ==========================================
# --- TUTORIEL D'ACCUEIL ---
# ==========================================
@st.dialog("👋 Bienvenue dans cette application de révision")
def afficher_tutoriel():
    st.markdown("""
        <style>
        .big-font { font-size: 1.25rem !important; line-height: 1.7 !important; color: #2D3748; }
        .step-title { font-weight: bold; color: #5B9BD5; font-size: 1.35rem; display: block; margin-top: 15px; }
        .mode-box { background-color: #F0F4F8; padding: 15px; border-radius: 12px; margin: 15px 0; border-left: 6px solid #5B9BD5; }
        </style>
        <div class="big-font">
        Cette application utilise les principes issus des <b>sciences cognitives</b> pour t'aider à réviser efficacement.<br>
        <div class="mode-box">
        <b>💡 Quel mode choisir ?</b><br><br>
        • <b>Mémorisation :</b> Pour retenir les définitions et les concepts "par cœur".<br><br>
        • <b>Compréhension :</b> Pour maîtriser ton cours en profondeur en l'expliquant avec tes propres mots.
        </div>
        <b>Comment l'utiliser en 3 étapes :</b><br>
        <span class="step-title">1. ⚙️ Règle l'application</span> Choisis ton mode et ton niveau.<br>
        <span class="step-title">2. 🧭 Donne-lui ton cours</span> Charge ton PDF ou colle ton texte.<br>
        <span class="step-title">3. 💬 Discute</span> Réponds aux questions dans le chat, et demande ton bilan à la fin !
        </div><br>
    """, unsafe_allow_html=True)
    if st.button("🚀 J'ai compris, c'est parti !", use_container_width=True):
        st.session_state.tutoriel_vu = True
        st.rerun()

# ==========================================
# --- DIALOGUE BILAN FINAL & WOOCLAP ---
# ==========================================
@st.dialog("📈 Ton Bilan de Révision", width="large")
def afficher_bilan():
    if len(st.session_state.messages) > 1:
        with st.spinner("Analyse métacognitive en cours..."):
            
            # Initialisation du client Albert API
            client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
            messages_bilan = []
            
            # 1. On prépare la base commune (Acquis et Erreurs)
            instruction_metacognitive = """Tu es un coach pédagogique. Fais un bilan métacognitif factuel, ultra-concis et encourageant. Adresse-toi à l'élève avec 'Tu'. Ne pose plus de question.

            CONTRAINTE STRICTE : Ton bilan doit être extrêmement bref, visuel et direct. Utilise des listes à puces et limite-toi à 1 ou 2 phrases maximum par point. Pas de longs paragraphes.

            Structure obligatoirement ton bilan ainsi :
            1. 🎯 Tes acquis : Va droit au but sur ce qui est su et ce qui reste à revoir (très bref).
            2. 💡 Tes erreurs : Dédramatise et donne LA stratégie précise à utiliser la prochaine fois (1 phrase).
            """

            # 2. On ajoute les conseils spécifiques selon le mode choisi (Le piège et l'étape "Maison")
            if "Mode A" in st.session_state.objectif:
                instruction_metacognitive += """3. ⏳ Le piège de la relecture : Rappelle en 1 courte phrase que relire le cours donne l'illusion de savoir (biais de fluence) et que seul l'effort de mémoire compte.
            4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : cacher son cours et forcer son cerveau à retrouver les informations sur une feuille blanche.
            """
            else:
                instruction_metacognitive += """3. ⏳ Le piège de la correction : Rappelle en 1 courte phrase que lire une correction donne l'illusion d'avoir compris. La vraie compréhension, c'est savoir l'expliquer soi-même.
            4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : reprendre un exercice et expliquer la méthode à voix haute comme à un camarade, ou chercher les erreurs.
            """

            messages_bilan.append({"role": "system", "content": instruction_metacognitive})

            if st.session_state.texte_cours_integral:
                messages_bilan.append({"role": "user", "content": f"BASE DE CONNAISSANCES DU COURS :\n{st.session_state.texte_cours_integral}"})
                messages_bilan.append({"role": "assistant", "content": "Compris."})
            
            for msg in st.session_state.messages:
                messages_bilan.append({"role": msg["role"], "content": msg["content"]})
                
            messages_bilan.append({"role": "user", "content": "La session est terminée. Donne-moi mon bilan métacognitif ultra-concis selon tes instructions."})

            try:
                # Appel synchrone (sans stream) pour le bilan global
                reponse = client.chat.completions.create(
                    model=MODELE_ALBERT,
                    messages=messages_bilan,
                    temperature=0.7
                )
                
                st.success(reponse.choices[0].message.content)
                
                st.divider()
                
                st.markdown("### 📊 Évaluation de l'outil")
                st.write("Aide-nous à améliorer cette application en répondant à ce court questionnaire anonyme :")
                
                iframe_wooclap = """<iframe allowfullscreen frameborder="0" height="100%" mozallowfullscreen src="https://app.wooclap.com/FBXMBG/questionnaires/69ad313cc7cb13027e159133" style="min-height: 550px; min-width: 300px" width="100%"></iframe>"""
                components.html(iframe_wooclap, height=580)
                
                st.divider()
                
                if st.button("🔄 J'ai terminé, recommencer une nouvelle session", type="primary"):
                    st.session_state.session_active = False
                    st.session_state.messages = []
                    st.session_state.texte_cours_integral = ""
                    st.rerun()
            except Exception as e:
                st.error(f"Impossible de générer le bilan pour le moment : {e}")
    else:
        st.warning("Il faut d'abord discuter un peu avec le tuteur avant de pouvoir analyser tes réponses !")

# ==========================================
# 🛑 ZONE SANCTUAIRE : PROMPT SYSTÈME EXACT 🛑
# ==========================================
def generer_prompt_systeme(niveau_eleve, objectif_eleve, strategie_generative=None):  
    prompt_systeme = """<instructions_tuteur>
<role_et_mission>
Tu es un expert en ingénierie pédagogique cognitive et spécialiste EdTech.
Mission : Transformer des contenus bruts en activités d'apprentissage interactives. 
Contrainte absolue : Base-toi EXCLUSIVEMENT sur le document "BASE DE CONNAISSANCES DU COURS" fourni par l'utilisateur.
Objectif : Réduire la distance entre la compréhension de l'élève et la cible pédagogique, tout en développant sa métacognition.
</role_et_mission>

<gestion_mathematique>
- Tolérance syntaxique : L'élève ne dispose pas de clavier mathématique. Il saisira en texte brut (ex: "racine de x", "x au carre"). Tu DOIS interpréter ces notations pour évaluer son raisonnement.
- Affichage : Dans tes réponses, utilise systématiquement le format LaTeX (encadré par $) pour formater les formules (ex: $\\frac{x}{2}$) afin de minimiser la charge cognitive extrinsèque visuelle de l'élève.
</gestion_mathematique>

<interdictions_strictes>
- NE DONNE JAMAIS LA SOLUTION D'EMBLÉE.
- PAS DE JUGEMENT : Ne dis jamais "Tu es nul" ni de "Bravo !" vagues sans explication factuelle.
- AUCUNE COMPARAISON SOCIALE avec d'autres élèves.
- ANTI-HALLUCINATION : N'invente aucune règle hors du cours fourni. Si une donnée manque, réponds "Non rapporté dans le document".
</interdictions_strictes>

<directives_interaction>
1. Flux unitaire : Pose UNE SEULE question à la fois. Attends obligatoirement la réponse.
2. Maïeutique (Règle des 2 itérations) : Fournis des indices. SI l'élève échoue 2 fois de suite sur la même question, déclenche silencieusement la <structure_3_remediation>.
3. Concision : Feedbacks limités à 2 ou 3 phrases MAXIMUM. Aucun cours magistral.
4. Transparence Cognitive : Ne nomme pas tes étapes de réflexion à l'élève. En revanche, explique-lui simplement POURQUOI ton exercice aide son cerveau (ex: "pour forcer ton cerveau à faire des liens", "pour mémoriser plus longtemps"). Reste naturel.
5. Balayage : Scanne le document de haut en bas. Passe au concept suivant dès que l'objectif de la question est atteint, ou après un échec traité par la remédiation. Ne bloque jamais indéfiniment.
6. Clôture : À la fin du document, stoppe le questionnement. Invite l'élève à cliquer sur "Terminer et voir ma synthèse", puis à pratiquer l'apprentissage espacé en revenant dans quelques jours.
</directives_interaction>

<structures_feedback_obligatoires>
Choisis et applique implicitement UNE des trois structures suivantes pour ta réponse :

- <structure_1_processus> : 
  1. Constat factuel (Valide/Invalide). 
  2. Diagnostic précis du blocage. 
  3. Levier stratégique (indice cognitif SANS donner la réponse finale). Pousse l'élève à réfléchir.

- <structure_2_autoregulation> : 
  1. Effet miroir (décris l'action de l'élève). 
  2. Activation radar (fais-le réfléchir sur l'efficacité de sa méthode). 
  3. Ouverture (pousse à l'action corrective sans donner la solution).

- <structure_3_remediation> (EXCLUSIVEMENT APRÈS 2 ÉCHECS) : 
  1. Stoppe le questionnement. 
  2. Donne la bonne réponse exacte et démontre pas-à-pas avec le vocabulaire du cours. 
  3. Pose une question isomorphe (même logique, variables différentes) pour vérifier l'intégration.
</structures_feedback_obligatoires>
"""

    if niveau_eleve == "Novice":
        prompt_systeme += """
<profil_apprenant niveau="Novice">
- État cognitif : En construction, fort risque de surcharge cognitive.
- Règle 1 : INTERDICTION ABSOLUE d'utiliser la <structure_2_autoregulation>.
- Règle 2 : Utilise EXCLUSIVEMENT la <structure_1_processus> (guidage pas-à-pas) ou la <structure_3_remediation> (si 2 échecs).
</profil_apprenant>
"""
    else:
        prompt_systeme += """
<profil_apprenant niveau="Avancé">
- État cognitif : Possède les bases, sujet à l'étourderie ou l'excès de confiance.
- Règle 1 : Si erreur de méthode -> Active <structure_1_processus>.
- Règle 2 : Si étourderie -> Active <structure_2_autoregulation> pour générer un conflit cognitif.
</profil_apprenant>
"""

    if "Mode A" in objectif_eleve:
        prompt_systeme += """
<cadre_exercice mode="Ancrage_Memorisation">
- Principe : Testing Effect (1 question = 1 savoir atomique).
- Stratégie de leurre : Utilise des distracteurs basés sur la confusion conceptuelle, l'erreur intuitive ou l'inversion causale. Les leurres doivent être homogènes.
- Obligation : Explique toujours POURQUOI la réponse est juste ou fausse.
"""
        if niveau_eleve == "Novice":
            prompt_systeme += """- Format d'échafaudage : Utilise EXCLUSIVEMENT des QCM (laisse une ligne vide entre chaque choix).
</cadre_exercice>
"""
        else:
            prompt_systeme += """- Format d'échafaudage : Utilise EXCLUSIVEMENT le Rappel Libre (question directe sans choix multiples).
</cadre_exercice>
"""
    else:
        prompt_systeme += """
<cadre_exercice mode="Comprehension_Generative">
- Principe : L'élève a le cours sous les yeux. Le but est de créer des liens (modèle S.O.I.).
- Obligation : Avant de donner la correction finale, demande toujours à l'élève d'évaluer sa propre production (ex: "As-tu oublié un élément important ?").
"""
        if strategie_generative == "Effet_Protege":
            prompt_systeme += """
<jeu_de_role persona="Sacha_Camarade">
ATTENTION : TU N'ES PLUS LE TUTEUR. Tu es "Sacha", un camarade de classe qui ne comprend pas le cours. 
Ton but caché : Obliger l'utilisateur à vulgariser (Effet Protégé).

<regles_sacha>
1. Anti-récitation : N'utilise aucun jargon. Rejette les explications trop scolaires.
2. Scaffolding naïf : Explicite ta confusion dès le début. Pose UNE question naïve. Coupe l'élève s'il va trop vite.
3. Erreur intentionnelle : Injecte une erreur classique de novice pour forcer l'élève à te corriger.
4. Entêtement : Si l'élève valide ton erreur, aggrave-la au tour suivant.
5. Limite de blocage : Si l'élève échoue 2 fois à t'expliquer, simule une trouvaille ("Attends, le manuel dit que... du coup comment on fait ?").
6. Déclic : Si l'élève réussit, valorise sa pédagogie ("Ah, j'ai compris grâce à ton exemple !"). Demande-lui une question piège pour te tester.
</regles_sacha>
</jeu_de_role>
</cadre_exercice>
"""
        else:
            prompt_systeme += """
<posture_generative>
- RÈGLE STRICTE : Bannis les questions littérales (ne demande pas de retrouver une phrase du texte).
- Stratégies (choisis-en une) :
  1. Amorçage : Pose des questions d'inférence avant lecture complète.
  2. Auto-explication : Demande de justifier une information correcte du document (pas l'erreur de l'élève).
  3. Résumé : Exige une réorganisation, refuse la paraphrase.
  4. Détection d'erreurs : Rédige un calcul ou paragraphe contenant une erreur typique et fais chercher la règle violée.
</posture_generative>
"""
            if niveau_eleve == "Novice":
                prompt_systeme += """
<echafaudage_generatif niveau="Novice">
- Structure : Impose 3 à 5 mots-clés obligatoires à utiliser.
- Détection d'erreurs : Indique EXACTEMENT où est l'erreur, l'élève doit juste expliquer pourquoi.
</echafaudage_generatif>
</cadre_exercice>
"""
            else:
                prompt_systeme += """
<echafaudage_generatif niveau="Avancé">
- Structure : Questions larges sans mots-clés imposés.
- Détection d'erreurs : Ne dis pas où est l'erreur. L'élève doit l'identifier et la justifier seul.
</echafaudage_generatif>
</cadre_exercice>
"""

    prompt_systeme += """
<exemples_few_shot>
- Feedback Processus + Transparence : "Tu as bien identifié que la photosynthèse nécessite de la lumière. Cependant, il manque un gaz. Pour forcer ton cerveau à faire le lien, pense à ce que nous expirons : la plante utilise ce même gaz. Quel est-il ?"
- Feedback Autorégulation : "Tu as écrit que la Révolution a commencé en 1792. Regarde la chronologie dans ton document. Quel événement de 1789 marque réellement le début ?"
</exemples_few_shot>
</instructions_tuteur>"""

    return prompt_systeme
    
# POSTURE TUTEUR COGNITIF (INFÉRENCE ET GÉNÉRATION)
RÈGLE D'INFÉRENCE STRICTE : Bannis les questions littérales. Ne demande jamais de retrouver une information explicitement écrite. Force l'élève à déduire des liens (causaux, chronologiques) ou à cibler le "Pourquoi".

MENU GÉNÉRATIF (Choisis la stratégie la plus pertinente si non précisée) :
1. Pré-test (Amorçage) : Pose 3 à 5 questions d'inférence ciblées AVANT la lecture complète.
2. Auto-explication ciblée : Demande à l'élève de justifier une information ou une étape CORRECTE du document (ex: "Quelle hypothèse scientifique justifie ce calcul/ce choix ?"). Ne lui demande pas de justifier son propre raisonnement initial pour éviter d'ancrer ses erreurs.
3. Résumé avec ses mots : Refuse la paraphrase littérale. Exige une réorganisation personnelle.
4. Détection d'erreurs : Rédige un court paragraphe, calcul ou raisonnement contenant une erreur typique de la discipline, et force l'élève à inférer la règle violée.
"""
        if niveau_eleve == "Novice" and strategie_generative != "Effet_Protege":
            prompt_systeme += """
# ÉCHAFAUDAGE NOVICE
- Consignes très structurées : Impose 3 à 5 mots-clés OBLIGATOIRES du cours.
- Détection d'erreurs : Indique précisément OÙ se trouve l'erreur dans ton texte, la seule tâche de l'élève est d'expliquer pourquoi c'est faux.
- Support : Utilise des textes à trous pour guider l'inférence.
"""
        elif niveau_eleve != "Novice" and strategie_generative != "Effet_Protege":
            prompt_systeme += """
# ÉCHAFAUDAGE AVANCÉ
- Consignes ouvertes : Pose des questions larges SANS fournir de mots-clés.
- Détection d'erreurs : Ne dis pas où est l'erreur. L'élève doit chercher, identifier ET justifier l'erreur seul.
"""

    return prompt_systeme

# ==========================================
# FONCTIONS TECHNIQUES & EXTRACTION PDF
# ==========================================
def extraire_texte_pdf(uploaded_file):
    """Extrait l'intégralité du texte d'un fichier PDF page par page."""
    texte_complet = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        nb_pages = len(pdf_reader.pages)
        for num_page in range(nb_pages):
            page = pdf_reader.pages[num_page]
            texte_page = page.extract_text()
            if texte_page:
                texte_complet += f"\n--- Page {num_page + 1} ---\n{texte_page}"
        return texte_complet
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        return None

def construire_messages_albert(prompt_systeme, historique, nouvel_input, texte_cours):
    """Formate le contexte selon le standard OpenAI requis par Albert API."""
    messages = [{"role": "system", "content": prompt_systeme}]
    
    # Injection systématique de la base de connaissances (cours intégral)
    if texte_cours:
        messages.append({"role": "user", "content": f"BASE DE CONNAISSANCES DU COURS :\n{texte_cours}"})
        messages.append({"role": "assistant", "content": "J'ai bien mémorisé l'intégralité de la base de connaissances. Je suis prêt à formuler mes questions en me basant strictement sur ce contenu."})

    # Ajout de l'historique conversationnel récent
    for msg in historique[-MAX_HISTORIQUE_MESSAGES:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Ajout de la nouvelle entrée de l'élève
    if nouvel_input:
        messages.append({"role": "user", "content": nouvel_input})
        
    return messages

def extraire_texte_stream(reponse):
    """Générateur sécurisé pour lire le flux (stream) du modèle mot à mot."""
    for chunk in reponse:
        # Vérification 1 : Le paquet possède-t-il bien l'attribut 'choices' et n'est-il pas vide ?
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            # Vérification 2 : Le delta contient-il un attribut 'content' avec du texte ?
            if hasattr(delta, 'content') and delta.content is not None:
                yield delta.content

# ==========================================
# INTERFACE UTILISATEUR (UI)
# ==========================================
st.title("🦉 Réviser avec les sciences cognitives")
st.markdown("*Outil anonyme : Ne saisis aucune donnée personnelle dans ce chat.*")

if not st.session_state.tutoriel_vu:
    afficher_tutoriel()

# --- PANNEAU LATÉRAL ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    session_en_cours = st.session_state.session_active
    
    niveau_eleve = st.radio("Ton niveau :", ["Novice", "Avancé"], disabled=session_en_cours)
    objectif_eleve = st.radio("Ton objectif :", ["Mode A : Mémorisation", "Mode B : Compréhension"], disabled=session_en_cours)
    
    strat_display = "Classique"
    strategie_generative_val = "Classique"
    
    if "Mode B" in objectif_eleve:
        strat_display = st.radio(
            "Stratégie de révision :", 
            ["Classique", "Explique à un camarade"], 
            disabled=session_en_cours
        )
        if strat_display == "Explique à un camarade":
            strategie_generative_val = "Effet_Protege"

    st.divider()
    
    source_type = st.radio("Source du cours :", ["Fichier PDF", "Texte libre"], disabled=session_en_cours)
    
    if source_type == "Fichier PDF":
        uploaded_file = st.file_uploader("Charge ton cours (PDF)", type=["pdf"], disabled=session_en_cours)
        txt_input = None
    else:
        txt_input = st.text_area("Colle ton texte de cours ici :", height=200, disabled=session_en_cours, placeholder="Ex: La mitochondrie est l'organite responsable de la respiration cellulaire...")
        uploaded_file = None
    
    pret_a_demarrer = uploaded_file is not None or (txt_input is not None and len(txt_input.strip()) > 10)
    
    if st.button("🚀 Démarrer la session", disabled=session_en_cours or not pret_a_demarrer):
        try:
            # Récupération de la clé Albert API dans les secrets Streamlit
            api_key = st.secrets["ALBERT_API_KEY"]
            
            # Traitement dynamique du contenu textuel au démarrage
            if uploaded_file:
                with st.spinner("⏳ Extraction du contenu complet de ton document..."):
                    texte_extrait = extraire_texte_pdf(uploaded_file)
                    if texte_extrait:
                        st.session_state.texte_cours_integral = texte_extrait
                    else:
                        st.stop()
            else:
                st.session_state.texte_cours_integral = txt_input
            
            st.session_state.api_key = api_key
            st.session_state.niveau = niveau_eleve
            st.session_state.objectif = objectif_eleve
            st.session_state.strategie = strategie_generative_val
            st.session_state.session_active = True
            st.rerun()
        except KeyError:
            st.error("⚠️ La clé ALBERT_API_KEY est introuvable dans l'onglet 'Secrets' de Streamlit Cloud.")
        except Exception as e:
            st.error(f"Erreur : {e}")

    if st.session_state.session_active:
        st.divider()
        if st.button("🛑 Terminer et voir ma synthèse"):
            afficher_bilan()

# --- ZONE DE DISCUSSION ---
if st.session_state.session_active:
    # Initialisation du client Albert API
    client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
    prompt_sys = generer_prompt_systeme(st.session_state.niveau, st.session_state.objectif, st.session_state.strategie)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Amorçage (1ère question)
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            with st.spinner("Je prépare l'exercice..."):
                messages = construire_messages_albert(
                    prompt_sys, 
                    st.session_state.messages, 
                    "Salut ! Je suis prêt, commence l'exercice sur le cours.", 
                    st.session_state.texte_cours_integral
                )
                
                reponse_stream = client.chat.completions.create(
                    model=MODELE_ALBERT,
                    messages=messages,
                    stream=True,
                    temperature=0.3 # Température basse pour privilégier la rigueur logique
                )
                
                reponse_complete = st.write_stream(extraire_texte_stream(reponse_stream))
                st.session_state.messages.append({"role": "assistant", "content": reponse_complete})

    # Boucle d'interaction
    if prompt := st.chat_input("Écris ta réponse ici..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyse de ta réponse..."):
                messages = construire_messages_albert(
                    prompt_sys, 
                    st.session_state.messages[:-1], # On exclut le dernier message car on l'envoie en paramètre
                    prompt, 
                    st.session_state.texte_cours_integral
                )
                
                reponse_stream = client.chat.completions.create(
                    model=MODELE_ALBERT,
                    messages=messages,
                    stream=True,
                    temperature=0.3
                )
                
                # Gestion sécurisée du streaming
                reponse_complete = ""
                try:
                    reponse_complete = st.write_stream(extraire_texte_stream(reponse_stream))
                except Exception as e:
                    st.error("Une erreur réseau est survenue avec le serveur Albert.")
                    
        st.session_state.messages.append({"role": "assistant", "content": reponse_complete})

else:
    st.info("👈 Choisis tes paramètres et donne-moi ton cours pour commencer !")



