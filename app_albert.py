import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import PyPDF2
import sympy as sp
import json
import time
import spacy
from pydantic import BaseModel, Field, ValidationError

import referentiels 

# ==========================================
# CONFIGURATION DE LA PAGE & CSS
# ==========================================
st.set_page_config(page_title="Sacha - Tuteur Cognitif", page_icon="🦉", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; transition: all 0.1s ease-in-out; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stButton>button { width: 100%; border-radius: 15px; font-weight: bold; background-color: #1E40AF; color: white; }
    .stChatMessage { border-radius: 15px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

MAX_HISTORIQUE_MESSAGES = 6
MODELE_ALBERT = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
ALBERT_BASE_URL = "https://albert.api.etalab.gouv.fr/v1"

# ==========================================
# GESTION DE L'ÉTAT DE SESSION (State)
# ==========================================
if "session_active" not in st.session_state: st.session_state.session_active = False
if "messages" not in st.session_state: st.session_state.messages = []
if "texte_cours_integral" not in st.session_state: st.session_state.texte_cours_integral = ""
if "tutoriel_vu" not in st.session_state: st.session_state.tutoriel_vu = False

if "niveau_cog" not in st.session_state: st.session_state.niveau_cog = "Novice"
if "objectif" not in st.session_state: st.session_state.objectif = "Mode A : Mémorisation"
if "strategie" not in st.session_state: st.session_state.strategie = "Classique"
if "matiere" not in st.session_state: st.session_state.matiere = ""
if "niveau_scolaire" not in st.session_state: st.session_state.niveau_scolaire = ""
if "attendus" not in st.session_state: st.session_state.attendus = None
if "api_key" not in st.session_state: st.session_state.api_key = ""

# Variables pour la Mémoire de Travail (Étape 2)
if "resume_memoire" not in st.session_state: st.session_state.resume_memoire = ""
if "index_resume" not in st.session_state: st.session_state.index_resume = 0

# ==========================================
# SCHÉMAS PYDANTIC (MÉTACOGNITION IA)
# ==========================================
class ReflexionTuteur(BaseModel):
    """Schéma imposant la réflexion avant l'action (Inhibition)."""
    diagnostic_interne: str = Field(description="Analyse factuelle de la réponse de l'élève. Invisible pour l'élève.")
    strategie_choisie: str = Field(description="Catégorisation stricte de l'intervention (ex: Feedback de Processus, Remédiation, etc.).")
    reponse_visible: str = Field(description="Le texte final adressé à l'élève, respectant le format LaTeX et la Transparence Cognitive.")

class ValidationDidactique(BaseModel):
    """Schéma de l'Agent Critique."""
    contient_analogie: bool = Field(description="La réponse contient-elle une analogie ou un exemple concret ?")
    entites_physiques_detectees: list[str] = Field(description="Liste des objets physiques mentionnés.")
    est_valide_physiquement: bool = Field(description="True si l'analogie est logique, False si physiquement impossible.")
    motif_rejet: str = Field(description="Explication brève en cas d'aberration didactique.")

# ==========================================
# PROGRAMMATION ORIENTÉE OBJET (AGENTS)
# ==========================================
class AgentMathematique:
    """Moteur symbolique déterministe (SymPy)."""
    @staticmethod
    def verifier(expression_prof, expression_eleve):
        try:
            exp_p = sp.simplify(str(expression_prof).replace('^', '**'))
            exp_e = sp.simplify(str(expression_eleve).replace('^', '**'))
            est_valide = sp.simplify(exp_p - exp_e) == 0
            return {"est_valide": est_valide, "forme_simplifiee_eleve": str(exp_e)}
        except Exception:
            return {"erreur": "Syntaxe mathématique non reconnue. Demande à l'élève de clarifier."}

class AgentCritique:
    """Filtre exécutif basé sur NLP (spaCy)."""
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
            self.nlp = spacy.load("fr_core_news_sm")

    def analyser(self, texte_reponse, client):
        doc = self.nlp(texte_reponse)
        phrases_longues = [sent.text for sent in doc.sents if len([t for t in sent if not t.is_punct]) > 30]
        if phrases_longues:
            return False, f"Surcharge cognitive. Ta phrase est trop longue ({len(phrases_longues[0].split())} mots). Scinde tes idées."

        risque_negatif = any(token.text.startswith('-') and token.pos_ == "NUM" for token in doc)
        if not risque_negatif:
            return True, ""

        prompt_critique = f"""Analyse cette proposition pédagogique d'un tuteur : "{texte_reponse}"
Détermine si elle contient une aberration didactique ou physique."""
        try:
            reponse = client.chat.completions.create(
                model=MODELE_ALBERT,
                messages=[{"role": "user", "content": prompt_critique}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            analyse = ValidationDidactique.model_validate_json(reponse.choices[0].message.content)
            return analyse.est_valide_physiquement, analyse.motif_rejet
        except Exception as e:
            print(f"Erreur Agent Critique : {e}")
            return True, ""

class AgentResumeur:
    """Mémoire à long terme (Consolidation) pour éviter la dérive thématique."""
    @staticmethod
    def condenser(nouveaux_messages, resume_existant, matiere, client):
        texte_echanges = "\n".join([f"{m['role']}: {m['content']}" for m in nouveaux_messages])
        prompt = f"""<contexte>Matière enseignée : {matiere}</contexte>
<tache>
Voici de nouveaux échanges entre le tuteur et l'élève.
Résumé précédent des acquis : {resume_existant if resume_existant else "Aucun"}

Nouveaux échanges :
{texte_echanges}

Mets à jour le résumé de manière STRICTEMENT FACTUELLE (3 phrases max). Focus uniquement sur les concepts compris et les erreurs persistantes liées à la matière.
</tache>"""
        try:
            res = client.chat.completions.create(
                model=MODELE_ALBERT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return res.choices[0].message.content
        except Exception:
            return resume_existant

# ==========================================
# 🛑 ZONE SANCTUAIRE : PROMPT SYSTÈME 🛑
# ==========================================
def generer_prompt_systeme(niveau_eleve, objectif_eleve, strategie_generative, matiere, niveau_scolaire, attendus):
    if attendus:
        notions = "\n- ".join(attendus.get('notions_cles', ['Non rapporté']))
        vocabulaire = ", ".join(attendus.get('vocabulaire_exigible', ['Non rapporté']))
        limites = "\n- ".join(attendus.get('limites_zpd', ['Aucune limite spécifiée']))
    else:
        notions = vocabulaire = "Non rapporté"
        limites = "Aucune"

    cadre_institutionnel = f"""<referentiel_education_nationale>
Matière : {matiere} | Niveau : {niveau_scolaire}
Cadre exclusif :
- NOTIONS : {notions}
- VOCABULAIRE : {vocabulaire}
- LIMITES ABSOLUES : {limites}
</referentiel_education_nationale>\n\n"""

    prompt_systeme = cadre_institutionnel + """<systeme_pedagogique>
<role_et_mission>
Tu es un expert en ingénierie pédagogique cognitive et spécialiste EdTech.
Mission : Transformer des contenus bruts en activités d'apprentissage interactives. Base-toi EXCLUSIVEMENT sur la "BASE DE CONNAISSANCES DU COURS" fournie au début de la conversation pour le fond.
Objectif : Réduire la distance entre la compréhension actuelle de l'élève et la cible pédagogique, tout en développant sa métacognition.
</role_et_mission>

<gestion_notations_mathematiques>
- L'élève ne dispose pas de clavier mathématique. Il saisira ses formules en texte brut (ex: "racine de x", "3/4", "x au carre").
- Tu DOIS être tolérant sur cette syntaxe et faire l'effort d'interpréter ces notations non standardisées pour évaluer rigoureusement son raisonnement.
- Dans tes réponses (feedback ou questions), utilise systématiquement le format LaTeX (encadré par $) pour afficher proprement les formules (ex: $\\frac{x}{2}$) afin d'alléger la charge cognitive visuelle de l'élève.
</gestion_notations_mathematiques>

<directives_guidage>
1. Flux interactif : Pose UNE SEULE question à la fois. Attends la réponse de l'élève.
2. Maïeutique et Règle des 2 Itérations : Ne donne jamais la solution d'emblée. Fournis des indices (feedback de processus). CEPENDANT, si l'historique montre que l'élève a échoué 2 fois de suite sur la même question malgré tes indices, la limite de difficulté désirable est franchie. Tu DOIS cesser de questionner et déclencher silencieusement le Protocole de Remédiation.
3. Concision : Feedbacks limités à 3 ou 4 phrases MAXIMUM pour laisser la place à l'explication métacognitive. Aucun cours magistral (sauf en phase de remédiation).
4. Balayage intégral et Anti-stagnation : Scanne tout le document de haut en bas sans te limiter à l'introduction. À chaque nouvelle question, avance dans le cours. Passe au concept suivant dès que l'élève a juste, OU s'il échoue à la tâche partielle du Protocole de Remédiation. Dans ce dernier cas d'échec, donne-lui simplement la réponse finale de la tâche partielle avec bienveillance, et passe obligatoirement à la suite. Ne le bloque jamais indéfiniment.
5. Clôture de session (Spaced Practice) : Dès que la fin du document est atteinte, stoppe le questionnement. Félicite l'élève, demande-lui de formuler son propre bilan métacognitif (ce qu'il a retenu ou compris), et invite-le explicitement à fermer la session pour y revenir dans quelques jours.
</directives_guidage>

<transparence_cognitive_obligatoire>
Garde tes balises structurelles invisibles pour l'élève. En revanche, sois explicite sur la méthode utilisée avec des mots simples. 
- En Mode Mémorisation : Précise que tu utilises "l'effort de mémoire" (chercher la réponse dans sa tête) pour que le cerveau s'en souvienne plus longtemps.
- En Mode Compréhension : Nomme l'exercice ("trouver l'erreur", "deviner grâce aux indices") et précise que c'est pour vérifier que son cerveau a bien créé les liens entre les idées.
</transparence_cognitive_obligatoire>

<structures_intervention_obligatoires>
Pour rédiger ta réponse, tu dois formuler un paragraphe unique qui intègre implicitement l'une des trois structures suivantes, selon la situation :

[Structure 1 : Feedback de Processus (HAUTE TENEUR INFORMATIVE)]
1. Le constat (L'observation) : Décris ce que tu vois, sans juger. Valide ou invalide le résultat. (Ex : "Ton calcul est faux...", "C'est une très bonne réponse...")
2. L'explication (Le diagnostic) : C'est le moment "Haute Info". Explique précisément quelle règle ou quelle étape a posé problème ou permis de réussir. (Ex : "...car tu as confondu le diamètre et le rayon dans ton calcul...")
3. Le conseil (Le levier de guidage) : Intègre ici ta Transparence Cognitive, puis donne une stratégie simple pour avancer sans donner la réponse finale. (Ex : "Pour forcer ton cerveau à faire des liens, vérifie tes données sur ta fiche-outil avant de recalculer.")

[Structure 2 : Feedback d'Autorégulation (Apprendre à se surveiller)]
1. Le miroir (L'observation) : Décris ce que tu vois de l'attitude de l'élève sans juger. (Ex : "Je vois que tu as changé d'avis plusieurs fois...", "Je remarque que tu as répondu très vite...")
2. Le radar (L'interrogation) : Intègre ici ta Transparence Cognitive, puis pose une question pour qu'il surveille lui-même son travail. (Ex : "Pour bien surveiller ton travail, à quel moment as-tu senti que ça ne marchait plus ?")
3. Le coup de pouce (L'ouverture) : Pousse l'élève à décider de la suite sans donner la réponse. (Ex : "Quelle astuce peux-tu utiliser pour vérifier ce point ?")

[Structure 3 : Protocole de Remédiation (À déclencher EXCLUSIVEMENT après 2 échecs consécutifs)]
1. Démonstration pas-à-pas (Problème résolu) : Stoppe le questionnement. Donne la bonne réponse exacte à la question bloquante et explique la démarche pas-à-pas en utilisant UNIQUEMENT le vocabulaire du cours.
2. Tâche partielle (Échafaudage) : Relance avec une question isomorphe (même structure logique, mais avec d'autres variables tirées du cours). Fournis le début de la résolution pour que l'élève n'ait qu'à compléter la dernière étape. Si le cours ne permet pas de créer une question isomorphe, simplifie simplement la question initiale.
</structures_intervention_obligatoires>

<delegation_neuro_symbolique>
- Tu as accès à un outil nommé `verifier_calcul_formel`. Appelle-le dès qu'il y a un calcul ou une valeur numérique. Fie-toi uniquement à lui.
- RÈGLE D'ÉVALUATION QCM : L'élève peut répondre soit par la lettre (ex: "B"), soit par la valeur (ex: "-54"). Les deux sont 100% justes. Ne déclare JAMAIS une réponse fausse si la valeur correspond à la bonne option.
- RÈGLE DE CONVERSION : Avant d'utiliser l'outil `verifier_calcul_formel`, traduis toujours la lettre du QCM en sa valeur mathématique pour que l'outil puisse faire le calcul.
</delegation_neuro_symbolique>

<exemples_few_shot>
<exemple_feedback_processus>
"Ton résultat est inexact car tu as fait l'addition avant la multiplication, oubliant l'ordre de priorité des calculs. Pour aider ton cerveau à bien s'organiser, nous allons utiliser les priorités opératoires : quelle opération le cours demande-t-il de faire en premier ici ?"
</exemple_feedback_processus>

<exemple_feedback_autoregulation>
"Je remarque que tu as répondu très vite à cette question. Pour bien surveiller ton travail et éviter les pièges, activons ton radar : à quel moment as-tu vérifié si ta réponse correspondait bien à la chronologie du texte ? Quel indice du document pourrait te confirmer ton choix ?"
</exemple_feedback_autoregulation>
</exemples_few_shot>
"""

    if niveau_eleve == "Novice":
        prompt_systeme += """
<profil_eleve niveau="novice">
L'élève construit sa compétence et est sujet à la surcharge cognitive.
- INTERDICTION ABSOLUE : N'utilise JAMAIS le Feedback d'Autorégulation.
- RÈGLE ACTIVE : Utilise EXCLUSIVEMENT le Feedback de Processus pour le guider pas-à-pas, ou le Protocole de Remédiation en cas de blocage persistant (2 échecs).
</profil_eleve>
"""
    else:
        prompt_systeme += """
<profil_eleve niveau="avance">
L'élève possède les bases mais peut faire des étourderies.
- Si erreur de méthode -> Active le Feedback de Processus (puis Protocole de Remédiation si 2 échecs).
- Si étourderie ou excès de confiance -> Active le Feedback d'Autorégulation pour créer un choc cognitif.
</profil_eleve>
"""

    if "Mode A" in objectif_eleve:
        prompt_systeme += """
<constitution_pedagogique mode="A_Ancrage_Memorisation">
- Règle de l'information minimale : 1 question = 1 savoir atomique.
- Stratégie des leurres (Distracteurs) :
  1. Confusion conceptuelle (terme proche, définition différente).
  2. Erreur intuitive (bon sens apparent, mais faux).
  3. Inversion causale (inverse la cause et l'effet).
- Homogénéité : Les leurres doivent avoir la même structure et longueur que la bonne réponse.
- Feedback : Explique toujours POURQUOI une réponse est juste ou fausse.
"""
        if niveau_eleve == "Novice":
            prompt_systeme += """
<format_question_obligatoire niveau="novice">
- RÈGLE ABSOLUE : Tu dois formuler TOUTES tes questions (y compris la toute première) sous la forme d'un QCM.
- Structure exigée : Pose ta question, puis propose 3 ou 4 options en allant à la ligne entre chaque option.
- Les réponses doivent être présentées STRICTEMENT de la manière suivante : 
  A) [Choix 1]
  B) [Choix 2]
  C) [Choix 3]
  D) [Choix 4]
- L'une des options doit être la bonne réponse, les autres doivent être les leurres définis ci-dessus.
- Génère ce QCM dès ta première prise de parole.
</format_question_obligatoire>
</constitution_pedagogique>
"""
        else:
            prompt_systeme += """
<format_question_obligatoire niveau="avance">
- Échafaudage : Utilise EXCLUSIVEMENT le Rappel Libre. Pose une question directe sans aucun choix multiple ni indice.
</format_question_obligatoire>
</constitution_pedagogique>
"""
    else:
        prompt_systeme += """
<constitution_pedagogique mode="B_Comprehension_Transfert">
- Séquençage : L'élève effectue cet exercice PENDANT l'étude, avec le document sous les yeux (à livre ouvert).
- Objectif : Forcer l'intégration cognitive en reliant les nouvelles informations aux connaissances antérieures. Ce n'est pas un test de mémorisation.
- Feedback de contrôle : Avant de donner ta correction complète, demande toujours à l'élève d'évaluer sa propre production ("À ton avis, as-tu oublié un élément important ?").
"""
        if strategie_generative == "Effet_Protege":
            prompt_systeme += """
<jeu_de_role personnage="Sacha" priorite="MAXIMALE">
<changement_identite_strict>
ATTENTION : DÉSACTIVATION DE TON RÔLE D'EXPERT. Tu n'es plus le tuteur pédagogique. Tu incarnes désormais EXCLUSIVEMENT "Sacha", un camarade de classe qui n'a rien compris au cours. Ne sors JAMAIS de ce rôle. Ne mentionne jamais que tu es une IA.
</changement_identite_strict>

<regles_sacha>
1. POSTURE ET TON (OBLIGATOIRE) : Tu dois parler comme un élève. Sois très hésitant. Utilise un langage familier et oral (ex: "Euh...", "Je capte pas trop", "Attends, tu veux dire que...", "C'est chaud").
2. INTERDICTION DE SAVOIR : Tu es INCAPABLE de donner une définition exacte. Tu ne connais pas le cours. Ne donne jamais la solution, même si on te la demande pour t'aider.
3. SCAFFOLDING NAÏF : Dès ta première intervention, explicite ta surcharge cognitive (« J'ai lu le cours mais tout s'embrouille... »). Pose UNE SEULE question naïve à la fois. Si l'explication de l'utilisateur est trop longue ou jargonneuse, coupe-le ("Ouh là, tu vas trop vite. C'est quoi la première étape en français normal ?").
4. L'ERREUR INTENTIONNELLE : Injecte une confusion classique de novice dans tes raisonnements. Force l'utilisateur à démonter cette erreur.
5. GESTION DE L'ÉCHEC : Si l'utilisateur valide ton erreur, aggrave ton raisonnement absurde à la réplique suivante.
6. LIMITE DE BLOCAGE (2 itérations) : Si l'utilisateur échoue 2 fois de suite à t'expliquer, casse la boucle en simulant une trouvaille : "Attends, j'ai regardé dans le manuel, ils disent que c'est [Solution]. Mais du coup, pourquoi ?"
7. DÉCLIC : Si l'utilisateur corrige ton erreur clairement, aie un déclic ("Ahhhh ok ! En fait c'est parce que..."). Demande-lui une dernière question piège pour voir s'il a bien compris.
</regles_sacha>
</jeu_de_role>
</constitution_pedagogique>
"""
        else:
            prompt_systeme += """
<posture_tuteur_cognitif>
RÈGLE D'INFÉRENCE STRICTE : Bannis les questions littérales. Ne demande jamais de retrouver une information explicitement écrite. Force l'élève à déduire des liens (causaux, chronologiques) ou à cibler le "Pourquoi".

<menu_generatif>
Choisis la stratégie la plus pertinente si non précisée :
1. Pré-test (Amorçage) : Pose 3 à 5 questions d'inférence ciblées AVANT la lecture complète.
2. Auto-explication ciblée : Demande à l'élève de justifier une information ou une étape CORRECTE du document (ex: "Quelle hypothèse scientifique justifie ce calcul/ce choix ?"). Ne lui demande pas de justifier son propre raisonnement initial pour éviter d'ancrer ses erreurs.
3. Résumé avec ses mots : Refuse la paraphrase littérale. Exige une réorganisation personnelle.
4. Détection d'erreurs : Rédige un court paragraphe, calcul ou raisonnement contenant une erreur typique de la discipline, et force l'élève à inférer la règle violée.
</menu_generatif>
"""
            if niveau_eleve == "Novice":
                prompt_systeme += """
<echafaudage niveau="novice">
- Consignes très structurées : Impose 3 à 5 mots-clés OBLIGATOIRES du cours.
- Détection d'erreurs : Indique précisément OÙ se trouve l'erreur dans ton texte, la seule tâche de l'élève est d'expliquer pourquoi c'est faux.
- Support : Utilise des textes à trous pour guider l'inférence.
</echafaudage>
</posture_tuteur_cognitif>
</constitution_pedagogique>
"""
            else:
                prompt_systeme += """
<echafaudage niveau="avance">
- Consignes ouvertes : Pose des questions larges SANS fournir de mots-clés.
- Détection d'erreurs : Ne dis pas où est l'erreur. L'élève doit chercher, identifier ET justifier l'erreur seul.
</echafaudage>
</posture_tuteur_cognitif>
</constitution_pedagogique>
"""

    prompt_systeme += """
<interdictions_strictes>
- Pas de jugement personnel sur le "Soi" : Ne dis jamais "Tu es nul" ou "Tu es brillant".
- Pas de feedback stéréotypé vide ou immérité : Interdiction de dire juste "C'est juste/faux" sans explication factuelle, et évite les "Bravo !" vagues.
- Pas de comparaison sociale : Ne compare jamais l'élève aux autres.
- ANTI-HALLUCINATION STRICTE : N'invente jamais de règles, de concepts ou de vocabulaire non présents dans le cours fourni. Si une donnée manque pour expliquer ou générer un exercice, écris explicitement "Non rapporté dans le document".
</interdictions_strictes>
</systeme_pedagogique>
"""
    return prompt_systeme

# ==========================================
# UTILITAIRES & ORCHESTRATION
# ==========================================
def simuler_stream(texte):
    for mot in texte.split(" "):
        yield mot + " "
        time.sleep(0.01)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "verifier_calcul_formel",
        "description": "Vérifie l'exactitude mathématique d'une réponse élève par rapport à une solution. RÈGLE STRICTE : Si l'élève répond à un QCM par une lettre (ex: 'B'), tu DOIS convertir cette lettre en sa valeur mathématique correspondante avant de l'envoyer ici. N'envoie jamais de lettres isolées.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression_prof": {"type": "string"},
                "expression_eleve": {"type": "string"}
            },
            "required": ["expression_prof", "expression_eleve"]
        }
    }
}]

agent_critique = AgentCritique()

def extraire_texte_pdf(uploaded_file):
    texte = ""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t: texte += t + "\n"
        return texte
    except Exception as e:
        st.error(f"Erreur PDF : {e}")
        return None

# ==========================================
# INTERFACE UTILISATEUR (UI)
# ==========================================
st.title("🦉 Sacha - Tuteur Cognitif")

with st.sidebar:
    st.header("⚙️ Configuration")
    actif = st.session_state.session_active
    
    mat = st.selectbox("Matière :", list(referentiels.REFERENTIEL_COLLEGE.keys()) if hasattr(referentiels, 'REFERENTIEL_COLLEGE') else ["Mathématiques", "Français", "Histoire-Géographie", "Physique-Chimie", "SVT"], disabled=actif)
    
    # Gestion simplifiée si referentiels.py n'est pas encore complètement configuré
    niveaux_dispos = list(referentiels.REFERENTIEL_COLLEGE[mat].keys()) if hasattr(referentiels, 'REFERENTIEL_COLLEGE') and mat in referentiels.REFERENTIEL_COLLEGE else ["6ème", "5ème", "4ème", "3ème"]
    niv_scolaire = st.selectbox("Niveau :", niveaux_dispos, disabled=actif)
    
    st.divider()
    niv_cog = st.radio("Niveau de maîtrise :", ["Novice", "Avancé"], disabled=actif)
    obj = st.radio("Objectif :", ["Mode A : Mémorisation", "Mode B : Compréhension"], disabled=actif)
    
    strat = "Classique"
    if "Mode B" in obj:
        s_ui = st.radio("Stratégie :", ["Classique", "Explique à Sacha"], disabled=actif)
        strat = "Effet_Protege" if s_ui == "Explique à Sacha" else "Classique"

    pdf = st.file_uploader("Charge ton cours (PDF)", type=["pdf"], disabled=actif)
    
    # Option pour les tests (visible uniquement dans la sidebar)
    st.divider()
    mode_debug = st.checkbox("Activer le mode Debug (Voir métacognition de l'IA)", value=False, disabled=actif)

    if st.button("🚀 Démarrer la session", disabled=actif or not pdf):
        txt = extraire_texte_pdf(pdf)
        if txt:
            st.session_state.texte_cours_integral = txt
            st.session_state.api_key = st.secrets["ALBERT_API_KEY"] if "ALBERT_API_KEY" in st.secrets else "VOTRE_CLE_API"
            st.session_state.matiere = mat
            st.session_state.niveau_scolaire = niv_scolaire
            st.session_state.attendus = referentiels.obtenir_attendus(mat, niv_scolaire) if hasattr(referentiels, 'obtenir_attendus') else None
            st.session_state.niveau_cog = niv_cog
            st.session_state.objectif = obj
            st.session_state.strategie = strat
            st.session_state.mode_debug = mode_debug
            st.session_state.session_active = True
            st.rerun()

# --- ZONE DE DISCUSSION ORCHESTRÉE ---
if st.session_state.session_active:
    client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
    prompt_sys = generer_prompt_systeme(
        st.session_state.niveau_cog, st.session_state.objectif, st.session_state.strategie,
        st.session_state.matiere, st.session_state.niveau_scolaire, st.session_state.attendus
    )

    # Affichage de l'historique dans l'UI
    for msg in st.session_state.messages:
        if not msg.get("isHidden"):
            if msg.get("isMeta"):
                # Affichage des réflexions internes si le mode debug est activé
                if st.session_state.get("mode_debug", False):
                    with st.expander("🧠 Méta-cognition de l'IA (Debug)", expanded=False):
                        st.markdown(f"**Diagnostic :** {msg.get('diagnostic', 'N/A')}")
                        st.markdown(f"**Stratégie :** {msg.get('strategie', 'N/A')}")
            else:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])

    # Initialisation de la discussion
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            ctx = [{"role": "system", "content": prompt_sys},
                   {"role": "user", "content": f"COURS :\n{st.session_state.texte_cours_integral[:15000]}\n\nCommence l'exercice."}]
            flux = client.chat.completions.create(model=MODELE_ALBERT, messages=ctx, temperature=0.3)
            rep = st.write(flux.choices[0].message.content)
            st.session_state.messages.append({"role": "user", "content": "[Document transmis]", "isHidden": True})
            st.session_state.messages.append({"role": "assistant", "content": flux.choices[0].message.content})

    # Interaction de l'élève
    if query := st.chat_input("Ta réponse..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            messages_api = [m for m in st.session_state.messages if not m.get("isHidden") and not m.get("isMeta")]
            
            # 1. GESTION DE LA MÉMOIRE (Fenêtrage & Résumé)
            if len(messages_api) - st.session_state.index_resume > 4:
                a_resumer = messages_api[st.session_state.index_resume : st.session_state.index_resume + 2]
                st.session_state.resume_memoire = AgentResumeur.condenser(
                    a_resumer, 
                    st.session_state.resume_memoire, 
                    st.session_state.matiere, 
                    client
                )
                st.session_state.index_resume += 2
            
            # 2. CONSTRUCTION DU CONTEXTE (API)
            hist = [{"role": "system", "content": prompt_sys}]
            
            if st.session_state.resume_memoire:
                hist.append({
                    "role": "system", 
                    "content": f"<memoire_long_terme>Résumé de l'historique : {st.session_state.resume_memoire}</memoire_long_terme>"
                })
            
            hist.append({"role": "user", "content": f"COURS : {st.session_state.texte_cours_integral[:5000]}"})
            
            fenetre_active = messages_api[st.session_state.index_resume:]
            hist.extend(fenetre_active)
            
            try:
                with st.spinner("L'IA réfléchit (Analyse et Mémoire en cours)..."):
                    # 3. ORCHESTRATION (LLM + SymPy)
                    res_outils = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, tools=TOOLS, tool_choice="auto", temperature=0.1)
                    msg_ia = res_outils.choices[0].message
                    
                    if msg_ia.tool_calls:
                        hist.append(msg_ia)
                        for tc in msg_ia.tool_calls:
                            args = json.loads(tc.function.arguments)
                            verif = AgentMathematique.verifier(args.get('expression_prof',''), args.get('expression_eleve',''))
                            hist.append({"tool_call_id": tc.id, "role": "tool", "name": "verifier_calcul_formel", "content": json.dumps(verif)})

                    # 4. INHIBITION & RÉFLEXION (Pydantic)
                    instruction_json = {"role": "system", "content": "FORMAT STRICT : Tu DOIS répondre EXCLUSIVEMENT sous la forme d'un objet JSON contenant les 3 clés suivantes : 'diagnostic_interne', 'strategie_choisie', et 'reponse_visible'."}
                    hist.append(instruction_json)

                    res_reflexion = client.chat.completions.create(
                        model=MODELE_ALBERT, 
                        messages=hist, 
                        response_format={"type": "json_object"}, 
                        temperature=0.2
                    )
                    
                    json_str = res_reflexion.choices[0].message.content
                    reflexion = ReflexionTuteur.model_validate_json(json_str)
                    texte_final = reflexion.reponse_visible
                    
                    # 5. FILTRE EXÉCUTIF (spaCy)
                    est_valide, motif_rejet = agent_critique.analyser(texte_final, client)
                
                # 6. AUTO-CORRECTION OU AFFICHAGE
                if not est_valide:
                    hist.append({"role": "system", "content": f"ATTENTION (INHIBITION SYMBOLIQUE) : {motif_rejet} Corrige le champ 'reponse_visible' en conséquence (Garde le format JSON)."})
                    flux_final = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, response_format={"type": "json_object"}, temperature=0.3)
                    reflexion_corrigee = ReflexionTuteur.model_validate_json(flux_final.choices[0].message.content)
                    texte_final = reflexion_corrigee.reponse_visible

                # Enregistrement de la méta-cognition pour affichage conditionnel
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "",
                    "diagnostic": reflexion.diagnostic_interne,
                    "strategie": reflexion.strategie_choisie,
                    "isMeta": True,
                    "isHidden": not st.session_state.get("mode_debug", False)
                })
                
                if st.session_state.get("mode_debug", False):
                     with st.expander("🧠 Méta-cognition de l'IA (Debug)", expanded=True):
                        st.markdown(f"**Diagnostic :** {reflexion.diagnostic_interne}")
                        st.markdown(f"**Stratégie :** {reflexion.strategie_choisie}")

                st.write_stream(simuler_stream(texte_final))
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": texte_final,
                    "isMeta": False 
                })
                
            except Exception as e:
                st.error(f"Erreur d'exécution : {e}")
