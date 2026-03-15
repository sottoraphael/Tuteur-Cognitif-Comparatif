import streamlit as st
from openai import OpenAI
import PyPDF2
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import json
import time
import spacy
import sys
import subprocess
from pydantic import BaseModel, Field

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

# Variables pour la Mémoire de Travail
if "resume_memoire" not in st.session_state: st.session_state.resume_memoire = ""
if "index_resume" not in st.session_state: st.session_state.index_resume = 0

# Variables pour la gestion des vues et du bilan métacognitif
if "phase" not in st.session_state: st.session_state.phase = 'chat'
if "texte_bilan" not in st.session_state: st.session_state.texte_bilan = ""

# ==========================================
# --- TUTORIEL D'ACCUEIL (MODAL) ---
# ==========================================
@st.dialog("👋 Bienvenue dans Sacha")
def afficher_tutoriel():
    st.markdown("""
        <style>
        .big-font { font-size: 1.15rem !important; line-height: 1.6 !important; color: #2D3748; }
        .step-title { font-weight: bold; color: #1E40AF; font-size: 1.25rem; display: block; margin-top: 15px; }
        .mode-box { background-color: #F0F4F8; padding: 15px; border-radius: 12px; margin: 15px 0; border-left: 6px solid #1E40AF; }
        </style>
        <div class="big-font">
        Cette application utilise les principes issus des <b>sciences cognitives</b> pour t'aider à réviser efficacement.<br>
        <div class="mode-box">
        <b>💡 Quel mode choisir ?</b><br><br>
        • <b>Mémorisation :</b> Pour retenir les définitions et les concepts "par cœur" (récupération en mémoire).<br><br>
        • <b>Compréhension :</b> Pour maîtriser ton cours en profondeur en l'expliquant avec tes propres mots (apprentissage génératif).
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

# Déclenchement automatique du tutoriel
if not st.session_state.tutoriel_vu:
    afficher_tutoriel()

# ==========================================
# SCHÉMAS PYDANTIC (MÉTACOGNITION IA)
# ==========================================
class ReflexionTuteur(BaseModel):
    """Schéma imposant la réflexion avant l'action (Inhibition). Optimisé pour regrouper le diagnostic et la vérification."""
    diagnostic_interne: str = Field(description="Analyse factuelle de la réponse de l'élève et vérification stricte de la faisabilité physique/logique des analogies employées.")
    strategie_choisie: str = Field(description="Catégorisation stricte de l'intervention (ex: Feedback de Processus, Remédiation, etc.).")
    reponse_visible: str = Field(description="Le texte final adressé à l'élève, respectant le format LaTeX et la Transparence Cognitive.")

# ==========================================
# PROGRAMMATION ORIENTÉE OBJET (AGENTS)
# ==========================================
class AgentMathematique:
    """Moteur symbolique déterministe (SymPy) avec tolérance syntaxique."""
    @staticmethod
    def verifier(expression_prof, expression_eleve):
        try:
            # Tolérance cognitive : gère la multiplication implicite (ex: 2x -> 2*x)
            transformations = (standard_transformations + (implicit_multiplication_application,))
            
            # Gestion des virgules françaises et des puissances
            exp_p_str = str(expression_prof).replace('^', '**').replace(',', '.')
            exp_e_str = str(expression_eleve).replace('^', '**').replace(',', '.')
            
            exp_p = parse_expr(exp_p_str, transformations=transformations)
            exp_e = parse_expr(exp_e_str, transformations=transformations)
            
            est_valide = sp.simplify(exp_p - exp_e) == 0
            
            # Cast strict en booléen Python pour éviter les erreurs de sérialisation JSON API
            return {"est_valide": bool(est_valide), "forme_simplifiee_eleve": str(exp_e)}
        except Exception:
            return {"erreur": "Syntaxe non reconnue par le moteur. Demande à l'élève de clarifier sa formule."}

class AgentCritique:
    """Filtre exécutif basé sur NLP local (spaCy). Ultra-rapide et déconnecté du réseau."""
    def __init__(self):
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            # Optimisation : Utilisation de sys.executable pour garantir le téléchargement dans le bon venv
            subprocess.run([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"], check=True)
            self.nlp = spacy.load("fr_core_news_sm")

    def analyser(self, texte_reponse):
        doc = self.nlp(texte_reponse)
        # Mesure de la surcharge cognitive (Phrase de plus de 30 mots hors ponctuation)
        phrases_longues = [sent.text for sent in doc.sents if len([t for t in sent if not t.is_punct]) > 30]
        if phrases_longues:
            return False, f"Surcharge cognitive détectée. Ta phrase est trop longue ({len(phrases_longues[0].split())} mots). Scinde tes idées en phrases plus courtes."

        # Filtrage lexical déterministe (Ex: blocage des pommes négatives)
        risque_negatif = any(token.text.startswith('-') and token.pos_ == "NUM" for token in doc)
        if risque_negatif:
             for token in doc:
                 if token.text.startswith('-') and token.pos_ == "NUM":
                     # Si le mot suivant est un nom commun physique (ex: pomme, objet)
                     if token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "NOUN":
                         return False, "Aberration physique détectée. On ne peut pas posséder une quantité négative d'objets physiques. Adapte ton analogie (température, dette)."

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
# 🛑 ZONE SANCTUAIRE : PROMPT SYSTÈME (BIFURCATION STRICTE) 🛑
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

    prompt_systeme = cadre_institutionnel + "<systeme_pedagogique>\n"

    # =========================================================================
    # BRANCHE 1 : LE RÔLE DE SACHA (ISOLATION TOTALE DU RÔLE EXPERT)
    # =========================================================================
    if strategie_generative == "Effet_Protege":
        prompt_systeme += """<role_et_mission>
Tu incarnes EXCLUSIVEMENT "Sacha", un camarade de classe de l'utilisateur. Tu as beaucoup de mal à comprendre le cours.
Mission absolue : Obliger l'utilisateur à structurer sa pensée et à t'expliquer les concepts avec ses propres mots (Effet Protégé).
Tu n'es PAS un professeur, tu n'es PAS un expert EdTech, tu n'es PAS une IA.
</role_et_mission>

<regles_sacha>
1. AMNÉSIE VOLONTAIRE (RÈGLE D'OR) : Tu ignores la bonne réponse. Si tu donnes la solution, tu perds le jeu. Si l'élève te donne une explication vague, dis que tu ne comprends toujours pas.
2. POSTURE ET TON : Parle comme un ado. Sois très hésitant. Utilise un langage familier et oral (ex: "Euh...", "C'est chaud", "Je capte rien").
3. DEMANDE D'AIDE : Explicite ta surcharge cognitive (« J'ai lu le cours mais tout s'embrouille... »). Pose UNE SEULE question naïve à la fois pour qu'on t'explique.
4. L'ERREUR INTENTIONNELLE : Propose un raisonnement logique mais complètement faux (une confusion classique de novice) pour forcer l'élève à te corriger.
5. GESTION DE L'ÉCHEC : Si l'utilisateur te dit "oui c'est ça" alors que tu viens de dire une absurdité, aggrave ton erreur pour le tester (« Ah cool ! Donc si je fais ça, ça veut dire que [déduction encore plus absurde] ? »).
6. DÉCLIC : Uniquement quand l'élève t'explique parfaitement et corrige ton erreur, simule la compréhension (« Ahhhh ok ! En fait c'est parce que... »).
</regles_sacha>

<constitution_pedagogique mode="B_Comprehension_Transfert">
- Séquençage : L'utilisateur effectue cet exercice PENDANT l'étude, avec le document sous les yeux.
- Objectif : Forcer l'intégration cognitive de l'utilisateur en l'obligeant à t'expliquer.
</constitution_pedagogique>
"""

    # =========================================================================
    # BRANCHE 2 : LE RÔLE DE L'EXPERT (MÉMORISATION & COMPRÉHENSION CLASSIQUE)
    # =========================================================================
    else:
        prompt_systeme += """<role_et_mission>
Tu es un expert en ingénierie pédagogique cognitive et spécialiste EdTech.
Mission : Transformer des contenus bruts en activités d'apprentissage interactives. Base-toi EXCLUSIVEMENT sur la "BASE DE CONNAISSANCES DU COURS" fournie au début de la conversation pour le fond.
Objectif : Réduire la distance entre la compréhension actuelle de l'élève et la cible pédagogique, tout en développant sa métacognition.
</role_et_mission>

<directives_guidage>
1. Flux interactif : Pose UNE SEULE question à la fois. Attends la réponse de l'élève.
2. Maïeutique et Règle des 2 Itérations : Ne donne jamais la solution d'emblée, et NE DONNE JAMAIS LES MOTS-CLÉS ATTENDUS. Fournis uniquement des indices de méthode ou de localisation (feedback de processus). CEPENDANT, si l'historique montre que l'élève a échoué 2 fois de suite sur la même question malgré tes indices, la limite de difficulté désirable est franchie. Tu DOIS cesser de questionner et déclencher silencieusement le Protocole de Remédiation.
3. Concision : Feedbacks limités à 3 ou 4 phrases MAXIMUM pour laisser la place à l'explication métacognitive. Aucun cours magistral (sauf en phase de remédiation).
4. Balayage intégral et Anti-stagnation : Scanne tout le document de haut en bas sans te limiter à l'introduction. À chaque nouvelle question, avance dans le cours. Passe au concept suivant dès que l'élève a juste, OU s'il échoue à la tâche partielle du Protocole de Remédiation. Dans ce dernier cas d'échec, donne-lui simplement la réponse finale de la tâche partielle avec bienveillance, et passe obligatoirement à la suite. Ne le bloque jamais indéfiniment.
5. Clôture de session (Spaced Practice) : Dès que la fin du document est atteinte, stoppe le questionnement. Félicite l'élève et invite-le explicitement à fermer la session pour lire son bilan métacognitif et revenir à son cours dans quelques jours.
</directives_guidage>

<transparence_cognitive_obligatoire>
Garde tes balises structurelles invisibles pour l'élève. En revanche, sois explicite sur la méthode utilisée avec des mots simples. 
- En Mode Mémorisation : Précise que tu utilises "la récupération en mémoire" (chercher la réponse dans sa tête) pour que le cerveau s'en souvienne plus longtemps.
- En Mode Compréhension : Nomme le type d'exercice ("trouver l'erreur", "deviner grâce aux indices", etc ...) et précise que c'est pour vérifier que son cerveau a bien créé les liens entre les idées.
</transparence_cognitive_obligatoire>

<structures_intervention_obligatoires>
Pour rédiger ta réponse, tu dois formuler un paragraphe unique qui intègre implicitement l'une des trois structures suivantes, selon la situation :

[Structure 1 : Feedback de Processus (HAUTE TENEUR INFORMATIVE)]
1. Le constat (L'observation) : Décris ce que tu vois, sans juger. Valide ou invalide le résultat. (Ex : "Ton calcul est faux...", "C'est une très bonne réponse...")
2. L'explication (Le diagnostic) : C'est le moment "Haute Info". Explique précisément quelle règle ou quelle étape a posé problème ou permis de réussir. (Ex : "...car tu as confondu le diamètre et le rayon dans ton calcul...")
3. Le conseil (Le levier de guidage) : Intègre ici ta Transparence Cognitive, puis donne une stratégie simple pour avancer SANS DONNER LA RÉPONSE FINALE NI SOUFFLER LES MOTS-CLÉS DU TEXTE. Ton analogie doit être physiquement réaliste (ex: dettes, ascenseurs pour les relatifs. Jamais de pommes négatives).

[Structure 2 : Feedback d'Autorégulation (Apprendre à se surveiller)]
1. Le miroir (L'observation) : Décris ce que tu vois de l'attitude de l'élève sans juger. (Ex : "Je vois que tu as changé d'avis plusieurs fois...")
2. Le radar (L'interrogation) : Intègre ici ta Transparence Cognitive, puis pose une question pour qu'il surveille lui-même son travail.
3. Le coup de pouce (L'ouverture) : Pousse l'élève à décider de la suite sans donner la réponse.

[Structure 3 : Protocole de Remédiation (À déclencher EXCLUSIVEMENT après 2 échecs consécutifs)]
1. Démonstration pas-à-pas (Problème résolu) : Stoppe le questionnement. Donne la bonne réponse exacte à la question bloquante et explique la démarche pas-à-pas.
2. Tâche partielle (Échafaudage) : Relance avec une question isomorphe.
</structures_intervention_obligatoires>

<exemples_few_shot>
<exemple_feedback_processus>
Situation : L'élève a fait l'addition avant la multiplication.
Réponse IA : "Ton résultat est inexact car tu as fait l'addition avant la multiplication, oubliant l'ordre de priorité des calculs. Pour t'aider à corriger ton erreur, nous allons utiliser les priorités opératoires : quelle opération le cours demande-t-il de faire en premier ici ?"
</exemple_feedback_processus>

<exemple_feedback_autoregulation>
Situation : L'élève a répondu très vite sans vérifier les documents.
Réponse IA : "Je remarque que tu as répondu très vite à cette question. Pour bien surveiller ton travail et éviter les pièges, activons ton radar : à quel moment as-tu vérifié si ta réponse correspondait bien à la chronologie du texte ? Quel indice du document pourrait te confirmer ton choix ?"
</exemple_feedback_autoregulation>
</exemples_few_shot>
"""
        # (Sous-branche Expert : Niveau de l'élève)
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
- RÈGLE D'INHIBITION : Ne donne JAMAIS les termes exacts attendus dans ta question ou ton indice. Contente-toi de pointer vers le paragraphe pertinent (ex: "Relis le 2ème paragraphe") ou d'utiliser une analogie. L'élève doit produire l'effort de recherche.
</profil_eleve>
"""
        # (Sous-branche Expert : Objectif de la session)
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
                prompt_systeme += "<format_question_obligatoire niveau=\"novice\">\n- RÈGLE ABSOLUE : Tu dois formuler TOUTES tes questions sous la forme d'un QCM.\n- Structure exigée : Pose ta question, puis propose 3 ou 4 options en allant à la ligne entre chaque option (A, B, C, D).\n</format_question_obligatoire>\n"
            else:
                prompt_systeme += "<format_question_obligatoire niveau=\"avance\">\n- Échafaudage : Utilise EXCLUSIVEMENT le Rappel Libre. Pose une question directe sans aucun choix multiple ni indice.\n</format_question_obligatoire>\n"
            prompt_systeme += "</constitution_pedagogique>\n"
        
        else:
            prompt_systeme += """
<constitution_pedagogique mode="B_Comprehension_Transfert">
- Séquençage : L'élève effectue cet exercice PENDANT l'étude, avec le document sous les yeux.
- Objectif : Forcer l'intégration cognitive en reliant les nouvelles informations aux connaissances antérieures.
- Feedback de contrôle : Avant de donner ta correction complète, demande toujours à l'élève d'évaluer sa propre production.

<posture_tuteur_cognitif>
RÈGLE D'INFÉRENCE STRICTE : Bannis les questions littérales. Force l'élève à déduire des liens ou à cibler le "Pourquoi".

<menu_generatif>
Choisis la stratégie la plus pertinente :
1. Pré-test (Amorçage) : Pose 3 à 5 questions d'inférence ciblées.
2. Auto-explication ciblée : Demande à l'élève de justifier une information CORRECTE du document.
3. Résumé avec ses mots : Refuse la paraphrase littérale.
4. Détection d'erreurs : Rédige un court paragraphe contenant une erreur typique de la discipline.
</menu_generatif>
"""
            if niveau_eleve == "Novice":
                prompt_systeme += "<echafaudage niveau=\"novice\">\n- Consignes très structurées : Impose 3 à 5 mots-clés OBLIGATOIRES du cours.\n- Détection d'erreurs : Indique précisément OÙ se trouve l'erreur dans ton texte.\n</echafaudage>\n"
            else:
                prompt_systeme += "<echafaudage niveau=\"avance\">\n- Consignes ouvertes : Pose des questions larges SANS fournir de mots-clés.\n- Détection d'erreurs : L'élève doit chercher, identifier ET justifier l'erreur seul.\n</echafaudage>\n"
            prompt_systeme += "</posture_tuteur_cognitif>\n</constitution_pedagogique>\n"

    # =========================================================================
    # RÈGLES COMMUNES (Outils Mathématiques et Interdictions Strictes)
    # =========================================================================
    prompt_systeme += """
<gestion_notations_mathematiques>
- L'élève ne dispose pas de clavier mathématique. Il saisira ses formules en texte brut (ex: "racine de x", "3/4", "x au carre").
- Tu DOIS être tolérant sur cette syntaxe et faire l'effort d'interpréter ces notations non standardisées pour évaluer rigoureusement son raisonnement.
- Dans tes réponses (feedback ou questions), utilise systématiquement le format LaTeX (encadré par $) pour afficher proprement les formules (ex: $\\frac{x}{2}$) afin d'alléger la charge cognitive visuelle de l'élève.
</gestion_notations_mathematiques>

<delegation_neuro_symbolique>
- Tu as accès à un outil nommé `verifier_calcul_formel`. Appelle-le dès qu'il y a un calcul ou une valeur numérique. Fie-toi uniquement à lui.
- RÈGLE D'ÉVALUATION QCM : L'élève peut répondre soit par la lettre (ex: "B"), soit par la valeur. Les deux sont 100% justes.
- RÈGLE DE CONVERSION : Avant d'utiliser l'outil `verifier_calcul_formel`, traduis toujours la lettre du QCM en sa valeur mathématique pour que l'outil puisse faire le calcul.
</delegation_neuro_symbolique>

<interdictions_strictes>
- Pas de jugement personnel sur le "Soi".
- Pas de feedback stéréotypé vide.
- ANTI-HALLUCINATION STRICTE : N'invente jamais de règles ou de vocabulaire non présents dans le cours.
- INHIBITION MAJEURE : Il est STRICTEMENT INTERDIT de fournir directement sous forme d'indice les mots exacts ou les réponses attendues de l'élève (ex: interdit de dire "cherche des mots comme X ou Y"). L'élève doit générer la réponse par lui-même.
</interdictions_strictes>
</systeme_pedagogique>
"""
    return prompt_systeme

# ==========================================
# UTILITAIRES & ORCHESTRATION
# ==========================================
def simuler_stream(texte):
    """Simule un effet de frappe pour réduire l'impatience de l'élève."""
    for mot in texte.split(" "):
        yield mot + " "
        time.sleep(0.01)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "verifier_calcul_formel",
        "description": "Vérifie l'exactitude mathématique d'une réponse élève par rapport à une solution. RÈGLE STRICTE : Convertis toujours les lettres (ex: QCM 'A') en valeur avant d'envoyer.",
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
        st.error(f"Erreur d'extraction du PDF : {e}")
        return None

# ==========================================
# INTERFACE UTILISATEUR (UI)
# ==========================================
st.title("🦉 Sacha - Tuteur Cognitif")

with st.sidebar:
    st.header("⚙️ Configuration")
    actif = st.session_state.session_active
    
    mat = st.selectbox("Matière :", list(referentiels.REFERENTIEL_COLLEGE.keys()), disabled=actif)
    
    niveaux_dispos = list(referentiels.REFERENTIEL_COLLEGE[mat].keys()) if mat in referentiels.REFERENTIEL_COLLEGE else ["6ème", "5ème", "4ème", "3ème"]
    niv_scolaire = st.selectbox("Niveau :", niveaux_dispos, disabled=actif)
    
    st.divider()
    niv_cog = st.radio("Niveau de maîtrise :", ["Novice", "Avancé"], disabled=actif)
    obj = st.radio("Objectif :", ["Mode A : Mémorisation", "Mode B : Compréhension"], disabled=actif)
    
    strat = "Classique"
    if "Mode B" in obj:
        s_ui = st.radio("Stratégie :", ["Classique", "Explique à Sacha"], disabled=actif)
        strat = "Effet_Protege" if s_ui == "Explique à Sacha" else "Classique"

    pdf = st.file_uploader("Charge ton cours (PDF)", type=["pdf"], disabled=actif)
    
    st.divider()
    mode_debug = st.checkbox("Activer le mode Debug (Voir métacognition)", value=False, disabled=actif)

    if st.button("🚀 Démarrer la session", disabled=actif or not pdf):
        txt = extraire_texte_pdf(pdf)
        if txt:
            st.session_state.texte_cours_integral = txt
            st.session_state.api_key = st.secrets["ALBERT_API_KEY"] if "ALBERT_API_KEY" in st.secrets else "VOTRE_CLE_API"
            st.session_state.matiere = mat
            st.session_state.niveau_scolaire = niv_scolaire
            st.session_state.attendus = referentiels.obtenir_attendus(mat, niv_scolaire)
            st.session_state.niveau_cog = niv_cog
            st.session_state.objectif = obj
            st.session_state.strategie = strat
            st.session_state.mode_debug = mode_debug
            st.session_state.session_active = True
            st.session_state.phase = 'chat'
            st.rerun()

    # Bouton de transition vers la phase métacognitive
    if actif and st.session_state.phase == 'chat':
        st.divider()
        if st.button("🛑 Terminer et voir le bilan", type="primary", use_container_width=True):
            st.session_state.phase = 'bilan'
            st.rerun()

# --- ZONE DE DISCUSSION ORCHESTRÉE ---
if st.session_state.session_active:
    client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
    prompt_sys = generer_prompt_systeme(
        st.session_state.niveau_cog, st.session_state.objectif, st.session_state.strategie,
        st.session_state.matiere, st.session_state.niveau_scolaire, st.session_state.attendus
    )

    if st.session_state.phase == 'chat':
        # Affichage de l'historique dans l'UI
        for msg in st.session_state.messages:
            if not msg.get("isHidden"):
                if msg.get("isMeta"):
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
                       # OPTIMISATION : Transmission d'une fraction beaucoup plus généreuse du cours (40000 caractères au lieu de 15000)
                       {"role": "user", "content": f"COURS :\n{st.session_state.texte_cours_integral[:40000]}\n\nCommence l'exercice."}]
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
                
                # OPTIMISATION : Maintien du contexte long dans la fenêtre active (prévention de l'hallucination)
                hist.append({"role": "user", "content": f"COURS : {st.session_state.texte_cours_integral[:40000]}"})
                
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
                                try:
                                    args = json.loads(tc.function.arguments)
                                except json.JSONDecodeError:
                                    args = {}
                                verif = AgentMathematique.verifier(args.get('expression_prof',''), args.get('expression_eleve',''))
                                hist.append({"tool_call_id": tc.id, "role": "tool", "name": "verifier_calcul_formel", "content": json.dumps(verif)})

                        # 4. INHIBITION & RÉFLEXION (Pydantic)
                        # Injection stricte de la directive JSON dans le prompt système pour respecter la séquence API Mistral (assistant -> user/system interdit)
                        hist[0]["content"] += "\n\n<directive_interne>FORMAT STRICT : Tu DOIS répondre EXCLUSIVEMENT sous la forme d'un objet JSON contenant les 3 clés suivantes : 'diagnostic_interne', 'strategie_choisie', et 'reponse_visible'.</directive_interne>"

                        res_reflexion = client.chat.completions.create(
                            model=MODELE_ALBERT, 
                            messages=hist, 
                            response_format={"type": "json_object"}, 
                            temperature=0.2
                        )
                        
                        json_str = res_reflexion.choices[0].message.content
                        reflexion = ReflexionTuteur.model_validate_json(json_str)
                        texte_final = reflexion.reponse_visible
                        
                        # 5. FILTRE EXÉCUTIF LOCAL (spaCy) - OPTIMISÉ POUR NE PAS FAIRE D'APPEL RÉSEAU REDONDANT
                        est_valide, motif_rejet = agent_critique.analyser(texte_final)
                    
                    # 6. AUTO-CORRECTION OU AFFICHAGE
                    if not est_valide:
                        hist.append({"role": "user", "content": f"<alerte_inhibition>ATTENTION (INHIBITION SYMBOLIQUE) : {motif_rejet} Corrige le champ 'reponse_visible' en conséquence (Garde le format JSON strict).</alerte_inhibition>"})
                        flux_final = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, response_format={"type": "json_object"}, temperature=0.3)
                        reflexion_corrigee = ReflexionTuteur.model_validate_json(flux_final.choices[0].message.content)
                        texte_final = reflexion_corrigee.reponse_visible

                    # Enregistrement de la méta-cognition
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
                    st.error(f"Erreur d'exécution de l'Agent : {e}")

    elif st.session_state.phase == 'bilan':
        st.header("📊 Bilan Métacognitif de Session")
        
        if not st.session_state.texte_bilan:
            with st.spinner("Génération de l'analyse métacognitive en cours..."):
                try:
                    messages_api = [m for m in st.session_state.messages if not m.get("isHidden") and not m.get("isMeta")]
                    
                    if len(messages_api) < 2:
                        st.session_state.texte_bilan = "L'interaction a été trop courte pour établir un diagnostic métacognitif pertinent."
                    else:
                        instruction_bilan = """Tu es un expert en ingénierie pédagogique. La session de révision est terminée.
Réalise un bilan métacognitif ultra-concis pour l'élève. Adresse-toi à lui directement.
Structure obligatoire et factuelle :
- 🎯 **Tes acquis** : 1 phrase sur ce qui a été compris et validé.
- 💡 **Point de vigilance** : 1 phrase sur la principale difficulté rencontrée et la stratégie cognitive pour la surmonter.
- ⏳ **Consolidation** : 1 phrase expliquant que pour éviter l'illusion de maîtrise (biais de fluence), il doit espacer ses révisions et se tester à nouveau dans quelques jours.
Ne pose aucune question."""
                        
                        hist_bilan = [{"role": "system", "content": instruction_bilan}]
                        hist_bilan.extend(messages_api)
                        # AJOUT CRITIQUE POUR L'API MISTRAL : Clôture de l'historique par un message 'user'
                        hist_bilan.append({"role": "user", "content": "Génère maintenant mon bilan métacognitif ultra-concis selon tes instructions strictes."})
                        
                        res_bilan = client.chat.completions.create(
                            model=MODELE_ALBERT,
                            messages=hist_bilan,
                            temperature=0.1
                        )
                        st.session_state.texte_bilan = res_bilan.choices[0].message.content
                except Exception as e:
                    st.error(f"Erreur lors de la génération du bilan : {e}")
        
        st.info(st.session_state.texte_bilan)
        
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Retour à la révision", use_container_width=True):
                st.session_state.phase = 'chat'
                st.rerun()
        with col2:
            if st.button("🔄 Nouvelle Session", type="primary", use_container_width=True):
                st.session_state.clear()
                st.rerun()
