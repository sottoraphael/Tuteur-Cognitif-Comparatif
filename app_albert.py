import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import PyPDF2
import sympy as sp
import json
import time
import spacy
from pydantic import BaseModel, Field, ValidationError

# Fichier local contenant les programmes (ZPD)
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
# CHARGEMENT DU MODÈLE NLP LOCAL (SPACY)
# ==========================================
@st.cache_resource
def charger_modele_nlp():
    """Charge le modèle linguistique français en RAM pour une analyse souveraine."""
    try:
        return spacy.load("fr_core_news_md")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_md"])
        return spacy.load("fr_core_news_md")

nlp = charger_modele_nlp()

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

# ==========================================
# MODULE 1 : CALCUL FORMEL (SYMPY)
# ==========================================
def verifier_calcul_formel(expression_prof, expression_eleve):
    """Moteur déterministe pour certifier l'équivalence mathématique."""
    try:
        exp_p = sp.simplify(str(expression_prof).replace('^', '**'))
        exp_e = sp.simplify(str(expression_eleve).replace('^', '**'))
        est_valide = sp.simplify(exp_p - exp_e) == 0
        return {"est_valide": est_valide, "forme_simplifiee_eleve": str(exp_e)}
    except Exception:
        return {"erreur": "Syntaxe mathématique non reconnue. Demande à l'élève de clarifier."}

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

# ==========================================
# MODULE 2 : AGENT CRITIQUE DIDACTIQUE (PYDANTIC + SPACY)
# ==========================================
class ValidationDidactique(BaseModel):
    contient_analogie: bool = Field(description="La réponse contient-elle une analogie ou un exemple concret ?")
    entites_physiques_detectees: list[str] = Field(description="Liste des objets physiques mentionnés.")
    est_valide_physiquement: bool = Field(description="True si l'analogie est logique, False si physiquement impossible (ex: quantité négative d'objets discrets).")
    motif_rejet: str = Field(description="Si invalide, expliquer brièvement l'erreur didactique.")

def analyser_coherence_semantique(texte_reponse, client):
    """
    Fonction exécutive d'inhibition.
    1. Utilise spaCy pour pré-détecter les nombres négatifs (analyse syntaxique rigoureuse).
    2. Si risque détecté, utilise Mistral pour structurer la validation dans le schéma Pydantic.
    """
    doc = nlp(texte_reponse)
    risque_negatif = any(token.text.startswith('-') and token.pos_ == "NUM" for token in doc)
    
    if not risque_negatif:
        return True, ""

    prompt_critique = f"""Analyse cette proposition pédagogique d'un tuteur :
"{texte_reponse}"

Détermine si elle contient une aberration didactique ou physique (ex: associer un nombre négatif à un objet physique dénombrable comme '-2 pommes').
Tu DOIS répondre UNIQUEMENT au format JSON strict correspondant aux clés suivantes : 
"contient_analogie", "entites_physiques_detectees", "est_valide_physiquement", "motif_rejet"."""
    
    try:
        reponse_critique = client.chat.completions.create(
            model=MODELE_ALBERT,
            messages=[{"role": "user", "content": prompt_critique}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        json_str = reponse_critique.choices[0].message.content
        
        # Validation stricte du schéma via Pydantic V2
        analyse = ValidationDidactique.model_validate_json(json_str)
        return analyse.est_valide_physiquement, analyse.motif_rejet
        
    except ValidationError as e:
        print(f"Échec de validation Pydantic de l'Agent Critique : {e}")
        return True, "" # Faille gracieuse pour ne pas bloquer l'interface
    except Exception as e:
        print(f"Erreur Agent Critique : {e}")
        return True, ""

def simuler_stream(texte):
    """Générateur pour maintenir la fluidité UX."""
    for mot in texte.split(" "):
        yield mot + " "
        time.sleep(0.01)

# ==========================================
# --- DIALOGUE BILAN FINAL & WOOCLAP ---
# ==========================================
@st.dialog("📈 Ton Bilan de Révision", width="large")
def afficher_bilan():
    if len(st.session_state.messages) > 1:
        with st.spinner("Analyse métacognitive en cours..."):
            client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
            messages_bilan = []
            
            instruction_metacognitive = """<bilan_metacognitif>
<role_et_ton>Tu es un coach pédagogique. Fais un bilan métacognitif factuel, ultra-concis et encourageant. Adresse-toi à l'élève avec 'Tu'. Ne pose plus de question.</role_et_ton>
<contraintes_format>Ton bilan doit être EXTRÊMEMENT BREF, visuel et direct. Utilise des listes à puces et limite-toi à 1 ou 2 phrases maximum par point. Pas de longs paragraphes.</contraintes_format>
<structure_obligatoire>
Structure obligatoirement ton bilan avec les points suivants :
1. 🎯 Tes acquis : Va droit au but sur ce qui est su et ce qui reste à revoir (très bref).
2. 💡 Tes erreurs : Dédramatise et donne LA stratégie précise à utiliser la prochaine fois (1 phrase).
"""
            if "Mode A" in st.session_state.objectif:
                instruction_metacognitive += """3. ⏳ Le piège de la relecture : Rappelle en 1 courte phrase que relire le cours donne l'illusion de savoir (biais de fluence) et que seul l'effort de mémoire compte.
4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : cacher son cours et forcer son cerveau à retrouver les informations sur une feuille blanche.
"""
            else:
                instruction_metacognitive += """3. ⏳ Le piège de la correction : Rappelle en 1 courte phrase que lire une correction donne l'illusion d'avoir compris. La vraie compréhension, c'est savoir l'expliquer soi-même.
4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : reprendre un exercice et expliquer la méthode à voix haute comme à un camarade, ou chercher les erreurs.
"""
            instruction_metacognitive += "</structure_obligatoire>\n</bilan_metacognitif>"

            messages_bilan.append({"role": "system", "content": instruction_metacognitive})
            for msg in st.session_state.messages:
                if not msg.get("isHidden"): messages_bilan.append({"role": msg["role"], "content": msg["content"]})
            messages_bilan.append({"role": "user", "content": "Session terminée. Donne mon bilan."})

            try:
                reponse = client.chat.completions.create(model=MODELE_ALBERT, messages=messages_bilan, temperature=0.7)
                st.success(reponse.choices[0].message.content)
                st.divider()
                st.markdown("### 📊 Évaluation de l'outil")
                iframe_wooclap = """<iframe allowfullscreen frameborder="0" height="100%" src="https://app.wooclap.com/FBXMBG/questionnaires/69ad313cc7cb13027e159133" style="min-height: 550px; min-width: 300px" width="100%"></iframe>"""
                components.html(iframe_wooclap, height=580)
                if st.button("🔄 Nouvelle session"):
                    st.session_state.session_active = False
                    st.session_state.messages = []
                    st.rerun()
            except Exception as e:
                st.error(f"Erreur bilan : {e}")
    else:
        st.warning("Discute un peu avant d'analyser !")

# ==========================================
# 🛑 ZONE SANCTUAIRE : PROMPT SYSTÈME 🛑
# ==========================================
def generer_prompt_systeme(niveau_eleve, objectif_eleve, strategie_generative, matiere, niveau_scolaire, attendus):
    # Injection du référentiel (Bridage ZPD)
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
- Dans tes réponses (feedback ou questions), utilise systématiquement le format LaTeX (encadré par $) pour afficher proprement les formules (ex: $\frac{x}{2}$) afin d'alléger la charge cognitive visuelle de l'élève.
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
# FONCTIONS D'EXTRACTION DE TEXTE
# ==========================================
def extraire_texte_stream(reponse):
    for chunk in reponse:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                yield delta.content

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
    
    mat = st.selectbox("Matière :", list(referentiels.REFERENTIEL_COLLEGE.keys()), disabled=actif)
    niv_scolaire = st.selectbox("Niveau :", list(referentiels.REFERENTIEL_COLLEGE[mat].keys()), disabled=actif)
    
    st.divider()
    niv_cog = st.radio("Niveau de maîtrise :", ["Novice", "Avancé"], disabled=actif)
    obj = st.radio("Objectif :", ["Mode A : Mémorisation", "Mode B : Compréhension"], disabled=actif)
    
    strat = "Classique"
    if "Mode B" in obj:
        s_ui = st.radio("Stratégie :", ["Classique", "Explique à Sacha"], disabled=actif)
        strat = "Effet_Protege" if s_ui == "Explique à Sacha" else "Classique"

    pdf = st.file_uploader("Charge ton cours (PDF)", type=["pdf"], disabled=actif)
    
    if st.button("🚀 Démarrer la session", disabled=actif or not pdf):
        txt = extraire_texte_pdf(pdf)
        if txt:
            st.session_state.texte_cours_integral = txt
            st.session_state.api_key = st.secrets["ALBERT_API_KEY"]
            st.session_state.matiere = mat
            st.session_state.niveau_scolaire = niv_scolaire
            st.session_state.attendus = referentiels.obtenir_attendus(mat, niv_scolaire)
            st.session_state.niveau_cog = niv_cog
            st.session_state.objectif = obj
            st.session_state.strategie = strat
            st.session_state.session_active = True
            st.rerun()

# --- ZONE DE DISCUSSION ---
if st.session_state.session_active:
    col1, col2 = st.columns([3, 1])
    with col1: st.markdown(f"**📚 Révision :** {st.session_state.matiere} | **Mode :** {st.session_state.objectif.split(':')[0]}")
    with col2:
        if st.button("🛑 Terminer et Bilan", type="primary", use_container_width=True): afficher_bilan()
    st.divider()

    client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
    prompt_sys = generer_prompt_systeme(
        st.session_state.niveau_cog, st.session_state.objectif, st.session_state.strategie,
        st.session_state.matiere, st.session_state.niveau_scolaire, st.session_state.attendus
    )

    for msg in st.session_state.messages:
        if not msg.get("isHidden"):
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            ctx = [{"role": "system", "content": prompt_sys},
                   {"role": "user", "content": f"COURS :\n{st.session_state.texte_cours_integral[:15000]}\n\nCommence l'exercice."}]
            flux = client.chat.completions.create(model=MODELE_ALBERT, messages=ctx, stream=True, temperature=0.3)
            rep = st.write_stream(extraire_texte_stream(flux))
            st.session_state.messages.append({"role": "user", "content": "[Document transmis]", "isHidden": True})
            st.session_state.messages.append({"role": "assistant", "content": rep})

    if query := st.chat_input("Ta réponse..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            hist = [{"role": "system", "content": prompt_sys}]
            hist.append({"role": "user", "content": f"COURS : {st.session_state.texte_cours_integral[:5000]}"})
            for m in st.session_state.messages[-MAX_HISTORIQUE_MESSAGES:]:
                if not m.get("isHidden"): hist.append(m)
            
            try:
                # Affichage du spinner pendant le traitement silencieux des agents
                with st.spinner("L'IA analyse ta réponse..."):
                    # ÉTAPE 1 : Génération du brouillon (Agent Tuteur)
                    res_brouillon = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, tools=TOOLS, tool_choice="auto", temperature=0.1)
                    msg_ia = res_brouillon.choices[0].message
                    
                    # ÉTAPE 2 : Traitement des outils mathématiques (SymPy)
                    if msg_ia.tool_calls:
                        hist.append(msg_ia)
                        for tc in msg_ia.tool_calls:
                            args = json.loads(tc.function.arguments)
                            verif = verifier_calcul_formel(args.get('expression_prof',''), args.get('expression_eleve',''))
                            hist.append({"tool_call_id": tc.id, "role": "tool", "name": "verifier_calcul_formel", "content": json.dumps(verif)})
                        
                        res_brouillon = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, temperature=0.3)
                        msg_ia = res_brouillon.choices[0].message

                    brouillon_texte = msg_ia.content

                    # ÉTAPE 3 : Vérification sémantique (Agent Critique via Pydantic + Regex)
                    est_valide, motif_rejet = analyser_coherence_semantique(brouillon_texte, client)
                
                # ÉTAPE 4 : Auto-correction si incohérence détectée (Hors du spinner pour lancer le stream)
                if not est_valide:
                    # Ajout d'une consigne exécutive d'inhibition
                    hist.append({"role": "system", "content": f"ATTENTION (AUTO-CORRECTION) : Ta réponse précédente contenait une erreur didactique. Motif : {motif_rejet}. Régénère ta réponse en corrigeant ce point précis et en trouvant une analogie réaliste."})
                    flux_final = client.chat.completions.create(model=MODELE_ALBERT, messages=hist, stream=True, temperature=0.3)
                    rep = st.write_stream(extraire_texte_stream(flux_final))
                else:
                    # Si valide, on affiche simplement le brouillon
                    rep = st.write_stream(simuler_stream(brouillon_texte))
                
                st.session_state.messages.append({"role": "assistant", "content": rep})
                
            except Exception as e:
                st.error(f"Erreur d'exécution : {e}")
else:
    st.info("👈 Règle tes paramètres et charge ton cours pour commencer.")
