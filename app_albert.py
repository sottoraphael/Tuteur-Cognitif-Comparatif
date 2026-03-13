import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import PyPDF2
import referentiels # Importation de la base de données de programmes (ZPD)
import sympy as sp
import json

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
# GESTION DE L'ÉTAT DE SESSION (State) - CORRECTIF INTÉGRÉ
# ==========================================
# Variables d'état général
if "session_active" not in st.session_state: st.session_state.session_active = False
if "messages" not in st.session_state: st.session_state.messages = []
if "texte_cours_integral" not in st.session_state: st.session_state.texte_cours_integral = ""
if "tutoriel_vu" not in st.session_state: st.session_state.tutoriel_vu = False

# Variables pédagogiques et institutionnelles (Évite les AttributeError)
if "niveau_cog" not in st.session_state: st.session_state.niveau_cog = "Novice"
if "objectif" not in st.session_state: st.session_state.objectif = "Mode A : Mémorisation"
if "strategie" not in st.session_state: st.session_state.strategie = "Classique"
if "matiere" not in st.session_state: st.session_state.matiere = ""
if "niveau_scolaire" not in st.session_state: st.session_state.niveau_scolaire = ""
if "attendus" not in st.session_state: st.session_state.attendus = None
if "api_key" not in st.session_state: st.session_state.api_key = ""

# ==========================================
# OUTIL DE CALCUL FORMEL (DÉLÉGATION NEURO-SYMBOLIQUE)
# ==========================================
def verifier_calcul_formel(expression_prof, expression_eleve):
    """
    Moteur déterministe pour certifier l'équivalence mathématique.
    Élimine les hallucinations de calcul du LLM.
    """
    try:
        exp_p = sp.simplify(str(expression_prof).replace('^', '**'))
        exp_e = sp.simplify(str(expression_eleve).replace('^', '**'))
        est_valide = sp.simplify(exp_p - exp_e) == 0
        return {"est_valide": est_valide, "forme_simplifiee_eleve": str(exp_e)}
    except Exception as e:
        return {"erreur": "Syntaxe mathématique non reconnue par le moteur formel. Demande à l'élève de clarifier sa notation."}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "verifier_calcul_formel",
        "description": "Vérifie l'exactitude mathématique stricte d'un résultat fourni par l'élève par rapport au résultat attendu du cours.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression_prof": {"type": "string", "description": "L'expression correcte (la solution exacte)."},
                "expression_eleve": {"type": "string", "description": "L'expression saisie par l'élève."}
            },
            "required": ["expression_prof", "expression_eleve"]
        }
    }
}]

def simuler_stream(texte):
    """Générateur pour maintenir la fluidité UX (Testing Effect visuel) si aucun outil n'est appelé."""
    for mot in texte.split(" "):
        yield mot + " "
        time.sleep(0.02)

# ==========================================
# --- TUTORIEL D'ACCUEIL ---
# ==========================================
@st.dialog("👋 Bienvenue dans cette application de révision")
def afficher_tutoriel():
    st.markdown("""
        <div style="font-size: 1.15rem; line-height: 1.6; color: #2D3748;">
        Cette application utilise les principes issus des <b>sciences cognitives</b> pour t'aider à réviser efficacement.<br><br>
        <b>💡 Quel mode choisir ?</b><br>
        • <b>Mémorisation :</b> Pour retenir les définitions "par cœur".<br>
        • <b>Compréhension :</b> Pour maîtriser ton cours en profondeur en l'expliquant.<br><br>
        <b>Comment l'utiliser :</b><br>
        1. Règle ta matière et ton niveau scolaire.<br>
        2. Charge ton PDF de cours.<br>
        3. Discute avec Sacha, ton tuteur !
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
            
            client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
            messages_bilan = []
            
            # --- DÉBUT DU BLOC DE BILAN MÉTACOGNITIF (VERSION MISTRAL XML) ---
            instruction_metacognitive = """<bilan_metacognitif>
<role_et_ton>
Tu es un coach pédagogique. Fais un bilan métacognitif factuel, ultra-concis et encourageant. Adresse-toi à l'élève avec 'Tu'. Ne pose plus de question.
</role_et_ton>

<contraintes_format>
Ton bilan doit être EXTRÊMEMENT BREF, visuel et direct. Utilise des listes à puces et limite-toi à 1 ou 2 phrases maximum par point. Pas de longs paragraphes.
</contraintes_format>

<structure_obligatoire>
Structure obligatoirement ton bilan avec les points suivants :
1. 🎯 Tes acquis : Va droit au but sur ce qui est su et ce qui reste à revoir (très bref).
2. 💡 Tes erreurs : Dédramatise et donne LA stratégie précise à utiliser la prochaine fois (1 phrase).
"""

            # On ajoute les conseils spécifiques selon le mode choisi ET on ferme proprement les balises XML
            if "Mode A" in st.session_state.objectif:
                instruction_metacognitive += """3. ⏳ Le piège de la relecture : Rappelle en 1 courte phrase que relire le cours donne l'illusion de savoir (biais de fluence) et que seul l'effort de mémoire compte.
4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : cacher son cours et forcer son cerveau à retrouver les informations sur une feuille blanche.
</structure_obligatoire>
</bilan_metacognitif>
"""
            else:
                instruction_metacognitive += """3. ⏳ Le piège de la correction : Rappelle en 1 courte phrase que lire une correction donne l'illusion d'avoir compris. La vraie compréhension, c'est savoir l'expliquer soi-même.
4. 📝 Prochaine étape : Suggère en 1 courte phrase de faire à la maison exactement comme aujourd'hui : reprendre un exercice et expliquer la méthode à voix haute comme à un camarade, ou chercher les erreurs.
</structure_obligatoire>
</bilan_metacognitif>
"""
            # --- FIN DU BLOC ---

            messages_bilan.append({"role": "system", "content": instruction_metacognitive})

            if st.session_state.texte_cours_integral:
                messages_bilan.append({"role": "user", "content": f"BASE DE CONNAISSANCES DU COURS :\n{st.session_state.texte_cours_integral}"})
                messages_bilan.append({"role": "assistant", "content": "Compris."})
            
            for msg in st.session_state.messages:
                messages_bilan.append({"role": msg["role"], "content": msg["content"]})
                
            messages_bilan.append({"role": "user", "content": "La session est terminée. Donne-moi mon bilan métacognitif ultra-concis selon tes instructions."})

            try:
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
def generer_prompt_systeme(niveau_eleve, objectif_eleve, strategie_generative, matiere, niveau_scolaire, attendus):
    # 1. Construction de la couche Programme Officiel (Grounding)
    if attendus:
        notions = "\n- ".join(attendus.get('notions_cles', ['Non rapporté']))
        vocabulaire = ", ".join(attendus.get('vocabulaire_exigible', ['Non rapporté']))
        limites = "\n- ".join(attendus.get('limites_zpd', ['Aucune limite spécifiée']))
    else:
        notions = "Non rapporté"
        vocabulaire = "Non rapporté"
        limites = "Aucune limite spécifiée"

    cadre_institutionnel = f"""<referentiel_education_nationale>
Matière : {matiere} | Niveau scolaire : {niveau_scolaire}
Pour garantir le respect strict de la Zone Proximale de Développement (ZPD) de l'élève, voici ton cadre institutionnel exclusif :
- NOTIONS AU PROGRAMME :
{notions}

- VOCABULAIRE ATTENDU : {vocabulaire}

- LIMITES ABSOLUES (INTERDICTION D'ABORDER CES POINTS) :
{limites}
</referentiel_education_nationale>\n\n"""

    prompt_systeme = cadre_institutionnel + """# RÔLE ET MISSION
Tu es un expert en ingénierie pédagogique cognitive et spécialiste EdTech.
Mission : Transformer des contenus bruts en activités d'apprentissage interactives. Base-toi EXCLUSIVEMENT sur la "BASE DE CONNAISSANCES DU COURS" fournie au début de la conversation pour le fond.
Objectif : Réduire la distance entre la compréhension actuelle de l'élève et la cible pédagogique, tout en développant sa métacognition.

# ➗ GESTION DES NOTATIONS SCIENTIFIQUES ET MATHÉMATIQUES
- L'élève ne dispose pas de clavier mathématique. Il saisira ses formules en texte brut (ex: "racine de x", "3/4", "x au carre").
- Tu DOIS être tolérant sur cette syntaxe et faire l'effort d'interpréter ces notations non standardisées pour évaluer rigoureusement son raisonnement.
- Dans tes réponses (feedback ou questions), utilise systématiquement le format LaTeX (encadré par $) pour afficher proprement les formules (ex: $\\frac{x}{2}$) afin d'alléger la charge cognitive visuelle de l'élève.

# DIRECTIVES DE GUIDAGE (STRICTES)
1. Flux interactif : Pose UNE SEULE question à la fois. Attends la réponse de l'élève.
2. Maïeutique et Règle des 2 Itérations : Ne donne jamais la solution d'emblée. Fournis des indices (feedback de processus). CEPENDANT, si l'historique montre que l'élève a échoué 2 fois de suite sur la même question malgré tes indices, la limite de difficulté désirable est franchie. Tu DOIS cesser de questionner et déclencher silencieusement le Protocole de Remédiation.
3. Concision extrême : Feedbacks limités à 2 ou 3 phrases MAXIMUM. Aucun cours magistral (sauf en phase de remédiation).
4. Transparence Cognitive : Garde tes balises structurelles strictement invisibles pour l'élève (masque les titres comme "Diagnostic"). En revanche, sois explicite sur la méthode d'apprentissage en utilisant un vocabulaire simple, adapté à un élève. Nomme la stratégie que tu utilises (ex: "récupération en mémoire", "détection d'erreur", "démonstration") et justifie brièvement *pourquoi* elle est utile pour son cerveau (ex:"pour mémoriser plus longtemps", "pour éviter l'illusion de maîtrise", "pour forcer ton cerveau à faire des liens"). Ton texte visible doit rester naturel et conversationnel.
5. Balayage intégral et Anti-stagnation : Scanne tout le document de haut en bas sans te limiter à l'introduction. À chaque nouvelle question, avance dans le cours. Passe au concept suivant dès que l'objectif d'apprentissage de la question est atteint (en Mode Compréhension, cela peut impliquer de demander à l'élève de justifier une réponse juste avant d'avancer), OU s'il échoue à la tâche partielle du Protocole de Remédiation. Dans ce dernier cas d'échec, donne-lui simplement la réponse finale avec bienveillance, et passe obligatoirement à la suite. Ne le bloque jamais indéfiniment.
6. Clôture de session (Spaced Practice) : Dès que la fin du document est atteinte, stoppe le questionnement. Félicite l'élève pour son effort cognitif, et invite-le explicitement à cliquer sur le bouton "🛑 Terminer et voir ma synthèse" situé dans le panneau principal pour découvrir son bilan, puis à fermer l'application pour y revenir dans quelques jours.

# 🛑 CONTRAINTES ET INTERDICTIONS (ANTI-PROMPTS)
- Pas de jugement personnel sur le "Soi" : Ne dis jamais "Tu es nul" ou "Tu es brillant".
- Pas de feedback stéréotypé vide ou immérité : Interdiction de dire juste "C'est juste/faux" sans explication factuelle, et évite les "Bravo !" vagues.
- Pas de comparaison sociale : Ne compare jamais l'élève aux autres.
- ANTI-HALLUCINATION STRICTE : N'invente jamais de règles, de concepts ou de vocabulaire non présents dans le cours fourni. Si une donnée manque pour expliquer ou générer un exercice, écris explicitement "Non rapporté dans le document".

<delegation_neuro_symbolique>
- Tu as accès à un outil de calcul formel nommé `verifier_calcul_formel`.
- DÈS QUE l'élève saisit une réponse contenant un calcul, une expression algébrique ou une valeur numérique, tu DOIS appeler cet outil pour comparer sa réponse avec la solution exacte.
- Ne fais JAMAIS de calcul mental pour évaluer l'élève. Fie-toi uniquement au retour de l'outil pour formuler ton Constat et ton Diagnostic.
</delegation_neuro_symbolique>

# STRUCTURES D'INTERVENTION OBLIGATOIRES
Pour rédiger ta réponse, tu dois formuler un paragraphe unique qui intègre implicitement l'une des trois structures suivantes, selon la situation :

Structure 1 : Feedback de Processus
Intègre ces 3 étapes de manière fluide :
1. Constat factuel : Valide ou invalide le résultat objectivement.
2. Diagnostic : Identifie précisément la règle ou l'étape bloquante/réussie (Haute Info).
3. Levier stratégique : Indique une méthode cognitive pour déduire la réponse (analogie, décomposition, indice logique basé sur le cours), SANS donner la réponse finale. Interdiction stricte de dire simplement "relis le cours". Pousse l'élève à utiliser sa réflexion.

Structure 2 : Feedback d'Autorégulation et Monitorage (Métacognition)
Intègre ces 3 étapes de manière fluide :
1. Effet miroir : Décris la réponse de l'élève de manière factuelle, sans jugement.
2. Activation radar : Interroge son système de détection pour le faire réfléchir sur son action OU demande-lui d'évaluer l'efficacité de la méthode qu'il vient d'utiliser.
3. Ouverture : Pousse-le à la décision ou à l'action corrective sans donner la réponse.

Structure 3 : Protocole de Remédiation (À déclencher EXCLUSIVEMENT après 2 échecs consécutifs)
1. Démonstration pas-à-pas (Problème résolu) : Stoppe le questionnement. Donne la bonne réponse exacte à la question bloquante et explique la démarche pas-à-pas en utilisant UNIQUEMENT le vocabulaire du cours.
2. Tâche partielle (Échafaudage) : Relance avec une question isomorphe (même structure logique, mais avec d'autres variables tirées du cours). Fournis le début de la résolution pour que l'élève n'ait qu'à compléter la dernière étape. Si le cours ne permet pas de créer une question isomorphe, simplifie simplement la question initiale.

# EXEMPLES DE RÉPONSES ATTENDUES (FEW-SHOT PROMPTING)
Voici comment tu dois formuler tes réponses pour qu'elles soient naturelles et intègrent les étapes sans les nommer :

Exemple de Feedback de Processus avec Transparence Cognitive :
"Tu as bien identifié que la photosynthèse nécessite de la lumière. Cependant, tu as oublié un élément gazeux indispensable dans ton équation. Pour forcer ton cerveau à faire le lien, pense à ce que les êtres humains expirent lors de la respiration : la plante utilise précisément ce gaz de l'air pour se nourrir. Quel est-il ?"

Exemple de Feedback d'Autorégulation attendu :
"Tu as écrit que la Révolution a commencé en 1792. Regarde attentivement la chronologie dans ton document. Quel événement majeur de 1789 marque réellement le début de cette période ?"
"""

    if niveau_eleve == "Novice":
        prompt_systeme += """
# 🌳 PROFIL ÉLÈVE : NOVICE
L'élève construit sa compétence et est sujet à la surcharge cognitive.
- INTERDICTION ABSOLUE : N'utilise JAMAIS le Feedback d'Autorégulation.
- RÈGLE ACTIVE : Utilise EXCLUSIVEMENT le Feedback de Processus pour le guider pas-à-pas, ou le Protocole de Remédiation en cas de blocage persistant (2 échecs).
"""
    else:
        prompt_systeme += """
# 🌳 PROFIL ÉLÈVE : AVANCÉ
L'élève possède les bases mais peut faire des étourderies.
- Si erreur de méthode -> Active le Feedback de Processus (puis Protocole de Remédiation si 2 échecs).
- Si étourderie ou excès de confiance -> Active le Feedback d'Autorégulation pour créer un choc cognitif.
"""

    if "Mode A" in objectif_eleve:
        prompt_systeme += """
<constitution_pedagogique mode="A_Ancrage_Memorisation">
- Règle de l'information minimale : 1 question = 1 savoir atomique.
- Stratégie des leurres (Distracteurs) :
  1. Confusion conceptuelle (terme proche, définition différente).
  2. Erreur intuitive (bon sens apparent, mais faux).
  3. Inversion causale (inverse la cause et l'effet).
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
# LA "CONSTITUTION" PÉDAGOGIQUE - MODE B : COMPRÉHENSION & TRANSFERT (Apprentissage Génératif)
- Séquençage : L'élève effectue cet exercice PENDANT l'étude, avec le document sous les yeux (à livre ouvert).
- Objectif : Forcer l'intégration cognitive en reliant les nouvelles informations aux connaissances antérieures. Ce n'est pas un test de mémorisation.
- Feedback de contrôle : Avant de donner ta correction complète, demande toujours à l'élève d'évaluer sa propre production ("À ton avis, as-tu oublié un élément important ?").
"""
        if strategie_generative == "Effet_Protege":
            prompt_systeme += """
# 🎭 RÔLE TEMPORAIRE : LE CAMARADE EN DIFFICULTÉ (EFFET PROTÉGÉ / PEER TUTORING)
ATTENTION : Oublie ton rôle de tuteur expert. Tu es "Sacha", un élève qui a du mal à comprendre le cours.
Ton but caché est d'obliger l'utilisateur à structurer sa pensée et vulgariser le concept.

🛑 RÈGLES STRICTES DU JEU DE RÔLE :
1. ANTI-RÉCITATION : N'utilise AUCUN terme technique avant l'utilisateur. Rejette le jargon ("C'est trop compliqué, on dirait le prof. Tu peux m'expliquer simplement ?").
2. SCAFFOLDING : Dès ta première intervention, explicite ta surcharge cognitive (« J'ai lu le cours mais tout s'embrouille, par quoi je dois commencer ? »). Ensuite, pose UNE SEULE question naïve à la fois. Si l'explication est trop longue, coupe-le ("Attends, tu vas trop vite. C'est quoi l'étape 1 ?").
3. L'ERREUR INTENTIONNELLE : Injecte la confusion la plus classique que font les novices. Force l'utilisateur à démonter cette erreur logique.
4. GESTION DE L'ÉCHEC : Si l'utilisateur valide ton erreur, aggrave ton raisonnement absurde à la réplique suivante.
5. LIMITE DE BLOCAGE (2 itérations) : Si l'utilisateur échoue 2 fois de suite à t'expliquer ou tourne en rond, casse la boucle en simulant une trouvaille dans le cours : "Attends, j'ai regardé dans le manuel, ils disent que c'est [Solution du cours]. Mais du coup, comment on applique ça pour [Question similaire] ?"
6. DÉCLIC ET ÉVALUATION INVERSÉE : Si l'utilisateur corrige ton erreur clairement, reformule avec ses mots. Valorise sa pédagogie en explicitant le déclic ("Ton exemple m'a débloqué parce qu'avant je confondais avec [X]"). Demande-lui une question piège pour te tester.
"""
        else:
            prompt_systeme += """
# POSTURE TUTEUR COGNITIF (INFÉRENCE ET GÉNÉRATION)
RÈGLE D'INFÉRENCE STRICTE : Bannis les questions littérales. Ne demande jamais de retrouver une information explicitement écrite. Force l'élève à déduire des liens (causaux, chronologiques) ou à cibler le "Pourquoi".

MENU GÉNÉRATIF (Choisis la stratégie la plus pertinente si non précisée) :
1. Pré-test (Amorçage) : Pose 3 à 5 questions d'inférence ciblées AVANT la lecture complète.
2. Auto-explication ciblée : Demande à l'élève de justifier une information ou une étape CORRECTE du document (ex: "Quelle hypothèse scientifique justifie ce calcul/ce choix ?"). Ne lui demande pas de justifier son propre raisonnement initial pour éviter d'ancrer ses erreurs.
3. Résumé avec ses mots : Refuse la paraphrase littérale. Exige une réorganisation personnelle.
4. Détection d'erreurs : Rédige un court paragraphe, calcul ou raisonnement contenant une erreur typique de la discipline, et force l'élève à inférer la règle violée.
"""
            if niveau_eleve == "Novice":
                prompt_systeme += """
# ÉCHAFAUDAGE NOVICE
- Consignes très structurées : Impose 3 à 5 mots-clés OBLIGATOIRES du cours.
- Détection d'erreurs : Indique précisément OÙ se trouve l'erreur dans ton texte, la seule tâche de l'élève est d'expliquer pourquoi c'est faux.
- Support : Utilise des textes à trous pour guider l'inférence.
"""
            else:
                prompt_systeme += """
# ÉCHAFAUDAGE AVANCÉ
- Consignes ouvertes : Pose des questions larges SANS fournir de mots-clés.
- Détection d'erreurs : Ne dis pas où est l'erreur. L'élève doit chercher, identifier ET justifier l'erreur seul.
"""

    return prompt_systeme

# ==========================================
# FONCTIONS TECHNIQUES
# ==========================================
def extraire_texte_pdf(uploaded_file):
    texte_complet = ""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            texte_page = page.extract_text()
            if texte_page:
                texte_complet += texte_page + "\n"
        return texte_complet
    except Exception as e:
        st.error(f"Erreur PDF : {e}")
        return None

def construire_messages_albert(prompt_systeme, historique, nouvel_input, texte_cours):
    messages = [{"role": "system", "content": prompt_systeme}]
    if texte_cours:
        messages.append({"role": "user", "content": f"BASE DE CONNAISSANCES DU COURS :\n{texte_cours[:20000]}"})
        messages.append({"role": "assistant", "content": "J'ai bien mémorisé l'intégralité de la base de connaissances. Je suis prêt à formuler mes questions en me basant strictement sur ce contenu et sur les limites de mon programme."})
    for msg in historique[-MAX_HISTORIQUE_MESSAGES:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    if nouvel_input:
        messages.append({"role": "user", "content": nouvel_input})
    return messages

def extraire_texte_stream(reponse):
    for chunk in reponse:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content is not None:
                yield delta.content

# ==========================================
# INTERFACE UTILISATEUR (UI)
# ==========================================
st.title("🦉 Sacha - Tuteur Cognitif")

if not st.session_state.tutoriel_vu:
    afficher_tutoriel()

with st.sidebar:
    st.header("⚙️ Configuration")
    actif = st.session_state.session_active
    
    list_mat = list(referentiels.REFERENTIEL_COLLEGE.keys())
    mat_select = st.selectbox("Matière :", list_mat, disabled=actif)
    list_niv = list(referentiels.REFERENTIEL_COLLEGE[mat_select].keys())
    niv_select = st.selectbox("Niveau scolaire :", list_niv, disabled=actif)
    
    st.divider()
    niv_cog = st.radio("Maîtrise cognitive :", ["Novice", "Avancé"], disabled=actif)
    obj_rev = st.radio("Objectif :", ["Mode A : Mémorisation", "Mode B : Compréhension"], disabled=actif)
    
    strat = "Classique"
    if "Mode B" in obj_rev:
        s_ui = st.radio("Stratégie :", ["Classique", "Explique à Sacha"], disabled=actif)
        strat = "Effet_Protege" if s_ui == "Explique à Sacha" else "Classique"

    st.divider()
    pdf_file = st.file_uploader("Charge ton cours (PDF)", type=["pdf"], disabled=actif)
    
    if st.button("🚀 Démarrer la session", disabled=actif or not pdf_file):
        try:
            with st.spinner("Analyse du document et alignement programme..."):
                txt = extraire_texte_pdf(pdf_file)
                if txt:
                    st.session_state.texte_cours_integral = txt
                    st.session_state.api_key = st.secrets["ALBERT_API_KEY"]
                    st.session_state.matiere = mat_select
                    st.session_state.niveau_scolaire = niv_select
                    st.session_state.attendus = referentiels.obtenir_attendus(mat_select, niv_select)
                    st.session_state.niveau_cog = niv_cog
                    st.session_state.objectif = obj_rev
                    st.session_state.strategie = strat
                    st.session_state.session_active = True
                    st.rerun()
        except KeyError:
            st.error("⚠️ La clé ALBERT_API_KEY est introuvable dans les secrets.")
        except Exception as e:
            st.error(f"Erreur d'initialisation : {e}")

# --- ZONE DE DISCUSSION ---
if st.session_state.session_active:
    
    # Bouton d'arrêt placé bien en évidence au-dessus du chat
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**📚 Révision en cours :** {st.session_state.matiere} | **Mode :** {st.session_state.objectif.split(':')[0]}")
    with col2:
        if st.button("🛑 Terminer et voir ma synthèse", type="primary", use_container_width=True):
            afficher_bilan()
            
    st.divider()

    client = OpenAI(api_key=st.session_state.api_key, base_url=ALBERT_BASE_URL)
    prompt_sys = generer_prompt_systeme(
        st.session_state.niveau_cog, 
        st.session_state.objectif, 
        st.session_state.strategie,
        st.session_state.matiere,
        st.session_state.niveau_scolaire,
        st.session_state.attendus
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            with st.spinner("Je prépare l'exercice..."):
                messages = construire_messages_albert(
                    prompt_sys, 
                    st.session_state.messages, 
                    "Salut ! Je suis prêt, commence l'exercice sur le cours.", 
                    st.session_state.texte_cours_integral
                )
                try:
                    reponse_stream = client.chat.completions.create(
                        model=MODELE_ALBERT,
                        messages=messages,
                        stream=True,
                        temperature=0.3
                    )
                    reponse_complete = st.write_stream(extraire_texte_stream(reponse_stream))
                    st.session_state.messages.append({"role": "assistant", "content": reponse_complete})
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")

    if query := st.chat_input("Écris ta réponse ici..."):
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            with st.spinner("Analyse de ta réponse..."):
                messages = construire_messages_albert(
                    prompt_sys, 
                    st.session_state.messages[:-1],
                    query, 
                    st.session_state.texte_cours_integral
                )
                try:
                    # 1er Appel : Vérification d'outil (Sans stream pour capturer proprement le Tool Call)
                    reponse = client.chat.completions.create(
                        model=MODELE_ALBERT,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=0.1
                    )
                    
                    response_msg = reponse.choices[0].message
                    
                    # 2ème étape : Interception du Tool Call (Délégation à SymPy)
                    if response_msg.tool_calls:
                        messages.append(response_msg) # Ajout de la décision d'outil à l'historique
                        
                        for tool_call in response_msg.tool_calls:
                            if tool_call.function.name == "verifier_calcul_formel":
                                args = json.loads(tool_call.function.arguments)
                                resultat = verifier_calcul_formel(args.get('expression_prof', ''), args.get('expression_eleve', ''))
                                
                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": "verifier_calcul_formel",
                                    "content": json.dumps(resultat)
                                })
                        
                        # 3ème étape : Génération du feedback final avec le résultat mathématique certifié
                        reponse_finale_stream = client.chat.completions.create(
                            model=MODELE_ALBERT,
                            messages=messages,
                            stream=True,
                            temperature=0.3
                        )
                        reponse_complete = st.write_stream(extraire_texte_stream(reponse_finale_stream))
                        st.session_state.messages.append({"role": "assistant", "content": reponse_complete})
                    else:
                        # Si l'IA n'a pas appelé d'outil (ex: question de cours textuelle)
                        # On simule le streaming pour maintenir la cohérence UX
                        reponse_complete = st.write_stream(simuler_stream(response_msg.content))
                        st.session_state.messages.append({"role": "assistant", "content": reponse_complete})
                        
                except Exception as e:
                    st.error(f"Erreur réseau avec le serveur Albert : {e}")

else:
    st.info("👈 Configure les paramètres et charge ton cours pour commencer.")
