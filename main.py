"""
Backend principale per il Psicologo Virtuale.
Include RAG con Chroma, gestione conversazioni e API per il frontend.
Integrazione con ElevenLabs per analisi conversazioni vocali.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
from pathlib import Path

# LangChain imports
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate



from fastapi import Cookie, Depends, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, Cookie, Depends, status

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")  # Aggiungi questa variabile a .env

# Configurazione percorsi
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_PATH = DATA_DIR / "vector_store"

# Configurazione LLM
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.5  # Leggermente più alto per risposte più empatiche
MAX_TOKENS = 15000
SIMILARITY_TOP_K = 8  # Numero di documenti simili da recuperare
MAX_HISTORY_LENGTH = 6  # Storia più lunga per mantenere contesto terapeutico

# Configurazione ElevenLabs API
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/convai"


# Modelli per l'autenticazione
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    email: Optional[str] = None

# Modelli Pydantic per le richieste e risposte API
class Source(BaseModel):
    file_name: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    session_id: str
    mood: Optional[str] = None  # Opzionale: per tracciare l'umore del paziente

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Source]] = []
    analysis: Optional[str] = None  # Analisi psicologica opzionale

class ResetRequest(BaseModel):
    session_id: str

class ResetResponse(BaseModel):
    status: str
    message: str

class SessionSummaryResponse(BaseModel):
    summary_html: str

class MoodAnalysisResponse(BaseModel):
    mood_analysis: str

# Nuovi modelli per ElevenLabs
class ElevenLabsConversation(BaseModel):
    agent_id: str
    conversation_id: str
    start_time_unix_secs: Optional[int] = None
    call_duration_secs: Optional[int] = None
    message_count: Optional[int] = None
    status: str
    call_successful: Optional[str] = None
    agent_name: Optional[str] = None

class ElevenLabsConversationsResponse(BaseModel):
    conversations: List[ElevenLabsConversation]
    has_more: bool
    next_cursor: Optional[str] = None

class ElevenLabsTranscriptMessage(BaseModel):
    role: str
    time_in_call_secs: int
    message: Optional[str] = None

class ElevenLabsConversationDetail(BaseModel):
    agent_id: str
    conversation_id: str
    status: str
    transcript: List[ElevenLabsTranscriptMessage]
    metadata: Dict[str, Any]

# Modello per la richiesta e risposta dei resource
class ResourceRequest(BaseModel):
    query: str
    session_id: str

class ResourceResponse(BaseModel):
    resources: List[Dict[str, str]]

# Nuovi modelli per l'analisi combinata
class AnalysisSourceRequest(BaseModel):
    session_id: str
    analyze_chatbot: bool = True
    analyze_elevenlabs: bool = False
    elevenlabs_conversation_id: Optional[str] = None
    
 # Modelli per l'analisi delle patologie   
    
class PathologyAnalysisRequest(BaseModel):
    session_id: str
    analyze_chatbot: bool = True
    analyze_elevenlabs: bool = False
    elevenlabs_conversation_id: Optional[str] = None

class PathologyItem(BaseModel):
    name: str
    description: str
    confidence: float
    key_symptoms: List[str]
    source: Optional[str] = None

class PathologyAnalysisResponse(BaseModel):
    possible_pathologies: List[PathologyItem]
    analysis_summary: str

# Memoria delle conversazioni per ogni sessione
conversation_history: Dict[str, List[Dict[str, str]]] = {}
mood_history: Dict[str, List[str]] = {}  # Traccia l'umore nel tempo

# Gestione sessioni e utenti
active_sessions: Dict[str, Dict[str, Any]] = {}  # token: {user_id, email, created_at}
user_sessions: Dict[str, List[str]] = {}  # user_id: [tokens]



# Funzione per verificare l'autenticazione
async def get_current_user(session_token: Optional[str] = Cookie(None, alias="session_token")):
    if not session_token or session_token not in active_sessions:
        return None
    
    # Verifica che la sessione non sia scaduta (24 ore)
    session_data = active_sessions[session_token]
    if time.time() - session_data.get("created_at", 0) > 86400:  # 24 ore
        # Rimuovi la sessione scaduta
        del active_sessions[session_token]
        user_id = session_data.get("user_id")
        if user_id and user_id in user_sessions:
            user_sessions[user_id] = [t for t in user_sessions[user_id] if t != session_token]
        return None
    
    return session_data

# Inizializza FastAPI
app = FastAPI(title="Psicologo Virtuale API")

# Configurazione CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Collega i file statici
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Sistema di prompt
condense_question_prompt = PromptTemplate.from_template("""
Data la seguente conversazione terapeutica e una domanda di follow-up, riformula la domanda
in modo autonomo considerando il contesto della conversazione precedente.

Storico conversazione:
{chat_history}

Domanda di follow-up: {question}

Domanda autonoma riformulata:
""")

qa_prompt = PromptTemplate.from_template("""
Sei uno psicologo virtuale professionale. Il tuo ruolo è quello di fornire supporto psicologico, ascoltare 
con empatia e offrire risposte ponderate basate sulle migliori pratiche psicologiche.

Devi:
1. Mantenere un tono empatico, rispettoso e non giudicante
2. Utilizzare tecniche di ascolto attivo e di riflessione
3. Fare domande aperte che incoraggino l'introspezione
4. Evitare diagnosi definitive (non sei un sostituto di un professionista in carne ed ossa)
5. Suggerire tecniche di auto-aiuto basate su evidenze scientifiche
6. Identificare eventuali segnali di crisi e suggerire risorse di emergenza quando appropriato

Ricorda: in caso di emergenza o pensieri suicidi, devi sempre consigliare di contattare immediatamente 
i servizi di emergenza o le linee telefoniche di supporto psicologico.

Stato emotivo attuale dichiarato dal paziente: {current_mood}
Adatta il tuo approccio terapeutico in base a questo stato emotivo. Per esempio:
- Se il paziente si sente "ottimo", sostieni il suo stato positivo ma esplora comunque aree di crescita
- Se il paziente si sente "male", usa un tono più delicato, empatico e supportivo
- Se il paziente è "neutrale", aiutalo a esplorare e identificare meglio le sue emozioni
Ricorda che lo stato emotivo dichiarato è solo un punto di partenza e potrebbe non riflettere completamente 
la complessità emotiva del paziente.

Base di conoscenza:
{context}

Conversazione precedente:
{chat_history}

Domanda: {question}

Risposta:
""")

def get_vectorstore():
    """Carica il vector store da disco."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    if not VECTOR_STORE_PATH.exists():
        raise FileNotFoundError(f"Vector store non trovato in {VECTOR_STORE_PATH}. Eseguire prima ingest.py.")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embeddings
    )
    return vector_store

def get_conversation_chain(session_id: str):
    """Crea la catena conversazionale con RAG."""
    # Inizializza la memoria se non esiste
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Prepara la memoria per la conversazione
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
        input_key="question"  # Aggiungiamo l'input_key per evitare che current_mood vada nella memoria
    )
    
    # Carica la conversazione dalla memoria
    for message in conversation_history[session_id]:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message["content"])
        else:
            memory.chat_memory.add_ai_message(message["content"])
    
    # Carica il vectorstore
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SIMILARITY_TOP_K}
    )
    
    # Configura il modello LLM
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    # Crea la catena conversazionale
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    return chain

def format_sources(source_docs) -> List[Source]:
    """Formatta i documenti di origine in un formato più leggibile."""
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        
        # Estrai il nome del file dal percorso completo
        file_name = None
        if "source" in metadata:
            # Gestisci sia percorsi con / che con \
            path = metadata["source"].replace('\\', '/')
            file_name_with_ext = path.split('/')[-1]
            
            # Rimuovi l'estensione
            file_name = os.path.splitext(file_name_with_ext)[0]
        
        source = Source(
            file_name=file_name,
            page=metadata.get("page", None),
            text=doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        )
        sources.append(source)
    return sources

# Funzioni per l'integrazione con ElevenLabs
def get_elevenlabs_headers():
    """Restituisce gli headers per le chiamate all'API di ElevenLabs."""
    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    return {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

def get_elevenlabs_conversations(agent_id: Optional[str] = None, page_size: int = 30):
    """Ottiene l'elenco delle conversazioni di ElevenLabs."""
    url = f"{ELEVENLABS_API_BASE}/conversations"
    params = {"page_size": page_size}
    
    if agent_id:
        params["agent_id"] = agent_id
    
    try:
        response = requests.get(url, headers=get_elevenlabs_headers(), params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Errore nel recupero delle conversazioni ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel recupero delle conversazioni ElevenLabs: {str(e)}")

def get_elevenlabs_conversation(conversation_id: str):
    """Ottiene i dettagli di una specifica conversazione di ElevenLabs."""
    url = f"{ELEVENLABS_API_BASE}/conversations/{conversation_id}"
    
    try:
        response = requests.get(url, headers=get_elevenlabs_headers())
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Errore nel recupero della conversazione ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Errore nel recupero della conversazione ElevenLabs: {str(e)}")

def format_elevenlabs_transcript(conversation_detail):
    """Formatta il transcript di ElevenLabs in un formato leggibile per l'analisi."""
    formatted_messages = []
    
    for msg in conversation_detail.get("transcript", []):
        role = "Paziente" if msg.get("role") == "user" else "Psicologo"
        message = msg.get("message", "")
        if message:
            formatted_messages.append(f"{role}: {message}")
    
    return "\n".join(formatted_messages)

@app.get("/", response_class=HTMLResponse)
async def read_root(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Endpoint principale che serve la pagina HTML."""
    if not current_user:
        # Serve la pagina di login se l'utente non è autenticato
        with open(STATIC_DIR / "login.html", "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Serve la pagina principale se l'utente è autenticato
        with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
            content = f.read()
    return HTMLResponse(content=content)





@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Endpoint di login fittizio per la demo."""
    # Nel mondo reale, verificheresti le credenziali nel database
    # Per la demo, accettiamo qualsiasi credenziale valida nel formato
    
    if not request.email or not request.password or '@' not in request.email:
        return LoginResponse(success=False, message="Credenziali non valide.")
    
    # Genera un user_id basato sull'email
    user_id = f"user_{hash(request.email) % 10000}"
    
    # Genera un token di sessione
    token = secrets.token_hex(16)
    
    # Memorizza la sessione
    session_data = {
        "user_id": user_id,
        "email": request.email,
        "created_at": time.time()
    }
    active_sessions[token] = session_data
    
    # Associa il token all'utente
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    user_sessions[user_id].append(token)
    
    # Crea una risposta con il cookie
    response = JSONResponse(
        content=LoginResponse(
            success=True,
            message="Login effettuato con successo!",
            user_id=user_id,
            email=request.email
        ).dict()
    )
    
    # Imposta il cookie di sessione
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,  # Il cookie non è accessibile via JavaScript
        max_age=86400,  # 24 ore
        samesite="lax"
    )
    
    return response

@app.post("/api/logout")
async def logout(request: Request, current_user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """Endpoint per il logout."""
    response = JSONResponse(
        content={"success": True, "message": "Logout effettuato con successo!"}
    )
    
    # Se c'è un utente autenticato, rimuovi la sessione
    if current_user:
        token = request.cookies.get("session_token")
        
        if token and token in active_sessions:
            user_id = active_sessions[token].get("user_id")
            del active_sessions[token]
            
            if user_id and user_id in user_sessions:
                user_sessions[user_id] = [t for t in user_sessions[user_id] if t != token]
    
    # Cancella il cookie di sessione
    response.delete_cookie(key="session_token")
    
    return response




@app.post("/therapy-session", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Endpoint per processare le domande dell'utente e fornire supporto psicologico."""
    try:
        # Ottiene o crea la catena conversazionale
        chain = get_conversation_chain(request.session_id)
        
        # Salva la domanda utente nella storia
        conversation_history.setdefault(request.session_id, [])
        conversation_history[request.session_id].append({
            "role": "user",
            "content": request.query
        })
        
        # Gestione dell'umore
        current_mood = "non specificato"
        
        # Traccia l'umore se fornito
        if request.mood:
            mood_history.setdefault(request.session_id, [])
            mood_history[request.session_id].append(request.mood)
            current_mood = request.mood
        # Se non fornito ma c'è una storia di umore, usa l'ultimo
        elif request.session_id in mood_history and mood_history[request.session_id]:
            current_mood = mood_history[request.session_id][-1]
        
        # Mantiene la storia limitata per evitare di superare i limiti del contesto
        if len(conversation_history[request.session_id]) > MAX_HISTORY_LENGTH * 2:
            conversation_history[request.session_id] = conversation_history[request.session_id][-MAX_HISTORY_LENGTH*2:]
        
        # Esegue la query con l'umore corrente
        result = chain({"question": request.query, "current_mood": current_mood})
        
        # Salva la risposta nella storia
        conversation_history[request.session_id].append({
            "role": "assistant",
            "content": result["answer"]
        })
        
        # Formatta le fonti
        sources = format_sources(result.get("source_documents", []))
        
        # Genera un'analisi opzionale (non mostrata all'utente ma utile per il backend)
        analysis = None
        if len(conversation_history[request.session_id]) > 3:
            llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.1)
            messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[request.session_id][-6:]])
            
            # Includi l'umore dichiarato nell'analisi
            mood_info = ""
            if current_mood != "non specificato":
                mood_info = f"\nIl paziente ha dichiarato di sentirsi: {current_mood}"
            
            analysis_prompt = f"""
            Analizza brevemente questa conversazione terapeutica e identifica:
            1. Temi principali emersi
            2. Stato emotivo del paziente
            3. Eventuali segnali di allarme
            4. Se lo stato emotivo espresso nel contenuto della conversazione corrisponde all'umore dichiarato
            
            {mood_info}
            
            Conversazione:
            {messages_text}
            """
            analysis_response = llm.invoke(analysis_prompt)
            analysis = analysis_response.content
        
        # Ritorna il risultato
        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            analysis=analysis
        )
    
    except Exception as e:
        logger.error(f"Errore nel processare la query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore del server: {str(e)}")

@app.post("/reset-session", response_model=ResetResponse)
async def reset_conversation(request: ResetRequest):
    """Resetta la sessione terapeutica."""
    session_id = request.session_id
    
    if session_id in conversation_history:
        conversation_history[session_id] = []
        if session_id in mood_history:
            mood_history[session_id] = []
        return ResetResponse(status="success", message="Sessione resettata con successo")
    
    return ResetResponse(status="success", message="Nessuna sessione trovata per questo ID")

@app.get("/api/session-summary/{session_id}", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """Genera un riepilogo della sessione terapeutica."""
    if session_id not in conversation_history or not conversation_history[session_id]:
        return SessionSummaryResponse(summary_html="<p>Nessuna sessione disponibile</p>")
    
    messages = conversation_history[session_id]
    
    # Formatta il riepilogo in HTML
    html = """
    <div class="p-4 bg-gray-50 rounded-lg">
        <h2 class="text-xl font-semibold mb-4 text-blue-700">Riepilogo della Sessione</h2>
    """
    
    for idx, message in enumerate(messages):
        role_class = "text-blue-600 font-medium" if message["role"] == "assistant" else "text-gray-700 font-medium"
        role_name = "Psicologo" if message["role"] == "assistant" else "Paziente"
        
        html += f"""
        <div class="mb-4 pb-3 border-b border-gray-200">
            <div class="mb-1"><span class="{role_class}">{role_name}:</span></div>
            <p class="pl-2">{message["content"]}</p>
        </div>
        """
    
    html += "</div>"
    
    # Se disponibile, aggiunge grafico dell'umore
    if session_id in mood_history and mood_history[session_id]:
        html += """
        <div class="mt-6 p-4 bg-gray-50 rounded-lg">
            <h3 class="text-lg font-semibold mb-2 text-blue-700">Tracciamento dell'Umore</h3>
            <div class="mood-chart">
                <!-- Qui si potrebbe inserire un grafico generato con D3.js o simili -->
                <p>Trend dell'umore rilevato durante la sessione.</p>
            </div>
        </div>
        """
    
    return SessionSummaryResponse(summary_html=html)

@app.post("/api/recommend-resources", response_model=ResourceResponse)
async def recommend_resources(request: ResourceRequest):
    """Raccomanda risorse psicologiche basate sulla conversazione."""
    if request.session_id not in conversation_history:
        return ResourceResponse(resources=[])
    
    # Prendi gli ultimi messaggi della conversazione
    messages = conversation_history[request.session_id][-8:]  # ultimi 8 messaggi
    messages_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Chiedi al modello di consigliare risorse
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.3)
    resource_prompt = f"""
    Basandoti su questa conversazione terapeutica, consiglia 3-5 risorse specifiche che potrebbero essere utili per il paziente.
    Per ogni risorsa, fornisci:
    - Titolo
    - Breve descrizione (1-2 frasi)
    - Tipo (libro, app, esercizio, tecnica, video, ecc.)
    
    Conversazione:
    {messages_text}
    
    Restituisci le risorse in formato JSON come questo:
    [
        {{"title": "Titolo della risorsa", "description": "Breve descrizione", "type": "Tipo di risorsa"}},
        ...
    ]
    """
    
    try:
        response = llm.invoke(resource_prompt)
        
        # Estrai JSON dalla risposta
        import re
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            resources = json.loads(json_str)
        else:
            # Fallback se il formato non è corretto
            resources = [{"title": "Mindfulness per principianti", "description": "Tecniche base di mindfulness per la gestione dello stress", "type": "Libro/App"}]
        
        return ResourceResponse(resources=resources)
    
    except Exception as e:
        logger.error(f"Errore nel generare risorse: {str(e)}", exc_info=True)
        return ResourceResponse(resources=[
            {"title": "Errore di generazione", "description": "Non è stato possibile generare risorse personalizzate", "type": "Errore"}
        ])

# Nuovi endpoint per ElevenLabs
@app.get("/api/elevenlabs/conversations", response_model=ElevenLabsConversationsResponse)
async def list_elevenlabs_conversations(agent_id: Optional[str] = None):
    """Restituisce l'elenco delle conversazioni di ElevenLabs."""
    try:
        conversations_data = get_elevenlabs_conversations(agent_id)
        return conversations_data
    except Exception as e:
        logger.error(f"Errore nel recuperare le conversazioni ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/elevenlabs/conversations/{conversation_id}", response_model=ElevenLabsConversationDetail)
async def get_elevenlabs_conversation_detail(conversation_id: str):
    """Restituisce i dettagli di una specifica conversazione di ElevenLabs."""
    try:
        conversation_data = get_elevenlabs_conversation(conversation_id)
        return conversation_data
    except Exception as e:
        logger.error(f"Errore nel recuperare la conversazione ElevenLabs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mood-analysis", response_model=MoodAnalysisResponse)
async def analyze_mood(request: AnalysisSourceRequest):
    """Analizza l'umore e il progresso del paziente basato su diverse fonti."""
    try:
        combined_text = ""
        
        # Raccogli conversazione dal chatbot se richiesto
        if request.analyze_chatbot:
            if request.session_id in conversation_history and conversation_history[request.session_id]:
                chatbot_messages = conversation_history[request.session_id]
                chatbot_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chatbot_messages])
                combined_text += "## Conversazione Chatbot:\n" + chatbot_text + "\n\n"
            else:
                combined_text += "## Conversazione Chatbot:\nNessuna conversazione disponibile\n\n"
        
        # Raccogli conversazione da ElevenLabs se richiesto
        if request.analyze_elevenlabs and request.elevenlabs_conversation_id:
            try:
                elevenlabs_data = get_elevenlabs_conversation(request.elevenlabs_conversation_id)
                elevenlabs_text = format_elevenlabs_transcript(elevenlabs_data)
                combined_text += "## Conversazione Vocale ElevenLabs:\n" + elevenlabs_text + "\n\n"
            except Exception as e:
                combined_text += f"## Conversazione Vocale ElevenLabs:\nErrore nel recupero della conversazione: {str(e)}\n\n"
        
        # Se non ci sono dati, ritorna un messaggio di errore
        if not combined_text.strip():
            return MoodAnalysisResponse(mood_analysis="# Analisi dell'Umore\n\n**Dati insufficienti per l'analisi.**\n\nNon ci sono conversazioni disponibili da analizzare.")
        
        # Analizza il testo combinato
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
        analysis_prompt = f"""
        Analizza questa conversazione terapeutica e fornisci:
        1. Una valutazione dell'umore generale del paziente
        2. Eventuali schemi di pensiero o comportamento ricorrenti
        3. Suggerimenti per il terapeuta su come procedere nella prossima sessione
        
        Formatta la risposta in Markdown seguendo questo formato:
        
        # Analisi della Conversazione Terapeutica
        
        ## 1. Valutazione dell'umore generale del paziente
        [Inserisci qui la tua analisi...]
        
        ## 2. Eventuali schemi di pensiero o comportamento ricorrenti
        [Inserisci qui la tua analisi...]
        
        ## 3. Suggerimenti per il terapeuta su come procedere nella prossima sessione
        - Punto 1
        - Punto 2
        - Punto 3
        
        Conversazione:
        {combined_text}
        """
        
        response = llm.invoke(analysis_prompt)
        return MoodAnalysisResponse(mood_analysis=response.content)
    
    except Exception as e:
        logger.error(f"Errore nell'analisi dell'umore: {str(e)}", exc_info=True)
        return MoodAnalysisResponse(
            mood_analysis=f"# Errore nell'Analisi\n\nSi è verificato un errore durante l'analisi dell'umore: {str(e)}"
        )

# Endpoint legacy per retrocompatibilità
@app.get("/api/mood-analysis/{session_id}", response_model=MoodAnalysisResponse)
async def analyze_mood_legacy(session_id: str):
    """Endpoint legacy per retrocompatibilità."""
    request = AnalysisSourceRequest(
        session_id=session_id,
        analyze_chatbot=True,
        analyze_elevenlabs=False
    )
    return await analyze_mood(request)


@app.post("/api/pathology-analysis", response_model=PathologyAnalysisResponse)
async def analyze_pathologies(request: PathologyAnalysisRequest):
    """Analizza le conversazioni per identificare possibili patologie psicologiche."""
    try:
        combined_text = ""
        
        # Raccogli conversazione dal chatbot se richiesto
        if request.analyze_chatbot:
            if request.session_id in conversation_history and conversation_history[request.session_id]:
                chatbot_messages = conversation_history[request.session_id]
                chatbot_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chatbot_messages])
                combined_text += "## Conversazione Chatbot:\n" + chatbot_text + "\n\n"
            else:
                combined_text += "## Conversazione Chatbot:\nNessuna conversazione disponibile\n\n"
        
        # Raccogli conversazione da ElevenLabs se richiesto
        if request.analyze_elevenlabs and request.elevenlabs_conversation_id:
            try:
                elevenlabs_data = get_elevenlabs_conversation(request.elevenlabs_conversation_id)
                elevenlabs_text = format_elevenlabs_transcript(elevenlabs_data)
                combined_text += "## Conversazione Vocale ElevenLabs:\n" + elevenlabs_text + "\n\n"
            except Exception as e:
                combined_text += f"## Conversazione Vocale ElevenLabs:\nErrore nel recupero della conversazione: {str(e)}\n\n"
        
        # Se non ci sono dati o sono insufficienti, ritorna un messaggio di errore
        if not combined_text.strip():
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary="Dati insufficienti per l'analisi. Non ci sono conversazioni disponibili da analizzare."
            )
            
        # Verifica che ci siano abbastanza messaggi dell'utente per un'analisi significativa
        # Contiamo quanti messaggi utente ci sono e quante parole contengono in totale
        user_messages_count = 0
        total_user_words = 0
        
        if request.analyze_chatbot and request.session_id in conversation_history:
            for msg in conversation_history[request.session_id]:
                if msg["role"] == "user":
                    user_messages_count += 1
                    total_user_words += len(msg["content"].split())
        
        # Se ci sono ElevenLabs, contiamo anche quelli
        if request.analyze_elevenlabs and request.elevenlabs_conversation_id:
            try:
                elevenlabs_data = get_elevenlabs_conversation(request.elevenlabs_conversation_id)
                for msg in elevenlabs_data.get("transcript", []):
                    if msg.get("role") == "user" and msg.get("message"):
                        user_messages_count += 1
                        total_user_words += len(msg.get("message", "").split())
            except Exception:
                # Se c'è un errore, ignoriamo i messaggi ElevenLabs
                pass
        
        # Requisiti minimi per procedere con l'analisi
        MIN_USER_MESSAGES = 1
        MIN_USER_WORDS = 10
        
        if user_messages_count < MIN_USER_MESSAGES or total_user_words < MIN_USER_WORDS:
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary=f"Dati insufficienti per un'analisi clinica significativa. Sono necessari almeno {MIN_USER_MESSAGES} messaggi e {MIN_USER_WORDS} parole dall'utente per procedere. Attualmente: {user_messages_count} messaggi, {total_user_words} parole."
            )
        
        # Utilizziamo il vector store per trovare documenti rilevanti sulle patologie
        vector_store = get_vectorstore()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": SIMILARITY_TOP_K}
        )
        
        # Estrai i sintomi e comportamenti rilevanti dalla conversazione
        llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0.2)
        extraction_prompt = f"""
        Analizza questa conversazione terapeutica ed estrai i sintomi principali, 
        comportamenti problematici o schemi di pensiero che potrebbero essere 
        rilevanti per un'analisi clinica. Fornisci solo i sintomi e comportamenti,
        senza interpretarli o diagnosticarli.
        
        Conversazione:
        {combined_text}
        
        Estrai e elenca solo i sintomi, comportamenti o schemi di pensiero rilevanti, 
        uno per riga. Sii specifico e dettagliato, concentrandoti sui fatti osservabili.
        """
        
        extraction_response = llm.invoke(extraction_prompt)
        extracted_behaviors = extraction_response.content
        
        # Utilizziamo i comportamenti estratti per interrogare il vector store
        docs = retriever.get_relevant_documents(extracted_behaviors)
        
        # Analisi delle patologie basata sui documenti recuperati e sui comportamenti estratti
        analysis_prompt = f"""
        Basandoti sui sintomi e comportamenti estratti dalla conversazione terapeutica e sui documenti
        clinici correlati, identifica possibili patologie psicologiche che potrebbero richiedere ulteriore
        valutazione. Per ogni patologia, fornisci un breve descrizione, i sintomi chiave che l'hanno fatta
        emergere dall'analisi, e una stima di confidenza (da 0.0 a 1.0) basata su quanti sintomi sono presenti.
        
        Sintomi estratti dalla conversazione:
        {extracted_behaviors}
        
        Documenti clinici rilevanti:
        {[doc.page_content for doc in docs]}
        
        Fornisci la risposta nel seguente formato JSON:
        {{
            "possible_pathologies": [
                {{
                    "name": "Nome della patologia",
                    "description": "Breve descrizione",
                    "confidence": 0.7,
                    "key_symptoms": ["sintomo 1", "sintomo 2", ...],
                    "source": "Nome del documento di riferimento"
                }},
                ...
            ],
            "analysis_summary": "Breve riassunto dell'analisi complessiva"
        }}
        
        Includi solo patologie con un minimo di confidenza (0.4 o superiore).
        """
        
        analysis_response = llm.invoke(analysis_prompt)
        
        # Estrai il JSON dalla risposta
        import re
        import json
        json_match = re.search(r'\{.*\}', analysis_response.content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            return PathologyAnalysisResponse(**result)
        else:
            # Fallback nel caso il formato non sia corretto
            return PathologyAnalysisResponse(
                possible_pathologies=[],
                analysis_summary="Non è stato possibile identificare patologie specifiche dai dati forniti. La conversazione potrebbe non contenere informazioni clinicamente rilevanti o potrebbe essere necessario un colloquio più approfondito."
            )
    
    except Exception as e:
        logger.error(f"Errore nell'analisi delle patologie: {str(e)}", exc_info=True)
        return PathologyAnalysisResponse(
            possible_pathologies=[],
            analysis_summary=f"Si è verificato un errore durante l'analisi: {str(e)}"
        )

# Avvio dell'applicazione
if __name__ == "__main__":
    import uvicorn
    try:
        # Verifica che il vector store esista
        get_vectorstore()
        logger.info("Vector store trovato. Avvio del server...")
    except FileNotFoundError:
        logger.error("Vector store non trovato. Eseguire prima ingest.py per indicizzare i documenti con conoscenze psicologiche.")
        exit(1)
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione: {str(e)}", exc_info=True)
        exit(1)
        
    # Avvia il server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)