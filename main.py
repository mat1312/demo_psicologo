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
SIMILARITY_TOP_K = 5
MAX_HISTORY_LENGTH = 6  # Storia più lunga per mantenere contesto terapeutico

# Configurazione ElevenLabs API
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/convai"

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

# Memoria delle conversazioni per ogni sessione
conversation_history: Dict[str, List[Dict[str, str]]] = {}
mood_history: Dict[str, List[str]] = {}  # Traccia l'umore nel tempo

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
async def read_root():
    """Endpoint principale che serve la pagina HTML."""
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

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