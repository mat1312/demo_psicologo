"""
Script per indicizzare i documenti della Croce Rossa Italiana nel vector database.
Questo script processa i file dalla cartella 'output' e li indicizza nel vector database.
Versione ottimizzata con operazioni asincrone.
"""

import os
import glob
import argparse
import asyncio
import random
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    DirectoryLoader,
    UnstructuredMarkdownLoader
)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurazione percorsi
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # Directory di output
DATA_DIR = os.path.join(BASE_DIR, "data") # Directory di dati
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store")

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Configurazione chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
# Configurazione per operazioni parallele
MAX_WORKERS = 16  # Parallelismo per operazioni generali
BATCH_SIZE = 800  # Ridotto per evitare rate limit
MAX_CONCURRENT_API_CALLS = 3  # Ridotto per rispettare i rate limit di OpenAI
RETRY_BASE_DELAY = 3  # Ritardo base per i retry in secondi
MAX_RETRIES = 10  # Numero massimo di tentativi prima di fallire


async def load_file(loader_class, file_path, **kwargs):
    """Carica un singolo file usando il loader specificato in modo asincrono."""
    try:
        # Usa ThreadPoolExecutor per operazioni di I/O bloccanti
        with ThreadPoolExecutor() as executor:
            loader = loader_class(file_path, **kwargs)
            docs = await asyncio.get_event_loop().run_in_executor(
                executor, loader.load
            )
            logger.info(f"Caricato: {file_path}")
            return docs
    except Exception as e:
        logger.error(f"Errore nel caricamento di {file_path}: {e}")
        return []


async def load_documents(input_dir: str) -> List[Document]:
    """Carica solo i primi 5 file .md più docsity-dsm-5-riassunto-completo-1.md dalla directory specificata."""
    if not os.path.exists(input_dir):
        logger.error(f"La directory {input_dir} non esiste.")
        return []
    
    tasks = []
    
    # Carica solo file markdown
    md_pattern = os.path.join(input_dir, "**/*.md")
    md_files = glob.glob(md_pattern, recursive=True)
    
    # Cerca il file specifico
    dsm_file = None
    other_md_files = []
    
    for md_path in md_files:
        filename = os.path.basename(md_path)
        if filename == "docsity-dsm-5-riassunto-completo-1.md":
            dsm_file = md_path
        else:
            other_md_files.append(md_path)
    
    # Prendi solo i primi 5 file MD (escluso il file DSM specifico)
    selected_md_files = other_md_files[:5]
    
    # Aggiungi il file DSM se esiste
    if dsm_file:
        selected_md_files.append(dsm_file)
    
    # Accodamento dei file selezionati
    for md_path in selected_md_files:
        logger.info(f"Accodamento MD: {md_path}")
        tasks.append(load_file(UnstructuredMarkdownLoader, md_path))
    
    # Attendi il completamento di tutti i task
    results = await asyncio.gather(*tasks)
    
    # Appiattisci i risultati
    documents = []
    for docs in results:
        documents.extend(docs)
    
    logger.info(f"Caricati {len(documents)} documenti in totale da {len(selected_md_files)} file MD")
    return documents


async def split_documents_async(documents: List[Document]) -> List[Document]:
    """Divide i documenti in chunks più piccoli in modo asincrono."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Usa ThreadPoolExecutor poiché RecursiveCharacterTextSplitter non è async-native
    with ThreadPoolExecutor() as executor:
        chunks = await asyncio.get_event_loop().run_in_executor(
            executor, text_splitter.split_documents, documents
        )
    
    logger.info(f"Documenti suddivisi in {len(chunks)} chunks")
    return chunks


async def create_embeddings_in_batches(chunks: List[Document], embeddings: OpenAIEmbeddings):
    """Crea embeddings in batch in modo asincrono."""
    logger.info(f"Creazione embeddings in batches di {BATCH_SIZE} chunks...")
    
    # Preparazione per batch processing
    total_chunks = len(chunks)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    
    # Processa in batch
    embeddings_list = []
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_chunks)
        batch_texts = texts[i:batch_end]
        logger.info(f"Processando batch {i//BATCH_SIZE + 1}/{(total_chunks-1)//BATCH_SIZE + 1} ({len(batch_texts)} chunks)")
        
        # Ottieni embeddings per il batch corrente
        # Nota: embeddings.embed_documents gestisce internamente il rate limiting
        batch_embeddings = await asyncio.get_event_loop().run_in_executor(
            None, embeddings.embed_documents, batch_texts
        )
        embeddings_list.extend(batch_embeddings)
    
    return embeddings_list, texts, metadatas


async def process_batch_embedding(batch, embeddings, semaphore, batch_idx, total_batches):
    """Processa un singolo batch di embedding con controllo della concorrenza e gestione avanzata degli errori."""
    async with semaphore:
        logger.info(f"Elaborando batch {batch_idx+1}/{total_batches} di {len(batch)} chunks")
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Implementazione robusta con backoff esponenziale
        retries = 0
        while retries < MAX_RETRIES:
            try:
                with ThreadPoolExecutor() as executor:
                    # Esegui l'embedding in modo asincrono
                    embedded_texts = await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        lambda: embeddings.embed_documents(texts)
                    )
                
                logger.info(f"Batch {batch_idx+1}/{total_batches} completato con successo")
                return embedded_texts, texts, metadatas
                
            except Exception as e:
                retries += 1
                # Calcola backoff esponenziale con jitter
                delay = RETRY_BASE_DELAY * (2 ** retries) + (random.random() * 2)
                if "rate_limit" in str(e).lower():
                    logger.warning(f"Rate limit raggiunto per il batch {batch_idx+1}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                else:
                    logger.warning(f"Errore per il batch {batch_idx+1}: {str(e)}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                
                await asyncio.sleep(delay)
        
        # Se arriviamo qui, tutti i tentativi sono falliti
        error_msg = f"Tutti i {MAX_RETRIES} tentativi falliti per il batch {batch_idx+1}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


async def create_vector_store(chunks: List[Document]):
    """Crea il vector store utilizzando Chroma in modo asincrono con gestione robusta dei rate limit."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY non trovata. Imposta la variabile d'ambiente.")
    
    start_time = time.time()
    
    # Importa random per il jitter nel backoff
    import random
    random.seed()
    
    # Usa OpenAI per gli embeddings con configurazione ottimizzata per rate limit
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        chunk_size=BATCH_SIZE,
        request_timeout=120,  # Timeout più lungo per gestire backoff
        max_retries=MAX_RETRIES
    )
    
    # Prepara chunks per l'inserimento nel database
    logger.info(f"Inizializzazione Chroma DB con {len(chunks)} chunks totali...")
    
    # Crea il vector store
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    # Anziché pre-calcolare tutti gli embedding, aggiungiamo direttamente i documenti a Chroma
    # in batch più piccoli. Questo approccio è più lento ma più robusto contro i rate limit
    batch_size = BATCH_SIZE
    total_batches = (len(chunks) - 1) // batch_size + 1
    logger.info(f"Elaborazione in {total_batches} batch di dimensione {batch_size}")
    
    # Crea elenco di batch di documenti
    doc_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    total_processed = 0
    
    # Creiamo un singolo executor per tutta la funzione per evitare problemi di shutdown
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Elaborazione sequenziale con backoff adattivo
        for batch_idx, doc_batch in enumerate(doc_batches):
            logger.info(f"Elaborazione batch {batch_idx + 1}/{len(doc_batches)} ({len(doc_batch)} chunks)")
            
            # Implementazione robusta con backoff esponenziale
            retries = 0
            success = False
            
            while not success and retries < MAX_RETRIES:
                try:
                    # Aggiungi documenti direttamente, lasciando che Chroma gestisca gli embedding
                    await asyncio.get_event_loop().run_in_executor(
                        executor, 
                        lambda: vector_store.add_documents(doc_batch)
                    )
                    
                    total_processed += len(doc_batch)
                    logger.info(f"Batch {batch_idx+1}/{len(doc_batches)} completato con successo")
                    success = True
                    
                    # Persist periodicamente per evitare perdita di dati
                    if (batch_idx + 1) % 5 == 0 or batch_idx == len(doc_batches) - 1:
                        logger.info(f"Salvataggio intermedio dopo il batch {batch_idx+1}/{len(doc_batches)}")
                        await asyncio.get_event_loop().run_in_executor(
                            executor, vector_store.persist
                        )
                        logger.info(f"Progressione: {total_processed}/{len(chunks)} chunks elaborati ({total_processed/len(chunks)*100:.1f}%)")
                    
                except Exception as e:
                    retries += 1
                    # Calcola backoff esponenziale con jitter
                    delay = RETRY_BASE_DELAY * (2 ** retries) + (random.random() * 2)
                    
                    if "rate_limit" in str(e).lower():
                        logger.warning(f"Rate limit raggiunto per il batch {batch_idx+1}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                    else:
                        logger.warning(f"Errore per il batch {batch_idx+1}: {str(e)}. Attesa di {delay:.2f}s. Tentativo {retries}/{MAX_RETRIES}")
                    
                    # Se siamo all'ultimo tentativo, salviamo comunque i progressi finora
                    if retries == MAX_RETRIES - 1:
                        try:
                            logger.info("Ultimo tentativo fallito, salvataggio dei progressi...")
                            await asyncio.get_event_loop().run_in_executor(
                                executor, vector_store.persist
                            )
                        except Exception as save_error:
                            logger.error(f"Errore durante il salvataggio di emergenza: {str(save_error)}")
                    
                    await asyncio.sleep(delay)
            
            # Se il batch ha fallito tutti i tentativi, logghiamo l'errore ma continuiamo con il prossimo batch
            if not success:
                logger.error(f"Tutti i {MAX_RETRIES} tentativi falliti per il batch {batch_idx+1}. Proseguo con il prossimo batch.")
        
        # Persisti il vector store su disco al termine (ancora usando lo stesso executor)
        logger.info("Persistenza finale del vector store su disco...")
        await asyncio.get_event_loop().run_in_executor(
            executor, vector_store.persist
        )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Vector store creato e salvato in {VECTOR_STORE_PATH} in {elapsed_time:.2f} secondi")
    logger.info(f"Totale chunks elaborati con successo: {total_processed}/{len(chunks)}")
    return vector_store


async def save_batches_to_vector_store(batch_results, vector_store, total_chunks):
    """Salva i batch elaborati nel vector store."""
    if not batch_results:
        return 0
    
    total_embedded = 0
    for batch_idx, (embedded_texts, texts, metadatas) in batch_results:
        logger.info(f"Aggiunta al vector store: batch {batch_idx + 1}")
        
        # Usa ThreadPoolExecutor per operazioni bloccanti
        with ThreadPoolExecutor() as executor:
            # Utilizziamo i metodi corretti disponibili in Chroma
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: vector_store.add_documents(
                    [Document(page_content=text, metadata=metadata) 
                     for text, metadata in zip(texts, metadatas)]
                )
            )
        
        total_embedded += len(texts)
    
    # Persist dopo ogni gruppo di batch
    with ThreadPoolExecutor() as executor:
        await asyncio.get_event_loop().run_in_executor(
            executor, vector_store.persist
        )
    
    logger.info(f"Progressione: {total_embedded} nuovi chunks elaborati")
    return total_embedded


async def main_async():
    # Configurazione parser per argomenti da riga di comando
    parser = argparse.ArgumentParser(description='Indicizzazione documenti per l\'assistente CRI')
    parser.add_argument('--input-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory contenente i documenti da indicizzare (default: {OUTPUT_DIR})')
    args = parser.parse_args()
    
    input_dir = args.input_dir
    logger.info(f"Inizializzazione processo di indicizzazione documenti da {input_dir}")
    
    start_time = time.time()
    
    try:
        # Carica i documenti in modo asincrono
        documents = await load_documents(input_dir)
        if not documents:
            logger.warning(f"Nessun documento trovato nella directory {input_dir}")
            return
        
        # Dividi i documenti in chunks in modo asincrono
        chunks = await split_documents_async(documents)
        
        # Crea e salva il vector store in modo asincrono
        await create_vector_store(chunks)
        
        total_time = time.time() - start_time
        logger.info(f"Processo di indicizzazione completato con successo in {total_time:.2f} secondi")
    
    except Exception as e:
        logger.error(f"ERRORE FATALE: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info(f"Processo di indicizzazione FALLITO dopo {time.time() - start_time:.2f} secondi")
        
        # In caso di errore fatale, rialza l'eccezione per far terminare il programma con errore
        raise


def main():
    """Funzione main che avvia il loop di eventi asincrono."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()