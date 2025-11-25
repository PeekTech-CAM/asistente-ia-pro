import google.generativeai as genai
import chromadb
import pypdf
import os
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION AND INITIALIZATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("FATAL ERROR: GOOGLE_API_KEY environment variable is not set.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# Load all models at the start
print("Loading models... This may take a moment.")
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
model_chat = genai.GenerativeModel('models/gemini-pro-latest')
client_chroma = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
print("Models loaded and DB client ready.")


# --- 2. MULTI-TENANT (COMPANY) FUNCTIONS ---

def get_company_collection(company_id: int):
    """
    Gets or creates a unique vector database collection for a specific company.
    THIS IS THE CORE OF OUR MULTI-TENANT STRATEGY.
    """
    collection_name = f"company_{company_id}"
    collection = client_chroma.get_or_create_collection(name=collection_name)
    return collection

def process_pdf(file_path: str, company_id: int):
    """
    Loads, processes, and saves a PDF into a specific company's collection.
    """
    print(f"Processing {file_path} for company {company_id}...")
    
    collection = get_company_collection(company_id)
    
    # 1. Read PDF
    try:
        reader = pypdf.PdfReader(file_path)
        texto_completo = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texto_completo += page_text + "\n"
    except Exception as e:
        return {"status": "error", "message": f"Error reading PDF: {e}"}

    if len(texto_completo) < 100:
        return {"status": "error", "message": "Text extraction failed."}

    # 2. Chunk text
    TAMANO_CHUNK = 1000
    SOLAPAMIENTO = 200
    chunks = []
    inicio = 0
    while inicio < len(texto_completo):
        fin = min(inicio + TAMANO_CHUNK, len(texto_completo))
        chunks.append(texto_completo[inicio:fin])
        inicio += (TAMANO_CHUNK - SOLAPAMIENTO)
    chunks = [chunk for chunk in chunks if len(chunk) > 100]

    if not chunks:
        return {"status": "error", "message": "No text chunks generated."}

    # 3. Generate Embeddings and Save
    try:
        lista_embeddings = model_embedding.encode(chunks).tolist()
        lista_ids = [f"{file_path}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path} for _ in chunks]
        
        collection.add(
            embeddings=lista_embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=lista_ids
        )
        return {"status": "success", "message": f"Successfully processed {file_path}."}
    
    except Exception as e:
        return {"status": "error", "message": f"Error in embedding/DB storage: {e}"}


def ask_question(pregunta: str, company_id: int):
    """
    Asks a question ONLY to a specific company's document collection.
    """
    print(f"Received question: '{pregunta}' for company {company_id}")
    
    collection = get_company_collection(company_id)
    
    # 1. Create embedding for the question
    try:
        query_embedding = model_embedding.encode(pregunta).tolist()
    except Exception as e:
        return {"status": "error", "answer": "Error processing question."}

    # 2. Query the database
    if collection.count() == 0:
        return {"status": "no_result", "answer": "This company has no documents."}
        
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
    except Exception as e:
        return {"status": "error", "answer": "Error finding relevant information."}
    
    if not results['documents'] or not results['documents'][0]:
        return {"status": "no_result", "answer": "I could not find an answer."}
    
    contexto = "\n\n".join(results['documents'][0])
    
    # 3. Generate answer with Gemini
    prompt_template = f"""
    Answer the user's question based ONLY on the following context.
    If the answer is not in the context, say "I could not find an answer."
    
    CONTEXT:
    {contexto}
    
    QUESTION:
    {pregunta}
    
    ANSWER:
    """
    
    try:
        respuesta_final = model_chat.generate_content(prompt_template)
        return {"status": "success", "answer": respuesta_final.text}
    except Exception as e:
        return {"status": "error", "answer": f"Error generating response from AI: {e}"}