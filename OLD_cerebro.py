import google.generativeai as genai
import chromadb
import pypdf
import os
import textwrap
import sys
import time
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURACIÓN INICIAL ---
# Pega tu clave API DE GOOGLE (la de 'asistenteia' / '...Shiio')
GOOGLE_API_KEY = 'AIzaSyCDCjgIlDEYvNPMRfYdlX4Wf5NfRYt7xgo' 
genai.configure(api_key=GOOGLE_API_KEY)

# Carga el modelo de embeddings LOCAL
print("Cargando modelo de embeddings local...")
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
print("¡Modelo de embeddings cargado con éxito!")

# Configura el cliente de la base de datos vectorial (ChromaDB)
try:
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.Settings(anonymized_telemetry=False)
    )
except ImportError:
    print("Error: La biblioteca 'chromadb' parece no estar instalada correctamente.")
    sys.exit(1)

# Crea o usa una "colección"
try:
    collection = client.get_collection(name="documentos_empresa")
    print("Colección 'documentos_empresa' cargada.")
except Exception as e:
    print(f"Colección no encontrada, creando una nueva...")
    collection = client.create_collection(name="documentos_empresa")
    print("Colección 'documentos_empresa' creada.")

# --- 2. FUNCIONES DEL CEREBRO ---

def cargar_y_procesar_pdf(ruta_pdf):
    # (Esta función ya sabemos que funciona, la dejamos como está)
    print(f"Procesando {ruta_pdf}...")
    try:
        reader = pypdf.PdfReader(ruta_pdf)
        texto_completo = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                texto_completo += page_text + "\n"
    except Exception as e:
        print(f"Error al leer el PDF {ruta_pdf}: {e}")
        return
    print(f"Texto extraído: {len(texto_completo)} caracteres.")
    if len(texto_completo) == 0:
        print("Error: No se pudo extraer texto del PDF.")
        return
    TAMANO_CHUNK = 1000
    SOLAPAMIENTO = 200
    chunks = []
    inicio = 0
    while inicio < len(texto_completo):
        fin = min(inicio + TAMANO_CHUNK, len(texto_completo))
        chunks.append(texto_completo[inicio:fin])
        inicio += (TAMANO_CHUNK - SOLAPAMIENTO)
        if inicio >= len(texto_completo):
            break
    chunks = [chunk for chunk in chunks if len(chunk) > 100]
    print(f"Documento dividido en {len(chunks)} trozos.")
    if len(chunks) == 0:
        print("Error: No se generaron trozos.")
        return
    print(f"Generando {len(chunks)} embeddings (vectores) localmente. Esto es rápido...")
    try:
        lista_embeddings = model_embedding.encode(chunks).tolist()
    except Exception as e:
        print(f"Error al generar embeddings localmente: {e}")
        return
    lista_documentos = chunks
    lista_metadatas = [{"source": ruta_pdf} for _ in chunks]
    lista_ids = [f"doc_{ruta_pdf}_{i}" for i in range(len(chunks))]
    if len(lista_ids) > 0:
        collection.add(
            embeddings=lista_embeddings,
            documents=lista_documentos,
            metadatas=lista_metadatas,
            ids=lista_ids
        )
        print(f"¡Éxito! {len(lista_ids)} trozos del documento {ruta_pdf} guardados en la base de datos.")
    else:
        print("No se guardaron nuevos trozos en la base de datos.")


def hacer_pregunta(pregunta):
    
    # 1. Crear embedding para la PREGUNTA (MODO LOCAL)
    try:
        query_embedding = model_embedding.encode(pregunta).tolist()
    except Exception as e:
        print(f"Error al generar embedding para la pregunta: {e}")
        return "Error: No pude procesar tu pregunta."

    # 2. Buscar en ChromaDB los 3 trozos más relevantes
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3 
        )
    except Exception as e:
        print(f"Error al buscar en la base de datos: {e}")
        return "Error: No pude encontrar información relevante."
    
    if not results['documents'] or not results['documents'][0]:
        return "No encontré información relevante para responder a tu pregunta."
        
    contexto = "\n".join(results['documents'][0])
    
    # 3. Generar la respuesta con Gemini (Chat)
    prompt_template = f"""
    Eres un asistente de conocimiento interno. Responde la pregunta del usuario
    basándote ÚNICA Y EXCLUSIVAMENTE en el siguiente contexto.
    
    CONTEXTO:
    {contexto}
    
    PREGUNTA:
    {pregunta}
    
    RESPUESTA:
    """
    
    try:
        # ¡EL CEREBRO INTELIGENTE!
        # Usamos el nombre del modelo que descubrimos con 'revisar_modelos.py'
        model_chat = genai.GenerativeModel('models/gemini-pro-latest')
        
        respuesta_final = model_chat.generate_content(prompt_template)
        return respuesta_final.text
    except Exception as e:
        print(f"Error al generar la respuesta del chat (API de Google): {e}")
        return "Error: Pude encontrar la información, pero fallé al generar la respuesta (Límite de API de Google)."

# --- 3. EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    if GOOGLE_API_KEY == 'TU_CLAVE_API_DE_GOOGLE_AQUI':
        print("="*80)
        print("ERROR: Por favor, edita el archivo 'cerebro.py' y cambia 'TU_CLAVE_API_DE_GOOGLE_AQUI'")
        print("por tu clave API real de Google AI Studio.")
        print("="*80)
        sys.exit(1)
        
    if len(sys.argv) < 3:
        print("Uso:")
        print("  Para cargar un documento: python cerebro.py cargar <ruta_al_pdf>")
        print("  Para hacer una pregunta:  python cerebro.py preguntar \"<tu_pregunta>\"")
    else:
        modo = sys.argv[1]
        argumento = sys.argv[2]
        
        if modo == "cargar":
            if not os.path.exists(argumento):
                print(f"Error: El archivo {argumento} no existe.")
            else:
                try:
                    client.delete_collection(name="documentos_empresa")
                    print("Colección antigua borrada.")
                except Exception as e:
                    print("Colección antigua no encontrada, continuando...")
                
                try:
                    collection = client.create_collection(name="documentos_empresa")
                    print("Colección 'documentos_empresa' creada.")
                    cargar_y_procesar_pdf(argumento)
                except Exception as e:
                    print(f"Error al crear la colección: {e}")
                    
        elif modo == "preguntar":
            respuesta = hacer_pregunta(argumento)
            print("\nRespuesta del Asistente:\n")
            print(textwrap.fill(respuesta, width=80))
        
        else:
            print("Modo no reconocido. Usa 'cargar' o 'preguntar'.")