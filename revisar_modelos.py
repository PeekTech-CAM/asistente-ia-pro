import google.generativeai as genai
import sys

# --- Pega aquí la misma clave API válida que usas en cerebro.py ---
GOOGLE_API_KEY = 'AIzaSyBGFzq59lKUS6IwNSFS9Qq2s--AfwkShio' 
genai.configure(api_key=GOOGLE_API_KEY)

if GOOGLE_API_KEY == 'TU_NUEVA_API_KEY_AQUI':
    print("="*80)
    print("ERROR: Por favor, abre 'revisar_modelos.py' y pon tu clave API real.")
    print("="*80)
    sys.exit(1)

print("Buscando modelos de CHAT (generateContent) disponibles para tu clave API...")
print("=====================================================================")

try:
    # Pedimos a Google la lista de todos los modelos
    for model in genai.list_models():
        # Filtramos solo los modelos que sirven para CHAT (generateContent)
        if 'generateContent' in model.supported_generation_methods:
            print(model.name)

except Exception as e:
    print(f"\nError al contactar con la API de Google: {e}")
    print("\nEsto puede ser por una clave API inválida o un problema de red.")

print("=====================================================================")
print("Fin de la lista. Si la lista está vacía, hay un problema con tu API key o proyecto.")