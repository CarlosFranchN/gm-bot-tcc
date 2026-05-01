import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carrega a sua GOOGLE_API_KEY do ficheiro .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("🔍 Modelos Disponíveis para Geração de Texto:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        # Mostra apenas o nome limpo (sem o prefixo 'models/')
        nome_limpo = m.name.replace('models/', '')
        print(f" - {nome_limpo}")