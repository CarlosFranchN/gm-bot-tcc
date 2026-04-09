import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

def configure_api():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(env_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("ERRO: Chave GOOGLE_API_KEY não encontrada no arquivo .env!")
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key

def testar_busca(pergunta, k=2):
    print(f"\n" + "="*50)
    print(f"🔍 BUSCANDO NO BANCO: '{pergunta}'")
    print("="*50)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "db", "chroma_dnd")
    
    # IMPORTANTE: Usando o mesmo modelo do ingest.py
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Conectando ao banco existente
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Realizando a busca por similaridade
    # Retorna os documentos e a pontuação (score) de distância
    resultados = vectorstore.similarity_search_with_score(pergunta, k=k)
    
    if not resultados:
        print("Nenhum resultado encontrado no banco de dados.")
        return

    for i, (doc, score) in enumerate(resultados):
        print(f"\n--- Resultado {i+1} ---")
        # Quanto MENOR o score, MAIS SIMILAR (mais próximo) é o resultado
        print(f"Distância (Score): {score:.4f}") 
        print(f"Categoria: {doc.metadata.get('category', 'N/A')}")
        print(f"Fonte: {doc.metadata.get('source', 'Desconhecida')} (Página {doc.metadata.get('page', 'N/A')})")
        
        # Imprime os primeiros 300 caracteres para você ver o contexto
        texto_limpo = doc.page_content.replace('\n', ' ')[:300]
        print(f"Texto: {texto_limpo}...\n")

if __name__ == "__main__":
    configure_api()
    
    print("Iniciando bateria de testes de recuperação (RAG)...")
    
    # Teste 1: Busca ampla de Lore (Altere para algo que exista no seu PDF)
    testar_busca("O que é o The Desert Wellspring? Quem vive lá?")
    
    # Teste 2: Busca por um personagem, local ou segredo específico da aventura
    testar_busca("Quais são os perigos ou monstros escondidos no oásis?")
    
    testar_busca("Qual é o clima ou atmosfera da aventura?")
    
    testar_busca("Quem é Ankohep?") 