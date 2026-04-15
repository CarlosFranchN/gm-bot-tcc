import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Hack para garantir que ele acha a pasta core e o .env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def raio_x_banco():
    # 1. Defina o caminho EXATO onde o ingest salvou
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "db" / "chroma_dnd" 
    
    # IMPORTANTE: Verifique se a pasta existe e tem arquivos dentro
    if not db_path.exists():
        print(f"❌ ERRO: A pasta do banco não existe em: {db_path}")
        return
        
    print(f"✅ Lendo banco de dados em: {db_path}")

    # 2. Conecta ao banco usando o mesmo modelo do ingest
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=str(db_path), 
        embedding_function=embeddings
    )

    # 3. Faz a mesma pergunta do Turno 1 do seu Benchmark
    pergunta_teste = "Você está perdido no deserto de Anauroch, exausto, com pouca água e precisa decidir o que fazer. [Contexto oculto para a busca: deserto calor extremo sobrevivência estrutura misteriosa dunas]"
    
    print("\n🔍 Buscando no RAG (Top 5 resultados)...")
    
    # Busca ignorando o threshold para ver tudo o que volta
    resultados = vectorstore.similarity_search_with_score(pergunta_teste, k=5)
    
    if not resultados:
        print("❌ O banco está vazio ou não retornou nada!")
        return

    # 4. Imprime os resultados e os scores
    for i, (doc, score) in enumerate(resultados):
        print(f"\n--- Resultado {i+1} ---")
        print(f"📊 Score (Distância): {score:.4f} (Lembrete: se for maior que 0.65, o motor está barrando)")
        print(f"📁 Arquivo Origem: {doc.metadata.get('source', 'Desconhecido')}")
        print(f"📜 Trecho do Texto:\n{doc.page_content[:300]}...\n")

if __name__ == "__main__":
    raio_x_banco()