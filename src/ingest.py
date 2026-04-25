import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def configure_api():
    env_path = PROJECT_ROOT / 'src' / '.env'
    print(env_path)
    load_dotenv(env_path)
    api_key = os.getenv("GOOGLE_API_KEY_MESTRE")
    if not api_key:
        raise ValueError("ERRO: Chave GOOGLE_API_KEY não encontrada no arquivo .env!")
    
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY_MESTRE"] = api_key

def process_pdfs_in_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            print(f"Lendo e extraindo metadados de: {filename}...")
            
            # Usando o PyMuPDFLoader que é muito melhor para RPGs
            loader = PyMuPDFLoader(file_path)
            docs_extraidos = loader.load()
            
            if not docs_extraidos:
                print(f"⚠️ AVISO: Nenhum texto encontrado no arquivo {filename}. Ele pode ser um PDF escaneado (só imagens).")
            else:
                documents.extend(docs_extraidos)
                
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # Note que usamos split_documents em vez de split_text para preservar os metadados
    chunks = text_splitter.split_documents(documents)
    
    # Adicionando uma categoria manual baseada no nome do arquivo para facilitar a filtragem depois
    for chunk in chunks:
        source_file = chunk.metadata.get('source', '').lower()
        if 'regra' in source_file or 'srd' in source_file:
            chunk.metadata['category'] = 'rules'
        else:
            chunk.metadata['category'] = 'lore'
            
    return chunks

def create_and_save_chroma(chunks):
    db_path = PROJECT_ROOT / "db" / "chroma_dnd"
    
    print(f"\nIniciando a vetorização e salvando no ChromaDB em: {db_path}...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY_MESTRE")
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path)
    )
    print("\n✅ Banco ChromaDB criado e salvo com sucesso!")

if __name__ == "__main__":
    configure_api()
    
    # Usa o PROJECT_ROOT para ir direto na pasta data/ original
    data_dir = PROJECT_ROOT / "data"
    
    # Verifica se a pasta existe e se tem arquivos dentro
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"Erro: A pasta '{data_dir}' não existe ou está vazia.")
        print("Coloque seus arquivos PDF lá dentro (ex: regras.pdf, lore.pdf).")
        exit()
        
    raw_documents = process_pdfs_in_directory(data_dir)
    text_chunks = get_text_chunks(raw_documents)
    
    print(f"\nTotal de pedaços (chunks) com metadados gerados: {len(text_chunks)}")
    
    if len(text_chunks) == 0:
        print("\n❌ ERRO CRÍTICO: Nenhum texto extraído. Abortando banco de dados.")
        exit()
        
    # Pausa para estabilizar API
    time.sleep(2) 
    
    create_and_save_chroma(text_chunks)