import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai


def configure_api():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    print(env_path)
    load_dotenv(env_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("ERRO: Chave GOOGLE_API_KEY não encontrada no arquivo .env!")
    
    genai.configure(api_key=api_key)
    os.environ["GOOGLE_API_KEY"] = api_key

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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "db", "chroma_dnd")
    
    print(f"\nIniciando a vetorização e salvando no ChromaDB em: {db_path}...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    print("\n✅ Banco ChromaDB criado e salvo com sucesso!")

if __name__ == "__main__":
    configure_api()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Erro: A pasta '{data_dir}' não existe ou está vazia.")
        print("Coloque seus arquivos PDF lá dentro (ex: regras.pdf, lore.pdf).")
        exit()
        
    raw_documents = process_pdfs_in_directory(data_dir)
    text_chunks = get_text_chunks(raw_documents)
    
    print(f"\nTotal de pedaços (chunks) com metadados gerados: {len(text_chunks)}")
    
    print(f"\nTotal de pedaços (chunks) com metadados gerados: {len(text_chunks)}")
    
    if len(text_chunks) == 0:
        print("\n❌ ERRO CRÍTICO: Nenhum texto extraído. Abortando banco de dados.")
        exit()
    # Para evitar rate limit na API do Google durante uma ingestão muito grande:
    time.sleep(2) 
    
    create_and_save_chroma(text_chunks)