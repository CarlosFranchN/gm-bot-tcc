import os
import json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,messages_from_dict, messages_to_dict


def configure():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    env_path = os.path.join(base_dir, ".env")
    
    print(f"Tentando carregar .env de: {env_path}") 
    
    load_dotenv(env_path)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        env_path_alt = os.path.join(script_dir, ".env")
        load_dotenv(env_path_alt)
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("❌ ERRO: GOOGLE_API_KEY não encontrada! Verifique se o arquivo .env existe e tem a chave.")
    
    os.environ["GOOGLE_API_KEY"] = api_key

def get_engine():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "db", "chroma_dnd")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # Temperatura em 0.8 para dar um tom narrativo sem perder a precisão do PDF
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.8)
    return vectorstore, llm

# def save_progress(chat_history):
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     save_path = os.path.join(base_dir, "db", "savegame.json")
#     # Converte os objetos de mensagem do LangChain para dicionários JSON
#     data = messages_to_dict(chat_history)
#     with open(save_path, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

def save_progress(full_history):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "db", "savegame.json")
    # Salva TUDO, não corta aqui!
    data = messages_to_dict(full_history)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# def load_progress():
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     save_path = os.path.join(base_dir, "db", "savegame.json")
#     if os.path.exists(save_path):
#         with open(save_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#             return messages_from_dict(data)
#     return []

def load_progress():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, "db", "savegame.json")
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return messages_from_dict(data)
    return []

if __name__ == "__main__":
    configure()
    vectorstore, llm = get_engine()
    
    # Carrega histórico COMPLETO do disco
    full_history = load_progress()
    
    print("\n" + "="*50)
    print("🎲 MESTRE DE RPG - SOLO ADVENTURE")
    if full_history:
        print(f">> Progresso carregado! {len(full_history)//2} turnos anteriores.")
    print("="*50 + "\n")

    while True:
        user_input = input("Você: ")
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Salvando progresso...")
            save_progress(full_history)
            break

        docs = vectorstore.similarity_search(user_input, k=4)
        contexto_lore = "\n\n".join([doc.page_content for doc in docs])

        prompt_sistema = f"""Você é um Mestre de RPG para uma aventura SOLO.
Diretrizes:
- Use sempre a 2ª pessoa do SINGULAR (Você/Tu).
- Use o CONTEXTO abaixo para detalhes.
- NUNCA narre as ações do jogador.
- Foque na atmosfera e descrição sensorial.

CONTEXTO DO LORE:
{contexto_lore}
"""
        
        # ✅ Separa Memória de Longo Prazo (Disco) de Curto Prazo (API)
        # Envia apenas os últimos 10 para a API (economiza token e evita confusão)
        context_history = full_history[-10:] if len(full_history) > 10 else full_history
        
        mensagens = [SystemMessage(content=prompt_sistema)]
        mensagens.extend(context_history)
        mensagens.append(HumanMessage(content=user_input))

        try:
            response = llm.invoke(mensagens)
            resposta_final = response.content
            # Tratamento de segurança para o conteúdo
            if isinstance(resposta_final, list):
                resposta_final = resposta_final[0].get('text', str(resposta_final))

            print(f"\n📜 Mestre: {resposta_final}\n")

            # ✅ Adiciona ao histórico COMPLETO (que fica na RAM e vai para o disco)
            full_history.append(HumanMessage(content=user_input))
            full_history.append(AIMessage(content=resposta_final))
            
            # Salva a cada turno para garantir segurança dos dados da pesquisa
            save_progress(full_history)
                
        except Exception as e:
            print(f"\n[ERRO]: {e}\n")
            # Salva mesmo em erro para não perder o que já foi feito
            save_progress(full_history)