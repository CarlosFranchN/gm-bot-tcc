import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.memory import ConversationMemory


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
    
    
class RAGEngine:
    def __init__(self):
        configure()
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(base_dir, "db", "chroma_dnd")
        self.save_path = os.path.join(base_dir, "db", "savegame.json")
        
        # Conexão com o Banco de Dados (Chroma)
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.vectorstore = Chroma(persist_directory=self.db_path, embedding_function=embeddings)
        
        # LLM Principal (Mestre) e LLM Secundário (Resumidor)
        # self.llm_mestre = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.8)
        self.llm_mestre = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
        llm_resumo = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
        
        # Inicializa a Memória
        self.memory = ConversationMemory(llm_resumo, max_turnos_recentes=2)
        self.carregar_progresso()

    def gerar_turno(self, user_input,regras_especificas=""):
        """
        Recebe a ação do jogador, busca o contexto, formata a memória e gera a resposta + 3 opções.
        """
        # 1. Busca semântica no PDF
        docs = self.vectorstore.similarity_search(user_input, k=3)
        contexto_lore = "\n\n".join([doc.page_content for doc in docs])
        
        print(f"\n[DEBUG CHROMA] Contexto recuperado: {len(contexto_lore)} caracteres.")
        if len(contexto_lore) == 0:
            print("[DEBUG CHROMA] ALERTA: O banco de dados não encontrou nada!")
        
        
        # 2. Pega a memória (Resumo + Histórico Recente)      
        contexto_memoria = self.memory.obter_contexto_formatado()

        # 3. Prompt forçando Chain-of-Thought para quebrar o Loop Lógico
        prompt_sistema = ChatPromptTemplate.from_messages([
            ("system", """Você é um Mestre de RPG para uma aventura SOLO.
            
            LORE DO MUNDO (PDF):
            {lore}
            
            MEMÓRIA DA CAMPANHA:
            {memoria}
            
            REGRAS ESPECÍFICAS DESTA CENA:
            {regras_cena}
            
            DIRETRIZES DE RESOLUÇÃO E ESTADO:
            1. RESOLUÇÃO IMEDIATA: Não faça suspense (cliffhangers). Se o jogador pediu para abrir algo, diga o que tem dentro NESTE turno. Se pediu para beber, diga o gosto e o efeito NESTE turno. A ação do jogador deve ser concluída antes de você gerar as próximas opções.
            2. Se o jogador pediu para explorar, diga o que ele achou.
            3. NUNCA ofereça opções repetidas.
            
            Você DEVE responder APENAS com um JSON válido neste formato exato:
            {{
                "raciocinio_estado": "Escreva aqui seu planejamento: Qual foi a ação do jogador? Como o mundo muda agora? Quais opções inéditas vou dar?",
                "narracao": "A sua narração imersiva da cena aqui...",
                "opcoes": [
                    "Ação 1 (nova e distinta)",
                    "Ação 2 (nova e distinta)",
                    "Ação 3 (nova e distinta)"
                ]
            }}"""),
            ("human", "Ação do Jogador: {acao}")
        ])
        chain = prompt_sistema | self.llm_mestre
        
        try:
            # 4. Gera a resposta
            resposta = chain.invoke({
                "lore": contexto_lore,
                "memoria": contexto_memoria,
                "regras_cena": regras_especificas, 
                "acao": user_input
            }).content
            
            # Limpa o markdown do JSON (caso o Gemini envie ```json ... ```)
            texto_limpo = resposta.replace("```json", "").replace("```", "").strip()
            dados = json.loads(texto_limpo)
            
            # 5. Salva na memória e no disco
            output_formatado_memoria = f"{dados['narracao']}\nOpções: {', '.join(dados['opcoes'])}"
            self.memory.adicionar_turno(user_input, output_formatado_memoria)
            self.salvar_progresso()
            dados['contexto_usado'] = contexto_lore
            
            return dados
            
        except Exception as e:
            print(f"❌ [ERRO NO ENGINE]: {e}")
            return {"narracao": "Houve um distúrbio na magia do mundo. Tente novamente.", "opcoes": ["Tentar novamente", "Esperar", "Observar"]}

    # --- GERENCIAMENTO DO SAVEGAME ---
    def salvar_progresso(self):
        dados_save = {
            "resumo_geral": self.memory.resumo_geral,
            "historico_recente": self.memory.historico_recente
        }
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(dados_save, f, ensure_ascii=False, indent=4)

    def carregar_progresso(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'r', encoding='utf-8') as f:
                dados_save = json.load(f)
                self.memory.resumo_geral = dados_save.get("resumo_geral", "")
                self.memory.historico_recente = dados_save.get("historico_recente", [])

# ==========================================
# MODO DE TESTE (Para você jogar no terminal)
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎲 ENGINE INICIADA - MODO TERMINAL")
    print("="*50 + "\n")
    
    engine = RAGEngine()
    
    while True:
        acao = input("\nVocê: ")
        if acao.lower() in ["sair", "exit"]:
            print("Salvando e saindo...")
            break
            
        resultado = engine.gerar_turno(acao)
        
        print(f"\n📜 Mestre:\n{resultado['narracao']}")
        print("\nO que você faz?")
        for i, opcao in enumerate(resultado['opcoes']):
            print(f"[{i+1}] {opcao}")

