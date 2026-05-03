import os
import json
import sys 
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Ajuste de path para importação da memória
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory import ConversationMemory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# 1. CONFIGURAÇÃO DE EXPERIMENTO
# =============================================================================
@dataclass
class ExpConfig:
    provedor: str = "google" # Opções: "google", "openai", "openrouter"
    usar_rag: bool = True
    
    llm_mestre: str = ""
    llm_resumo: str = ""
    
    embedding_model: str = "gemini-embedding-001" 
    
    temperature: float = 0.8
    retrieval_k: int = 10          
    similarity_threshold: float = 0.80 
    max_lore_tokens: int = 1500    
    
    db_path: Path = PROJECT_ROOT / "db" / "chroma_dnd"
    save_path: Path = PROJECT_ROOT / "db" / "savegame.json"

    def __post_init__(self):
        """Auto-configura os modelos corretos dependendo do provedor escolhido."""
        if self.provedor == "google":
            self.llm_mestre = "gemini-2.5-flash" 
            self.llm_resumo = "gemini-3-flash-preview" 
            
        elif self.provedor == "openai":
            self.llm_mestre = "gpt-4o-mini"
            self.llm_resumo = "gpt-4o-mini"
            
        elif self.provedor == "openrouter":
            if not self.llm_mestre:
                self.llm_mestre = "qwen/qwen-2.5-32b-instruct" 
            self.llm_resumo = "openai/gpt-4o-mini" # Usa sempre um modelo barato para o resumo
            
        else:
            raise ValueError(f"Provedor '{self.provedor}' inválido no ExpConfig.")

# =============================================================================
# 2. LOGGERS E SCHEMA
# =============================================================================
metrics_logger = logging.getLogger("benchmark")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    # Garante que a pasta existe antes de tentar escrever nela
    log_dir = PROJECT_ROOT / "db" / "benchmark_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    handler = logging.FileHandler(log_dir / "experiment_metrics.jsonl", encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(handler)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RAGEngine")

class TurnoRPG(BaseModel):
    raciocinio_estado: str = Field(description="O planejamento do mestre")
    narracao: str = Field(description="A narração imersiva")
    opcoes: List[str] = Field(description="Lista com 3 opções de ação")

# =============================================================================
# 3. ENGINE RAG UNIFICADA (O Exoesqueleto Cognitivo)
# =============================================================================
class RAGEngine:
    def __init__(self, config: ExpConfig = None):
        self.cfg = config if config else ExpConfig()
        load_dotenv() 
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.cfg.embedding_model)
        
        self.vectorstore = Chroma(
            persist_directory=str(self.cfg.db_path), 
            embedding_function=self.embeddings
        )
        
        # 👉 FÁBRICA DE LLMs (Limpo e organizado num método próprio)
        self.llm_mestre, llm_resumo = self._inicializar_fabrica_llm()
        
        self.memory = ConversationMemory(llm_resumo, max_turnos_recentes=2)
        self.carregar_progresso()
        
        self.parser = PydanticOutputParser(pydantic_object=TurnoRPG)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um Mestre de RPG para uma aventura SOLO.
            
            LORE DO MUNDO (PDF):
            {lore}
            
            MEMÓRIA DA CAMPANHA:
            {memoria}
            
            REGRAS ESPECÍFICAS:
            {regras}
            
            {format_instructions}"""),
            ("human", "Ação do Jogador: {acao}")
        ])
        
        self.chain = self.prompt | self.llm_mestre | self.parser

    def _inicializar_fabrica_llm(self):
        """Padrão Factory: Retorna a instância do Mestre e do Secretário baseados na config."""
        
        if self.cfg.provedor == "google":
            print(f"🟢 Fábrica: Construindo Mestre GEMINI ({self.cfg.llm_mestre})")
            mestre = ChatGoogleGenerativeAI(
                model=self.cfg.llm_mestre,
                temperature=self.cfg.temperature
            )
            # Como a Google não suporta model_kwargs={"response_format"}, contamos apenas com o Pydantic
            secretario = ChatGoogleGenerativeAI(model=self.cfg.llm_resumo, temperature=0.1)
            
        elif self.cfg.provedor == "openai":
            print(f"🔵 Fábrica: Construindo Mestre OPENAI ({self.cfg.llm_mestre})")
            mestre = ChatOpenAI(
                model=self.cfg.llm_mestre, 
                temperature=self.cfg.temperature,
                model_kwargs={"response_format": {"type": "json_object"}} # 👉 Trava de Segurança OpenAI
            )
            secretario = ChatOpenAI(model=self.cfg.llm_resumo, temperature=0.3)
            
        elif self.cfg.provedor == "openrouter":
            print(f"🟣 Fábrica: Construindo Mestre OPENROUTER ({self.cfg.llm_mestre})")
            mestre = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=self.cfg.llm_mestre,
                temperature=self.cfg.temperature,
                model_kwargs={"response_format": {"type": "json_object"}} # 👉 Trava de Segurança Qwen/OpenRouter
            )
            secretario = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=self.cfg.llm_resumo,
                temperature=0.3
            )
        else:
            raise ValueError("Provedor não suportado na fábrica.")
            
        return mestre, secretario

    def _get_token_count(self, text: str) -> int:
        try:
            if hasattr(self.llm_mestre, 'get_num_tokens'):
                return self.llm_mestre.get_num_tokens(text)
            return len(text) // 4
        except Exception:
            return len(text) // 4 

    async def gerar_turno_async(self, user_input: str, regras: str = "", session_id: str = "default") -> Dict[str, Any]:
        start_time = time.time()
        
        metrics = {
            "session_id": session_id,
            "action": user_input,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        }

        try:
            # 1. RAG
            contexto_gabarito = "Nenhum contexto encontrado no PDF."
            context_parts = []
            tokens_injetados = 0
            
            docs_scores = self.vectorstore.similarity_search_with_score(user_input, k=self.cfg.retrieval_k)
            qtd_docs_recuperados = len(docs_scores)
            
            for doc, score in docs_scores:
                if score > self.cfg.similarity_threshold:
                    continue
                
                doc_tokens = self._get_token_count(doc.page_content)
                if tokens_injetados + doc_tokens > self.cfg.max_lore_tokens:
                    break 
                
                context_parts.append(doc.page_content)
                tokens_injetados += doc_tokens

            if context_parts:
                contexto_gabarito = "\n\n".join(context_parts)

            # 2. ABLAÇÃO
            contexto_lore = "Nenhum contexto adicional. O Mestre deve usar seu próprio conhecimento."
            
            if getattr(self.cfg, 'usar_rag', True):
                contexto_lore = contexto_gabarito
            else:
                qtd_docs_recuperados = 0
                tokens_injetados = 0

            contexto_memoria = self.memory.obter_contexto_formatado()

            # 3. GERAÇÃO
            resposta: TurnoRPG = await self.chain.ainvoke({
                "lore": contexto_lore,
                "memoria": contexto_memoria,
                "regras": regras,
                "acao": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 4. SALVA MEMÓRIA
            output_formatado = f"{resposta.narracao}\nOpções: {', '.join(resposta.opcoes)}"
            self.memory.adicionar_turno(user_input, output_formatado)
            self.salvar_progresso()

            # 5. LOG DE SUCESSO NO JSONL 
            metrics.update({
                "success": True,
                "latency_s": round(time.time() - start_time, 2),
                "retrieved_docs": qtd_docs_recuperados,
                "filtered_docs": len(context_parts) if getattr(self.cfg, 'usar_rag', True) else 0,
                "tokens_injected": tokens_injetados
            })
            metrics_logger.info(json.dumps(metrics, ensure_ascii=False))

            resultado_final = resposta.model_dump()
            
            # Correção da Ilusão de Ótica: Grava no log exatamente o que a IA leu
            if getattr(self.cfg, 'usar_rag', True):
                resultado_final["contexto_usado"] = contexto_gabarito
            else:
                resultado_final["contexto_usado"] = "Nenhum contexto (Baseline)"
                    
            return resultado_final
            
        except Exception as e:
            metrics.update({
                "success": False,
                "error": str(e),
                "latency_s": round(time.time() - start_time, 2)
            })
            metrics_logger.info(json.dumps(metrics, ensure_ascii=False))
            logger.error(f"Erro no Motor RAG: {e}")
            
            # 👉 Adiciona as opções de recuperação de erro se a API cair para o simulador não congelar
            return {
                "raciocinio_estado": f"Ocorreu um erro técnico: {e}",
                "narracao": "Houve um distúrbio na magia do mundo. A conexão com o Mestre foi interrompida temporariamente.",
                "opcoes": ["Tentar novamente a mesma ação", "Esperar que a conexão volte", "Fazer uma pausa"],
                "contexto_usado": "Erro de API"
            }

    def salvar_progresso(self):
        dados_save = {
            "resumo_geral": self.memory.resumo_geral,
            "historico_recente": self.memory.historico_recente
        }
        # Cria a pasta caso não exista
        self.cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.save_path, 'w', encoding='utf-8') as f:
            json.dump(dados_save, f, ensure_ascii=False, indent=4)

    def carregar_progresso(self):
        if self.cfg.save_path.exists():
            with open(self.cfg.save_path, 'r', encoding='utf-8') as f:
                dados_save = json.load(f)
                self.memory.resumo_geral = dados_save.get("resumo_geral", "")
                self.memory.historico_recente = dados_save.get("historico_recente", [])