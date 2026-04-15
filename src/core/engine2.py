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
from pydantic import BaseModel, Field, ValidationError
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.memory import ConversationMemory

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# 1. CONFIGURAÇÃO DE EXPERIMENTO (O "Ouro" para o seu TCC)
# =============================================================================
@dataclass
class ExpConfig:
    # llm_model: str = "gemini-2.5-flash"
    llm_model: str = "gemini-3.1-flash-lite-preview"
    embedding_model: str = "gemini-embedding-001"
    temperature: float = 0.8
    
    # RAG: Busca até 10, mas só aceita os muito relevantes
    retrieval_k: int = 10          
    similarity_threshold: float = 0.65 
    max_lore_tokens: int = 1500    
    
    db_path: Path = PROJECT_ROOT / "db" / "chroma_dnd"
    save_path: Path = PROJECT_ROOT / "db" / "savegame.json"

# =============================================================================
# 2. LOGGERS E SCHEMA
# =============================================================================
# Logger de Métricas (Para gerar os gráficos do TCC)
metrics_logger = logging.getLogger("benchmark")
metrics_logger.setLevel(logging.INFO)
if not metrics_logger.handlers:
    handler = logging.FileHandler("experiment_metrics.jsonl", encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(handler)

# Logger padrão para o terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RAGEngine")

class TurnoRPG(BaseModel):
    raciocinio_estado: str = Field(description="O planejamento do mestre")
    narracao: str = Field(description="A narração imersiva")
    opcoes: List[str] = Field(description="Lista com 3 opções de ação")

# =============================================================================
# 3. ENGINE RAG UNIFICADA
# =============================================================================
class RAGEngine:
    def __init__(self, config: ExpConfig = ExpConfig()):
        self.cfg = config
        load_dotenv()
        
        # Conexão com Modelos
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.cfg.embedding_model)
        self.vectorstore = Chroma(
            persist_directory=str(self.cfg.db_path), 
            embedding_function=self.embeddings
        )
        
        # LLM Principal e Resumidor
        self.llm_mestre = ChatGoogleGenerativeAI(model=self.cfg.llm_model, temperature=self.cfg.temperature)
        llm_resumo = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
        
        # Memória Inteligente (Sua lógica original de volta!)
        self.memory = ConversationMemory(llm_resumo, max_turnos_recentes=2)
        self.carregar_progresso()
        
        # Parser e Chain
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

    def _get_token_count(self, text: str) -> int:
        """Conta tokens NATIVAMENTE sem usar gambiarras do tiktoken."""
        try:
            return self.llm_mestre.get_num_tokens(text)
        except Exception:
            # Fallback seguro caso a API engasgue, usando estimativa de caracteres
            return len(text) // 4 

    async def gerar_turno_async(self, user_input: str, regras: str = "", session_id: str = "default") -> Dict[str, Any]:
        start_time = time.time()
        
        # Setup das métricas
        metrics = {
            "session_id": session_id,
            "action": user_input,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        }

        try:
            # 1. RAG COM FILTRO INTELIGENTE
            docs_scores = self.vectorstore.similarity_search_with_score(user_input, k=self.cfg.retrieval_k)
            
            context_parts = []
            tokens_injetados = 0
            
            for doc, score in docs_scores:
                # O Chroma retorna distância (menor é mais parecido). Ignora se for muito distante.
                if score > self.cfg.similarity_threshold:
                    continue
                
                doc_tokens = self._get_token_count(doc.page_content)
                if tokens_injetados + doc_tokens > self.cfg.max_lore_tokens:
                    break # Orçamento de tokens atingido!
                
                context_parts.append(doc.page_content)
                tokens_injetados += doc_tokens

            contexto_lore = "\n\n".join(context_parts)
            contexto_memoria = self.memory.obter_contexto_formatado()

            # 2. GERAÇÃO
            resposta: TurnoRPG = await self.chain.ainvoke({
                "lore": contexto_lore,
                "memoria": contexto_memoria,
                "regras": regras,
                "acao": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 3. SALVA MEMÓRIA
            output_formatado = f"{resposta.narracao}\nOpções: {', '.join(resposta.opcoes)}"
            self.memory.adicionar_turno(user_input, output_formatado)
            self.salvar_progresso()

            # 4. LOG DE SUCESSO NO JSONL
            metrics.update({
                "success": True,
                "latency_s": round(time.time() - start_time, 2),
                "retrieved_docs": len(docs_scores),
                "filtered_docs": len(context_parts),
                "tokens_injected": tokens_injetados
            })
            metrics_logger.info(json.dumps(metrics, ensure_ascii=False))

                    
                    # 👉 NOVO: Pega o dicionário seguro do Pydantic e injeta o contexto do PDF para o Juiz!
            resultado_final = resposta.model_dump()
            resultado_final["contexto_usado"] = contexto_lore
                    
            return resultado_final
        except Exception as e:
            # LOG DE ERRO NO JSONL
            metrics.update({
                "success": False,
                "error": str(e),
                "latency_s": round(time.time() - start_time, 2)
            })
            metrics_logger.info(json.dumps(metrics, ensure_ascii=False))
            logger.error(f"Erro no Motor RAG: {e}")
            
            return {
                "narracao": "Houve um distúrbio na magia do mundo. O Google está sobrecarregado.",
                "opcoes": ["Tentar novamente", "Esperar"]
            }

    # --- GERENCIAMENTO DE SAVE (Sua lógica original) ---
    def salvar_progresso(self):
        dados_save = {
            "resumo_geral": self.memory.resumo_geral,
            "historico_recente": self.memory.historico_recente
        }
        with open(self.cfg.save_path, 'w', encoding='utf-8') as f:
            json.dump(dados_save, f, ensure_ascii=False, indent=4)

    def carregar_progresso(self):
        if self.cfg.save_path.exists():
            with open(self.cfg.save_path, 'r', encoding='utf-8') as f:
                dados_save = json.load(f)
                self.memory.resumo_geral = dados_save.get("resumo_geral", "")
                self.memory.historico_recente = dados_save.get("historico_recente", [])