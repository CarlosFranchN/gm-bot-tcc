import os
import json
import asyncio
import logging
import time
import tiktoken
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from google.api_core.exceptions import ResourceExhausted, RetryError, GoogleAPIError

# =============================================================================
# 1. CONFIGURAÇÃO DE EXPERIMENTO (Reprodutibilidade)
# =============================================================================
@dataclass
class ExpConfig:
    """Configuração centralizada para fácil iteração em benchmarks."""
    # Modelos
    # llm_model: str = "gemini-2.5-flash-lite"
    use_mock: bool = False
    llm_model: str = "gemini-3.1-flash-lite-preview"
    embedding_model: str = "gemini-embedding-001"
    temperature: float = 0.8
    
    # RAG / Retrieval
    retrieval_k: int = 10           # Busca mais para ter opção de filtrar
    similarity_threshold: float = 0.65  # Score máximo (distância) para aceitar doc
    max_lore_tokens: int = 1500     # Orçamento rígido para contexto externo
    
    # Paths
    db_path: Path = Path("db/chroma_dnd")
    log_file: Path = Path("logs/experiment.jsonl")
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExpConfig":
        import yaml
        from pathlib import Path

        yaml_path = Path(path).resolve()
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # 🛠️ Debug: mostra o que foi carregado
        print(f"📂 YAML carregado: {raw_data}")

        # Resolve paths relativos à raiz do projeto
        project_root = yaml_path.parent
        while not (project_root / "requirements.txt").exists():
            project_root = project_root.parent
            if project_root == project_root.root:
                break

        for key in ["db_path", "log_file"]:
            if key in raw_data and isinstance(raw_data[key], str):
                p = Path(raw_data[key])
                if not p.is_absolute():
                    raw_data[key] = project_root / p

        # Garante que apenas campos válidos da dataclass sejam passados
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in raw_data.items() if k in valid_fields}

        return cls(**filtered_data)

# =============================================================================
# 2. SCHEMA DE SAÍDA (Garante parse 100% para avaliação automática)
# =============================================================================
class TurnoRPG(BaseModel):
    raciocinio_estado: str = Field(description="Planejamento interno do mestre")
    narracao: str = Field(description="Narração imersiva da cena")
    opcoes: List[str] = Field(min_length=1, max_length=3, description="1 a 3 opções de ação")

# =============================================================================
# 3. LOGGER DE MÉTRICAS (Formato JSONL para Pandas)
# =============================================================================
# Configura logger para escrever JSON puro, uma entrada por linha
metrics_logger = logging.getLogger("benchmark")
metrics_logger.setLevel(logging.INFO)
# Remove handlers existentes para evitar duplicação em notebooks/reloads
if not metrics_logger.handlers:
    handler = logging.FileHandler("experiment_metrics.jsonl", encoding="utf-8", mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    metrics_logger.addHandler(handler)

def log_metric(**kwargs):
    """Loga uma métrica de experimento em formato JSONL."""
    metrics_logger.info(json.dumps(kwargs, ensure_ascii=False))

# Logger padrão para debug do sistema
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RAGEngine")

# =============================================================================
# 4. ENGINE PRINCIPAL
# =============================================================================
class RAGEngine:
    def __init__(self, config: ExpConfig):
        self.cfg = config
        self.cfg.db_path.mkdir(parents=True, exist_ok=True)
        load_dotenv() 
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
            "❌ GOOGLE_API_KEY não encontrada!\n"
            "1. Crie um arquivo .env na raiz do projeto\n"
            "2. Adicione: GOOGLE_API_KEY=sua_chaqui_aqui\n"
            "3. Ou defina a variável de ambiente manualmente"
            )
        # Inicializa tokenizador para controle preciso de contexto
        try:
            self.tokenizer = tiktoken.encoding_for_model("gemini-1.5-flash") # Aproximação segura para Gemini
        except KeyError:
            logger.warning("Tokenizer específico não encontrado, usando cl100k_base como fallback.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Inicializa dependências (LangChain)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.embedding_model, 
            google_api_key=api_key  # ← Passagem explícita
        )
        self.vectorstore = Chroma(persist_directory=str(config.db_path), embedding_function=self.embeddings)
        self.llm = ChatGoogleGenerativeAI(
            model=config.llm_model,
            temperature=config.temperature,
            google_api_key=api_key,
            max_retries=1  
        ) 
        
        # Parser e Chain
        self.parser = PydanticOutputParser(pydantic_object=TurnoRPG)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um Mestre de RPG SOLO.\nLORE: {lore}\nREGRAS: {regras}\n{format_instructions}"),
            ("human", "Ação do jogador: {acao}")
        ])
        # Chain montada: Prompt -> LLM -> Parser (validação Pydantic)
        self.chain = self.prompt | self.llm | self.parser
        
        # Estado simples da sessão (para MVP)
        self.memory_resumo = ""
        self.memory_recente: List[Dict] = []
        
        logger.info(f"Engine inicializada com config: k={config.retrieval_k}, threshold={config.similarity_threshold}")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def gerar_turno(self, user_input: str, regras: str = "", session_id: str = "default") -> Dict[str, Any]:
        """
        Gera um turno com métricas de retriever e tratamento de erro para benchmark.
        """
        start_time = time.time()
        # Dicionário base de métricas para este turno
        metrics = {
            "session_id": session_id, 
            "model": self.cfg.llm_model, 
            "action": user_input,
            # Converte Paths para string explicitamente
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()}
        }
        
        if self.cfg.use_mock:
            await asyncio.sleep(0.3)
            logger.info("🎭 Usando resposta MOCK (use_mock=True)")
            
            metrics.update({
                "success": True, "mock": True, "latency_s": 0.3,
                "tokens_injected": 0, "retrieved_docs": 0
            })
            log_metric(**metrics)
            
            # ✅ Retorna estrutura idêntica à API real
            return {
                "raciocinio_estado": "Mock mode: testando pipeline sem API",
                "narracao": f"*[MOCK]* Você: \"{user_input}\". O mundo reage de forma previsível para testes.",
                "opcoes": ["Opção de teste A", "Opção de teste B", "Opção de teste C"],
                "mock": True,
                "success": True,
                "latency_s": 0.3,
                "tokens_injected": 0
            }

        try:
            # 1. RETRIEVAL COM SCORE (Essencial para medir qualidade do RAG)
            docs_scores = self.vectorstore.similarity_search_with_score(user_input, k=self.cfg.retrieval_k)
            
            # 2. FILTRAGEM + TOKEN BUDGET (Evita ruído e estouro de contexto)
            context_parts = []
            tokens_injetados = 0
            
            for doc, score in docs_scores:
                # Filtra por similaridade (score é distância: menor = melhor)
                if score > self.cfg.similarity_threshold:
                    continue 
                
                doc_tokens = self._count_tokens(doc.page_content)
                if tokens_injetados + doc_tokens > self.cfg.max_lore_tokens:
                    break  # Orçamento atingido
                
                context_parts.append(doc.page_content)
                tokens_injetados += doc_tokens
            
            contexto_lore = "\n\n".join(context_parts)
            
            # 3. GERAÇÃO COM CHAIN
            resposta: TurnoRPG = await self.chain.ainvoke({
                "lore": contexto_lore, "regras": regras, "acao": user_input,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # 4. ATUALIZA MEMÓRIA (Simples, para MVP)
            self.memory_recente.append({"user": user_input, "response": resposta.narracao})
            if len(self.memory_recente) > 5: self.memory_recente.pop(0)
            
            # Sucesso: loga métricas de desempenho e retriever
            metrics.update({
                "success": True, "latency_s": round(time.time() - start_time, 3),
                "retrieved_docs": len(docs_scores), "filtered_docs": len(context_parts),
                "tokens_injected": tokens_injetados, "output_tokens": self._count_tokens(resposta.model_dump_json()),
                "similarity_scores": [round(float(s), 3) for _, s in docs_scores[:5]]
            })
            return resposta.model_dump()

        # Tratamento de Erros Específico para Pesquisa
        except ValidationError as e:
            # Erro de schema Pydantic: o LLM gerou JSON inválido
            metrics.update({"success": False, "error_type": "parse_error", "error": str(e)})
            logger.warning(f"Falha de parsing (Schema): {e}")
            return {"narracao": "Erro de formatação na resposta.", "opcoes": ["Repetir"], "error": True}
            
        except asyncio.CancelledError:
            metrics.update({"success": False, "error_type": "timeout_cancelled", "error": "Request cancelled"})
            logger.warning("⏱️  Request cancelado após retries.")
            return {"narracao": "Conexão interrompida. Tente novamente.", "opcoes": ["Repetir"], "error": True}
            
        except TimeoutError:
            metrics.update({"success": False, "error_type": "timeout", "error": "Excedeu tempo limite"})
            logger.warning("⏰ Timeout na geração.")
            return {"narracao": "Resposta demorou demais. Tente novamente.", "opcoes": ["Repetir"], "error": True}
            
        except (ResourceExhausted, RetryError) as e:
            # Erro de API Google (Quota/Rate Limit)
            metrics.update({"success": False, "error_type": "api_quota", "error": str(e)})
            logger.warning(f"Rate limit ou quota excedida: {e}")
            await asyncio.sleep(2)  # Backoff mínimo
            return {"narracao": "O oráculo está sobrecarregado. Tente novamente.", "opcoes": ["Aguardar"], "error": True}
            
        except GoogleAPIError as e:
            # Outros erros da API (Auth, Modelo não encontrado)
            metrics.update({"success": False, "error_type": "api_error", "error": str(e)})
            logger.error(f"Erro na API Google: {e}")
            return {"narracao": "Falha de conexão com o Mestre.", "opcoes": ["Sair"], "error": True}
            
        except Exception as e:
            # Erro genérico (não deve acontecer se o código estiver maduro)
            metrics.update({"success": False, "error_type": "unknown", "error": f"{type(e).__name__}: {e}"})
            logger.exception(f"Erro inesperado: {e}")
            return {"narracao": "Erro crítico no sistema.", "opcoes": ["Reiniciar"], "error": True}
            
        finally:
            # Garante que TUDO (sucesso ou falha) seja logado para análise posterior
            log_metric(**metrics)

    def salvar_estado(self, path: Optional[Path] = None):
        """Persiste o estado simples da sessão (útil para debug)."""
        path = path or Path("db/state.json")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"resumo": self.memory_resumo, "recente": self.memory_recente}, f, ensure_ascii=False, indent=2)
            
# =============================================================================
# BLOCO DE TESTE - Adicione ao final do engine.py
# =============================================================================
if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse de argumentos para flexibilidade em CLI
    parser = argparse.ArgumentParser(description="Testar RAGEngine para RPG Solo")
    parser.add_argument("--mode", choices=["interactive", "benchmark", "quick"], default="interactive",
                        help="Modo de execução: interactive (terminal), benchmark (dataset), quick (teste único)")
    parser.add_argument("--config", type=str, default=None, help="Caminho para arquivo YAML de configuração")
    parser.add_argument("--actions", type=str, nargs="+", default=None, help="Lista de ações para modo benchmark")
    parser.add_argument("--session", type=str, default="test_session", help="ID da sessão para logs")
    args = parser.parse_args()

    # Configuração inicial
    config = ExpConfig.from_yaml(args.config) if args.config and Path(args.config).exists() else ExpConfig()
    print(f"🎲 RAGEngine - Modo: {args.mode} | Config: temp={config.temperature}, k={config.retrieval_k}")
    
    async def run_interactive():
        """Modo terminal interativo - ideal para desenvolvimento e testes manuais."""
        engine = RAGEngine(config)
        print("\n" + "="*60)
        print("🎮 MODO INTERATIVO - Digite suas ações (ou 'sair' para encerrar)")
        print("="*60 + "\n")
        
        turnos = 0
        while True:
            try:
                user_input = input(f"\n[{turnos+1}] Você: ").strip()
                if user_input.lower() in ["sair", "exit", "quit", "/q"]:
                    print("\n💾 Salvando estado e encerrando...")
                    engine.salvar_estado()
                    break
                if not user_input:
                    continue
                
                print("🎲 Mestre pensando...", end="\r")
                resultado = await engine.gerar_turno(user_input, session_id=args.session)
                
                # Exibe resultado formatado
                print(f"\n📜 {resultado['narracao']}")
                if "opcoes" in resultado and resultado["opcoes"]:
                    print("\n✨ O que você faz?")
                    for i, op in enumerate(resultado["opcoes"], 1):
                        print(f"   [{i}] {op}")
                
                turnos += 1
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrompido pelo usuário. Salvando...")
                engine.salvar_estado()
                break
            except EOFError:
                break

    async def run_benchmark():
        """Modo benchmark - roda um conjunto de ações e gera relatório."""
        # Dataset de exemplo (substitua pelo seu dataset real)
        actions = args.actions or [
            "Entro na taverna escura e observo os frequentadores",
            "Ataco o goblin com minha espada",
            "Examinar o altar antigo em busca de pistas",
            "Converso com o tavernista sobre rumores da região",
            "Uso magia de detecção no corredor à frente",
            "Procuro por armadilhas na porta",
            "Descanso por uma hora para recuperar energias",
            "Abro o baú encontrado no canto da sala"
        ]
        
        engine = RAGEngine(config)
        print(f"\n🧪 MODO BENCHMARK - {len(actions)} ações | Session: {args.session}")
        print("-"*60)
        
        results = []
        start_global = time.time()
        
        for i, acao in enumerate(actions, 1):
            print(f"[{i}/{len(actions)}] Testando: {acao[:50]}{'...' if len(acao)>50 else ''}")
            
            resultado = await engine.gerar_turno(acao, session_id=args.session)
            results.append({
                "action": acao, 
                "success": resultado.get("success", True),
                "narracao_preview": resultado.get("narracao", "")[:100]
            })
            
            # Feedback visual imediato
            status = "✅" if resultado.get("success", False) else "❌"
            print(f"       {status} Latência: {resultado.get('latency_s', 'N/A')}s | Tokens: {resultado.get('tokens_injected', 'N/A')}")
        
        total_time = time.time() - start_global
        
        # Relatório consolidado no terminal
        print("\n" + "="*60)
        print("📊 RELATÓRIO DO BENCHMARK")
        print("="*60)
        
        # Lê os logs gerados para estatísticas
        try:
            import pandas as pd
            df = pd.read_json("experiment_metrics.jsonl", lines=True)
            df_session = df[df["session_id"] == args.session] if "session_id" in df.columns else df
            
            if not df_session.empty:
                print(f"\n✅ Taxa de Sucesso: {(df_session['success'].mean()*100):.1f}%")
                print(f"⏱️  Latência Média: {df_session['latency_s'].mean():.2f}s (±{df_session['latency_s'].std():.2f})")
                print(f"📦 Tokens Injetados (média): {df_session['tokens_injected'].mean():.0f}")
                print(f"🔍 Docs Recuperados (média): {df_session['retrieved_docs'].mean():.1f}")
                print(f"🗑️  Docs Filtrados (média): {df_session['filtered_docs'].mean():.1f}")
                
                # Tabela de erros, se houver
                if "error_type" in df_session.columns and df_session["error_type"].notna().any():
                    print(f"\n⚠️  Erros por tipo:")
                    print(df_session[df_session["success"]==False]["error_type"].value_counts().to_string())
            else:
                print("⚠️  Nenhum dado encontrado nos logs para esta sessão.")
                
        except ImportError:
            print("💡 Dica: instale pandas para ver estatísticas avançadas: pip install pandas")
        except Exception as e:
            print(f"⚠️  Não foi possível gerar relatório: {e}")
        
        print(f"\n🕐 Tempo total: {total_time:.2f}s | {len(actions)} ações")
        print(f"📁 Logs salvos em: experiment_metrics.jsonl")
        print("="*60)

    async def run_quick_test():
        """Teste rápido de sanity check - uma única ação."""
        engine = RAGEngine(config)
        test_action = "Eu entro na taverna e peço uma cerveja"
        
        print(f"\n⚡ MODO RÁPIDO - Teste único")
        print(f"Ação: {test_action}\n")
        
        try:
            resultado = await engine.gerar_turno(test_action, session_id=args.session)
            
            if resultado.get("success", True):
                print("✅ Sucesso!\n")
                print(f"📜 Narração:\n{resultado['narracao']}\n")
                if resultado.get("opcoes"):
                    print("✨ Opções sugeridas:")
                    for op in resultado["opcoes"]:
                        print(f"   • {op}")
                print(f"\n📊 Métricas: {resultado.get('latency_s', 'N/A')}s | {resultado.get('tokens_injected', 'N/A')} tokens injetados")
            else:
                print(f"❌ Falha: {resultado.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"💥 Erro crítico: {e}")
            import traceback
            traceback.print_exc()

    # Dispatcher dos modos
    try:
        if args.mode == "interactive":
            asyncio.run(run_interactive())
        elif args.mode == "benchmark":
            asyncio.run(run_benchmark())
        elif args.mode == "quick":
            asyncio.run(run_quick_test())
    except KeyboardInterrupt:
        print("\n\n🛑 Encerramento solicitado pelo usuário.")
    except Exception as e:
        logger.exception(f"Erro não tratado no main: {e}")
        sys.exit(1)