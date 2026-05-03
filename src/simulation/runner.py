import os
import sys
import json
import random
import time
import asyncio
from pathlib import Path

# Ajuste de path para importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine2 import RAGEngine, ExpConfig

# =========================================================================
# 1. A MENTE DO JOGADOR
# =========================================================================
class RandomBot:
    """Um robô que joga RPG escolhendo opções aleatoriamente."""
    def __init__(self, fallback_action="Eu continuo explorando com cautela."):
        self.fallback_action = fallback_action

    def escolher_acao(self, opcoes: list) -> str:
        """Escolhe uma opção aleatória da lista ou a ação de fallback."""
        if opcoes:
            return random.choice(opcoes)
        return self.fallback_action


# =========================================================================
# 2. O ORQUESTRADOR DE SIMULAÇÃO
# =========================================================================
class SimulationOrchestrator:
    """Orquestra a configuração, a simulação das partidas e o armazenamento dos logs."""
    
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.base_dir_results = self.project_root / "db" / "benchmark_results"
        self.cenarios = self._carregar_cenarios()
        
    def _carregar_cenarios(self) -> list:
        """Carrega a lista de cenários do ficheiro JSON."""
        caminho_scenarios = self.project_root / "datasets" / "scenarios.json"
        with open(caminho_scenarios, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def _jogar_partida(self, engine: RAGEngine, bot: RandomBot, cenario: dict) -> list:
        """Realiza o ping-pong de turnos entre o Motor (Mestre) e o Bot (Jogador)."""
        transcript = []
        
        regras_cena = "\n".join([f"- {r}" for r in cenario.get('regras_narrativas', [])])
        dicas_rag = " ".join(cenario.get('contexto_rag_hint', []))
        
        acao_atual = f"{cenario['prompt_inicial']} [Contexto oculto: {dicas_rag}]"
        
        turnos_maximos = cenario.get("turnos_maximos", 5)
        
        for turno in range(1, turnos_maximos + 1):
            print(f"\n🎬 TURNO {turno}/{turnos_maximos}")
            
            # 1. O Mestre Narra (Com tratamento de excepções e resiliência)
            try:
                resultado = await engine.gerar_turno_async(user_input=acao_atual, regras=regras_cena)
            except Exception as e:
                print(f"❌ Erro fatal na geração do turno {turno}: {e}")
                # Regista o erro no transcript e aborta a simulação atual para não gerar lixo
                transcript.append({"turno": turno, "erro": str(e), "acao_solicitada": acao_atual})
                break 

            # Feedback visual no terminal
            narracao_curta = resultado.get('narracao', '')[:100].replace('\n', ' ')
            print(f"📜 MESTRE: {narracao_curta}...")
            
            # 2. O Jogador Escolhe a próxima ação
            acao_deste_turno = acao_atual
            opcoes = resultado.get('opcoes', [])
            acao_atual = bot.escolher_acao(opcoes)
            
            print(f"🤖 ROBÔ ESCOLHEU: {acao_atual}")
            
            # 3. Salva no Diário
            transcript.append({
                "turno": turno,
                "acao_solicitada": acao_deste_turno,
                "resposta_mestre": resultado
            })
            
            # Pausa para evitar rate limits da API entre turnos
            if turno < turnos_maximos:
                await asyncio.sleep(35) 
                
        return transcript

    def _salvar_transcript(self, transcript: list, cenario_id: str, config: ExpConfig, repeticao: int):
        """Guarda o log da partida em ficheiro JSON formatado."""
        pasta_saida = self.base_dir_results / "transcript"
        pasta_saida.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        modelo_limpo = config.llm_mestre.replace("/", "_").replace(":", "-")
        modo_tag = "rag" if config.usar_rag else "baseline"
        
        nome_arquivo = f"transcript_{cenario_id}_{modelo_limpo}_{modo_tag}_rep{repeticao}_{timestamp}.json"
        arquivo_saida = pasta_saida / nome_arquivo
        
        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=4)
            
        print(f"🏁 Transcript salvo em: {arquivo_saida.name}")

    async def executar_benchmark(self, provedor_teste: str = "google", modelo_teste: str = "", usar_rag: bool = True, repeticoes: int = 1):
        """Orquestra as múltiplas repetições de um cenário para uma configuração específica."""
        
        if not self.cenarios:
             print("❌ Erro: Nenhum cenário encontrado para jogar.")
             return
             
        cenario_atual = self.cenarios[0] # Para já testa apenas o primeiro cenário

        for i in range(repeticoes):
            rep_atual = i + 1
            print(f"\n========================================================")
            print(f"🔄 REPETIÇÃO {rep_atual}/{repeticoes} | PROVEDOR: {provedor_teste.upper()} | MODELO: {modelo_teste} | RAG: {usar_rag}")
            print(f"========================================================")
            
            # 👉 AQUI ESTÁ A CORREÇÃO CRÍTICA!
            config = ExpConfig(
                provedor=provedor_teste, 
                llm_mestre=modelo_teste, # Força o modelo que veio da matriz
                usar_rag=usar_rag
            )
            
            if config.save_path.exists():
                config.save_path.unlink()
                
            engine = RAGEngine(config=config)
            bot = RandomBot()
            
            print(f"🚀 INICIANDO: {cenario_atual['titulo']} | Modelo alvo: {config.llm_mestre}")
            
            # Joga a partida e grava os resultados
            transcript_final = await self._jogar_partida(engine, bot, cenario_atual)
            
            self._salvar_transcript(transcript_final, cenario_atual['id'], config, rep_atual)

            # Pausa de segurança entre repetições do mesmo cenário
            if i < repeticoes - 1:
                print(f"⏳ Pausa de segurança (20s) a aguardar arrefecimento da API...")
                await asyncio.sleep(20)

        print(f"\n✅ Bateria de {repeticoes} repetições concluída com sucesso para {modelo_teste} (RAG={usar_rag})!")

# =========================================================================
# 3. GATILHO DA MATRIZ DE ABLAÇÃO
# =========================================================================
async def main():
    """Função de entrada que define a matriz de testes."""
    provedores = ["qwen/qwen-2.5-coder-32b-instruct",
                   "openai/gpt-4o-mini"]
    # provedores = ["google"]
    modos_rag = [True, False] 
    NUMERO_REPETICOES = 1
    
    orquestrador = SimulationOrchestrator()
    
    for provedor in provedores:
        for rag_ligado in modos_rag:
            try:
                await orquestrador.executar_benchmark(
                    provedor_teste="openrouter", # Tem que ser a string "openrouter"
                    modelo_teste=provedor,         # Tem que ser a variável do loop
                    usar_rag=rag_ligado, 
                    repeticoes=NUMERO_REPETICOES
                )
                print(f"⏳ Pausa de segurança (15s) entre trocas de contexto da matriz...")
                await asyncio.sleep(15) 
            except Exception as e:
                print(f"❌ Erro crítico ao testar provedor {provedor} (RAG={rag_ligado}): {e}")

if __name__ == "__main__":
    # Garante que o loop assíncrono corre de forma correta no ponto de entrada
    asyncio.run(main())