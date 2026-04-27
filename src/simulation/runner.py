import os
import sys
import json
import random
import time
import asyncio
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine2 import RAGEngine, ExpConfig

# =========================================================================
# 1. A MENTE DO JOGADOR (Fácil de escalar no futuro)
# =========================================================================
class RandomBot:
    """Um robô que joga RPG escolhendo opções aleatoriamente."""
    def __init__(self, fallback_action="Eu continuo explorando com cautela."):
        self.fallback_action = fallback_action

    def escolher_acao(self, opcoes: list) -> str:
        if opcoes:
            return random.choice(opcoes)
        return self.fallback_action

# =========================================================================
# 2. O TABULEIRO (Apenas o Loop do Jogo)
# =========================================================================
async def jogar_partida(engine: RAGEngine, bot: RandomBot, cenario: dict) -> list:
    """Realiza o ping-pong entre o Motor (Mestre) e o Bot (Jogador)."""
    transcript = []
    
    regras_cena = "\n".join([f"- {r}" for r in cenario.get('regras_narrativas', [])])
    dicas_rag = " ".join(cenario.get('contexto_rag_hint', []))
    
    acao_atual = f"{cenario['prompt_inicial']} [Contexto oculto: {dicas_rag}]"
    
    for turno in range(1, cenario["turnos_maximos"] + 1):
        print(f"\n🎬 TURNO {turno}/{cenario['turnos_maximos']}")
        
        # 1. O Mestre Narra
        resultado = await engine.gerar_turno_async(user_input=acao_atual, regras=regras_cena)
        
        print(f"📜 MESTRE: {resultado.get('narracao', '')[:100]}...") # Print curto para não poluir a tela
        
        # 2. O Jogador Escolhe
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
        
        if turno < cenario["turnos_maximos"]:
            await asyncio.sleep(10) # Respiro da API
            
    return transcript

# =========================================================================
# 3. O GERENTE DO ARQUIVO (Apenas Salva Dados)
# =========================================================================
def _salvar_transcript(transcript: list, cenario_id: str, config: ExpConfig, base_dir: Path, repeticao: int = 1):
    """Lida com a burocracia de criar pastas e gerar o nome do arquivo JSON."""
    pasta_saida = base_dir / "transcript"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    modelo_limpo = config.llm_mestre.replace("/", "_")
    modo_tag = "rag" if config.usar_rag else "baseline"
    
    nome_arquivo = f"transcript_{cenario_id}_{modelo_limpo}_{modo_tag}_rep{repeticao}_{timestamp}.json"
    arquivo_saida = pasta_saida / nome_arquivo
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
        
    print(f"🏁 Transcript salvo em: {arquivo_saida.name}")

# =========================================================================
# 4. O ORQUESTRADOR PRINCIPAL (A antiga Função Deus)
# =========================================================================
async def executar_benchmark(provedor_teste: str = "google", usar_rag: bool = True, repeticoes: int = 1):
    # 1. Carrega os dados iniciais (Apenas uma vez fora do loop)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    with open(PROJECT_ROOT / "datasets" / "scenarios.json", 'r', encoding='utf-8') as f:
        cenarios = json.load(f)
    
    cenario_atual = cenarios[0]
    base_dir_results = PROJECT_ROOT / "db" / "benchmark_results"

    # Inicia o loop de repetições
    for i in range(repeticoes):
        rep_atual = i + 1
        print(f"\n🔄 REPETIÇÃO {rep_atual}/{repeticoes} - {provedor_teste.upper()} (RAG={usar_rag})")
        
        # 2. Prepara os "Atores" para esta rodada específica
        config = ExpConfig(provedor=provedor_teste, usar_rag=usar_rag)
        engine = RAGEngine(config=config)
        bot = RandomBot()
        
        print(f"🚀 JOGANDO: {cenario_atual['titulo']} | Modelo: {config.llm_mestre}")
        
        # 3. Deixa eles jogarem!
        transcript_final = await jogar_partida(engine, bot, cenario_atual)
        
        # 👉 4. SALVAMENTO IMEDIATO (Dentro do loop para segurança)
        # Passamos o 'rep_atual' para que a função de salvar crie nomes únicos
        _salvar_transcript(
            transcript_final, 
            cenario_atual['id'], 
            config, 
            base_dir_results,
            repeticao=rep_atual # Novo parâmetro para o nome do arquivo
        )

        # Respiro entre repetições
        if i < repeticoes - 1:
            print(f"⏳ Pausa de segurança de 20s entre repetições...")
            await asyncio.sleep(20)

    print(f"\n✅ Bateria de {repeticoes} repetições concluída para {provedor_teste}!")
# =========================================================================
# GATILHO DA MATRIZ DE ABLAÇÃO
# =========================================================================
if __name__ == "__main__":
    provedores = ["google", "openrouter"]
    modos_rag = [True, False] 
    
    for provedor in provedores:
        for rag_ligado in modos_rag:
            try:
                asyncio.run(executar_benchmark(provedor_teste=provedor, usar_rag=rag_ligado))
                time.sleep(15) 
            except Exception as e:
                print(f"❌ Erro ao testar {provedor} (RAG={rag_ligado}): {e}")