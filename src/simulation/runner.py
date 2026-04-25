import os
import sys
import json
import random
import time
import asyncio
from pathlib import Path

# Hack para encontrar a pasta core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine2 import RAGEngine, ExpConfig

async def executar_benchmark(provedor_teste: str = "google"):
    # 1. Carrega o cenário usando Pathlib
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    caminho_cenario = PROJECT_ROOT / "datasets" / "scenarios.json"
    caminho_transcript = PROJECT_ROOT / "db" / "benchmark_results" / "transcript"
    caminho_transcript.mkdir(parents=True, exist_ok=True)
    with open(caminho_cenario, 'r', encoding='utf-8') as f:
        cenarios = json.load(f)
    
    cenario_atual = cenarios[0]
    
    # =========================================================================
    # 🧠 CONFIGURAÇÃO DA BATERIA DE TESTES (O "Piloto" do seu TCC)
    # =========================================================================
    # Escolha quem vai ser testado: "google", "openai" ou "openrouter"
    
    config_teste = ExpConfig(provedor=provedor_teste)
    # =========================================================================

    print("\n" + "="*50)
    print(f"🚀 INICIANDO RPGBENCH: {cenario_atual['titulo']}")
    print(f"🤖 Motor Avaliado: {config_teste.llm_mestre} (via {provedor_teste.upper()})")
    print("="*50)
    
    # Extrai as regras e dicas do JSON e formata como texto
    regras_lista = cenario_atual.get('regras_narrativas', [])
    regras_cena = "\n".join([f"- {regra}" for regra in regras_lista])
    
    dicas_rag = " ".join(cenario_atual.get('contexto_rag_hint', []))
    
    # 2. Inicializa o Motor PASSANDO A CONFIGURAÇÃO
    engine = RAGEngine(config=config_teste)
    transcript = []
    
    print("\n[ROBÔ]: Solicitando introdução baseada no cenário...")
    
    # O "Chute Inicial" embutido com as dicas ocultas
    acao_atual = f"{cenario_atual['prompt_inicial']} [Contexto oculto para a busca: {dicas_rag}]"
    
    for turno in range(1, cenario_atual["turnos_maximos"] + 1):
        print(f"\n🎬 TURNO {turno}/{cenario_atual['turnos_maximos']}")
        
        resultado = await engine.gerar_turno_async(user_input=acao_atual, regras=regras_cena)
        
        print(f"\n📜 MESTRE:\n{resultado.get('narracao', 'Erro na narração')}")
        print("\nOPÇÕES DADAS:")
        opcoes = resultado.get('opcoes', [])
        for i, opt in enumerate(opcoes):
            print(f"[{i+1}] {opt}")
            
        acao_deste_turno = acao_atual
            
        if opcoes:
            escolha_idx = random.randint(0, len(opcoes) - 1)
            acao_atual = opcoes[escolha_idx] 
        else:
            acao_atual = "Eu continuo explorando com cautela."
            
        print(f"\n🤖 [ROBÔ ESCOLHEU]: {acao_atual}")
        
        transcript.append({
            "turno": turno,
            "acao_solicitada": acao_deste_turno,
            "resposta_mestre": resultado
        })
        
        if turno < cenario_atual["turnos_maximos"]:
            print("⏳ Aguardando 10s para estabilizar a API...")
            await asyncio.sleep(10)

    # 4. Salva o resultado da simulação
    caminho_transcript.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    # Opcional: Adiciona o nome do modelo no arquivo para ficar mais fácil de achar!
    nome_modelo_limpo = config_teste.llm_mestre.replace("/", "_")
    arquivo_saida = caminho_transcript / f"transcript_{cenario_atual['id']}_{nome_modelo_limpo}_{timestamp}.json"
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
        
    print(f"\n🏁 Simulação concluída! Transcript salvo em: {arquivo_saida}")

if __name__ == "__main__":
    
    provedores_para_testar = ["google","openrouter"]
    
    for provedor in provedores_para_testar:
        print(f"\n{'#'*60}")
        print(f"🔥 INICIANDO BATERIA DE TESTES PARA O PROVEDOR: {provedor.upper()}")
        print(f"{'#'*60}\n")
        
        try:
            # Chama a função passando o provedor da vez
            asyncio.run(executar_benchmark(provedor_teste=provedor))
            
            # Dá um respiro para a API não bloquear a sua chave por spam
            print(f"✅ Teste com {provedor} finalizado. Pausando 15s antes do próximo...")
            time.sleep(15) 
            
        except Exception as e:
            print(f"❌ Erro ao testar o provedor {provedor}: {e}")
            continue # Se um der erro, ele não para o script, apenas pula para o próximo!
            
    print("\n🎉 TODAS AS BATERIAS DE TESTE FORAM CONCLUÍDAS COM SUCESSO!")