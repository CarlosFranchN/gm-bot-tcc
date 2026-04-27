import os
import sys
import json
import random
import time
import asyncio
from pathlib import Path

# Certifique-se de que o nome do arquivo da engine está correto (engine ou engine2)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine2 import RAGEngine, ExpConfig

# 👉 1. ADICIONAMOS O PARÂMETRO 'usar_rag'
async def executar_benchmark(provedor_teste: str = "google", usar_rag: bool = True):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    caminho_cenario = PROJECT_ROOT / "datasets" / "scenarios.json"
    caminho_transcript = PROJECT_ROOT / "db" / "benchmark_results" / "transcript"
    caminho_transcript.mkdir(parents=True, exist_ok=True)
    
    with open(caminho_cenario, 'r', encoding='utf-8') as f:
        cenarios = json.load(f)
    
    cenario_atual = cenarios[0]
    
    # =========================================================================
    # 🧠 CONFIGURAÇÃO DA BATERIA DE TESTES
    # =========================================================================
    # 👉 2. INJETAMOS A CHAVE DE ABLAÇÃO NA CONFIGURAÇÃO
    config = ExpConfig(
        provedor=provedor_teste,
        usar_rag=usar_rag 
    )
    
    modo_texto = "COM RAG LIGADO" if usar_rag else "BASELINE (SEM RAG)"

    print("\n" + "="*60)
    print(f"🚀 INICIANDO RPGBENCH: {cenario_atual['titulo']}")
    print(f"🤖 Motor Avaliado: {config.llm_mestre} (via {provedor_teste.upper()})")
    print(f"🎛️  Modo de Teste: {modo_texto}")
    print("="*60)
    
    regras_lista = cenario_atual.get('regras_narrativas', [])
    regras_cena = "\n".join([f"- {regra}" for regra in regras_lista])
    dicas_rag = " ".join(cenario_atual.get('contexto_rag_hint', []))
    
    engine = RAGEngine(config=config)
    transcript = []
    
    print("\n[ROBÔ]: Solicitando introdução baseada no cenário...")
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

    # 👉 3. NOMENCLATURA INTELIGENTE PARA O TRIBUNAL
    timestamp = int(time.time())
    nome_modelo_limpo = config.llm_mestre.replace("/", "_")
    modo_tag = "rag" if usar_rag else "baseline"
    
    # Ex: transcript_cenario1_gpt-4o-mini_baseline_171000000.json
    arquivo_saida = caminho_transcript / f"transcript_{cenario_atual['id']}_{nome_modelo_limpo}_{modo_tag}_{timestamp}.json"
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
        
    print(f"\n🏁 Simulação concluída! Transcript salvo em: {arquivo_saida}")

if __name__ == "__main__":
    
    provedores_para_testar = ["google", "openrouter"]
    # 👉 4. A MATRIZ DE ABLAÇÃO AUTOMÁTICA
    modos_de_teste = [True, False] # True = RAG, False = Baseline
    
    for provedor in provedores_para_testar:
        for usar_rag_agora in modos_de_teste:
            try:
                asyncio.run(executar_benchmark(provedor_teste=provedor, usar_rag=usar_rag_agora))
                print(f"✅ Teste concluído. Pausando 15s antes do próximo...")
                time.sleep(15) 
            except Exception as e:
                print(f"❌ Erro ao testar {provedor} (RAG={usar_rag_agora}): {e}")
                continue 
            
    print("\n🎉 TODAS AS BATERIAS DE TESTE FORAM CONCLUÍDAS COM SUCESSO!")