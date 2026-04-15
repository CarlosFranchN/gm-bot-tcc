import os
import sys
import json
import random
import time
import asyncio
from pathlib import Path

# Hack para encontrar a pasta core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine2 import RAGEngine 

async def executar_benchmark():
    # 1. Carrega o cenário usando Pathlib
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    caminho_cenario = PROJECT_ROOT / "datasets" / "scenarios.json"
    caminho_transcript = PROJECT_ROOT / "db" / "benchmark_results"
    
    with open(caminho_cenario, 'r', encoding='utf-8') as f:
        cenarios = json.load(f)
    
    cenario_atual = cenarios[0]
    
    print("\n" + "="*50)
    print(f"🚀 INICIANDO RPGBENCH: {cenario_atual['titulo']}")
    print("="*50)
    
    # Extrai as regras e dicas do JSON e formata como texto
    regras_lista = cenario_atual.get('regras_narrativas', [])
    regras_cena = "\n".join([f"- {regra}" for regra in regras_lista])
    
    dicas_rag = " ".join(cenario_atual.get('contexto_rag_hint', []))
    
    # 2. Inicializa o Motor
    engine = RAGEngine()
    transcript = []
    
    print("\n[ROBÔ]: Solicitando introdução baseada no cenário...")
    
    # O "Chute Inicial" agora embute as dicas ocultas para o RAG achar o PDF de primeira!
    acao_atual = f"{cenario_atual['prompt_inicial']} [Contexto oculto para a busca: {dicas_rag}]"
    
    for turno in range(1, cenario_atual["turnos_maximos"] + 1):
        print(f"\n🎬 TURNO {turno}/{cenario_atual['turnos_maximos']}")
        
        # 👉 CORREÇÃO: Usar o nome do método assíncrono correto!
        resultado = await engine.gerar_turno_async(user_input=acao_atual, regras=regras_cena)
        
        # Usando .get() para evitar crachar caso o retorno seja um erro formatado diferente
        print(f"\n📜 MESTRE:\n{resultado.get('narracao', 'Erro na narração')}")
        print("\nOPÇÕES DADAS:")
        opcoes = resultado.get('opcoes', [])
        for i, opt in enumerate(opcoes):
            print(f"[{i+1}] {opt}")
            
        # 1. FOTOGRAFA a ação que gerou a resposta deste turno
        acao_deste_turno = acao_atual
            
        # 2. O Robô escolhe uma opção aleatória PARA O PRÓXIMO TURNO
        if opcoes:
            escolha_idx = random.randint(0, len(opcoes) - 1)
            acao_atual = opcoes[escolha_idx] 
        else:
            acao_atual = "Eu continuo explorando com cautela."
            
        print(f"\n🤖 [ROBÔ ESCOLHEU]: {acao_atual}")
        
        # 3. Salva no log a 'acao_deste_turno' com a resposta certa
        transcript.append({
            "turno": turno,
            "acao_solicitada": acao_deste_turno,
            "resposta_mestre": resultado
        })
        
        # Pausa de segurança (Corrigido para o modo Assíncrono)
        if turno < cenario_atual["turnos_maximos"]:
            print("⏳ Aguardando 10s para estabilizar a API...")
            await asyncio.sleep(10)

    # 4. Salva o resultado da simulação
    caminho_transcript = PROJECT_ROOT / "db" / "benchmark_results"
    caminho_transcript.mkdir(parents=True, exist_ok=True)
    
    # 👉 CORREÇÃO: Adiciona Timestamp para não sobrescrever testes antigos!
    timestamp = int(time.time())
    arquivo_saida = caminho_transcript / f"transcript_{cenario_atual['id']}_{timestamp}.json"
    
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
        
    print(f"\n🏁 Simulação concluída! Transcript salvo em: {arquivo_saida}")

if __name__ == "__main__":
    # 👉 CORREÇÃO: Roda o loop assíncrono corretamente
    asyncio.run(executar_benchmark())