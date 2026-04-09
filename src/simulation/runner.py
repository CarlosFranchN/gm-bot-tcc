import os
import sys
import json
import random
import time

# Hack para encontrar a pasta core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import RAGEngine

def executar_benchmark():
    # 1. Carrega o cenário
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_cenario = os.path.join(base_dir, "datasets", "scenarios.json")
    
    with open(caminho_cenario, 'r', encoding='utf-8') as f:
        cenarios = json.load(f)
    
    cenario_atual = cenarios[0]
    
    print("\n" + "="*50)
    print(f"🚀 INICIANDO RPGBENCH: {cenario_atual['titulo']}")
    print("="*50)
    
    # 👉 NOVO: Extrai as regras e dicas do JSON e formata como texto
    regras_lista = cenario_atual.get('regras_narrativas', [])
    regras_cena = "\n".join([f"- {regra}" for regra in regras_lista])
    
    dicas_rag = " ".join(cenario_atual.get('contexto_rag_hint', []))
    
    # 2. Inicializa o Motor
    engine = RAGEngine()
    transcript = []
    
    print("\n[ROBÔ]: Solicitando introdução baseada no cenário...")
    
    # 👉 NOVO: O "Chute Inicial" agora embute as dicas ocultas para o RAG achar o PDF de primeira!
    acao_atual = f"{cenario_atual['prompt_inicial']} [Contexto oculto para a busca: {dicas_rag}]"
    
    for turno in range(1, cenario_atual["turnos_maximos"] + 1):
        print(f"\n🎬 TURNO {turno}/{cenario_atual['turnos_maximos']}")
        
        # 👉 NOVO: Passamos as 'regras_cena' para o Motor respeitar a Direção de Arte
        resultado = engine.gerar_turno(acao_atual, regras_especificas=regras_cena)
        
        print(f"\n📜 MESTRE:\n{resultado['narracao']}")
        print("\nOPÇÕES DADAS:")
        for i, opt in enumerate(resultado['opcoes']):
            print(f"[{i+1}] {opt}")
            
# 👉 1. FOTOGRAFA a ação que gerou a resposta deste turno
        acao_deste_turno = acao_atual
            
        # 👉 2. O Robô escolhe uma opção aleatória PARA O PRÓXIMO TURNO
        if resultado['opcoes']:
            escolha_idx = random.randint(0, len(resultado['opcoes']) - 1)
            acao_atual = resultado['opcoes'][escolha_idx] 
        else:
            acao_atual = "Eu continuo explorando com cautela."
            
        print(f"\n🤖 [ROBÔ ESCOLHEU]: {acao_atual}")
        
        # 👉 3. Salva no log a 'acao_deste_turno' (a correta!) com a resposta certa
        transcript.append({
            "turno": turno,
            "acao_solicitada": acao_deste_turno,
            "resposta_mestre": resultado
        })
        
        # Pausa de segurança para não estourar a cota da API gratuita
        time.sleep(10)

    # 4. Salva o resultado da simulação
    caminho_transcript = os.path.join(os.path.dirname(base_dir), "db", "benchmark_results")
    os.makedirs(caminho_transcript, exist_ok=True)
    
    arquivo_saida = os.path.join(caminho_transcript, f"transcript_{cenario_atual['id']}.json")
    with open(arquivo_saida, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=4)
        
    print(f"\n🏁 Simulação concluída! Transcript salvo em: {arquivo_saida}")

if __name__ == "__main__":
    executar_benchmark()