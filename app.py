import streamlit as st
import json
import os
from datetime import datetime

# ==========================================
# 1. IMPORTAÇÃO DO SEU MOTOR (ENGINE)
# ==========================================
# Adapte esta linha para importar a classe ou função principal do seu motor RAG
# Exemplo: from src.core.engine import SeuMotorDeRPG
# motor = SeuMotorDeRPG()

# ==========================================
# 2. CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="GM BOT TCC - Teste Humano", page_icon="🎲", layout="centered")

st.title("🎲 Oásis do Deserto - Teste Interativo")
st.markdown("*Ajudar a ciência nunca foi tão perigoso. Tem 5 turnos para sobreviver.*")

# ==========================================
# 3. INICIALIZAÇÃO DE VARIÁVEIS DE ESTADO
# ==========================================
if "turno_atual" not in st.session_state:
    st.session_state.turno_atual = 1
    
if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "transcript" not in st.session_state:
    st.session_state.transcript = []

if "mensagens_chat" not in st.session_state:
    # A primeira mensagem do Mestre (Introdução do cenário)
    st.session_state.mensagens_chat = [
        {"role": "assistant", "content": "Acorda com o sol escaldante de Anauroch a castigar a sua pele. A areia estende-se até ao horizonte e a sua água acabou. Ao longe, vê as ruínas de um antigo santuário de pedra. O que faz?"}
    ]

# ==========================================
# 4. FUNÇÃO PARA GUARDAR O TRANSCRIPT
# ==========================================
def guardar_transcript_humano():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "benchmark_results")
    os.makedirs(base_dir, exist_ok=True)
    
    # Gera um ID único baseado na data/hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_ficheiro = f"transcript_human_desert_{timestamp}.json"
    caminho_completo = os.path.join(base_dir, nome_ficheiro)
    
    with open(caminho_completo, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.transcript, f, ensure_ascii=False, indent=4)
        
    return caminho_completo

# ==========================================
# 5. RENDERIZAÇÃO DO ECRÃ
# ==========================================
# Mostra o painel lateral com o progresso
with st.sidebar:
    st.header("📊 Progresso do TCC")
    st.progress(st.session_state.turno_atual / 6.0) # Vai até 5
    st.write(f"**Turno Atual:** {min(st.session_state.turno_atual, 5)} / 5")
    st.info("No final do 5º turno, a sua sessão será gravada automaticamente para avaliação científica.")

# Mostra o histórico de mensagens no ecrã
for msg in st.session_state.mensagens_chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 6. LÓGICA DE INTERAÇÃO (INPUT DO JOGADOR)
# ==========================================
if not st.session_state.game_over:
    acao_jogador = st.chat_input("Escreva a sua ação aqui (ex: 'Corro para a sombra do santuário')...")

    if acao_jogador:
        # 1. Mostra a ação do jogador no ecrã
        st.session_state.mensagens_chat.append({"role": "user", "content": acao_jogador})
        with st.chat_message("user"):
            st.markdown(acao_jogador)

        # 2. Chama a IA (Simulação do Mestre)
        with st.chat_message("assistant"):
            with st.spinner("O Mestre está a consultar as regras do cenário..."):
                
                # ---------------------------------------------------------
                # ⚠️ ATENÇÃO: SUBSTITUA ISTO PELA CHAMADA REAL DO SEU MOTOR
                # resultado_ia = motor.gerar_turno(acao_jogador)
                # ---------------------------------------------------------
                
                # Mockup temporário para podermos testar a interface:
                resultado_ia = {
                    "narracao": "A estrutura de pedra oferece uma sombra fria e reconfortante. Encontra uma porta dupla entalhada e um baú coberto de pó.",
                    "opcoes": ["Abrir o baú", "Inspecionar a porta", "Procurar armadilhas na entrada"],
                    "contexto_usado": "O exterior do santuário tem uma porta dupla e um baú."
                }
                
                # Formata a resposta para exibir no ecrã de forma elegante
                opcoes_texto = "\n\n**Opções:**\n" + "\n".join([f"- {op}" for op in resultado_ia.get("opcoes", [])])
                resposta_ecra = resultado_ia["narracao"] + opcoes_texto
                
                st.markdown(resposta_ecra)
                st.session_state.mensagens_chat.append({"role": "assistant", "content": resposta_ecra})

        # 3. Regista o turno para o Tribunal (judge_creative.py) ler depois
        st.session_state.transcript.append({
            "turno": st.session_state.turno_atual,
            "acao_solicitada": acao_jogador,
            "resposta_mestre": resultado_ia # O JSON completo (narração, opções, etc.)
        })

        # 4. Avança o turno e verifica o fim do jogo
        st.session_state.turno_atual += 1
        
        if st.session_state.turno_atual > 5:
            st.session_state.game_over = True
            caminho_salvo = guardar_transcript_humano()
            st.success(f"🎉 Sessão terminada! Muito obrigado por participar no meu TCC. Ficheiro gravado com sucesso.")
            st.balloons()
            st.rerun() # Força a atualização do ecrã para bloquear o input

else:
    st.warning("O jogo terminou. Atualize a página se quiser tentar outra vez!")