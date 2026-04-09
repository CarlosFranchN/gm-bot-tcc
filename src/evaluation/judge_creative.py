import os
import sys
import json
import time
import csv
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

def configure():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    env_path = os.path.join(base_dir, ".env")
    
    print(f"Tentando carregar .env de: {env_path}") 
    
    load_dotenv(env_path)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        env_path_alt = os.path.join(script_dir, ".env")
        load_dotenv(env_path_alt)
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("❌ ERRO: GOOGLE_API_KEY não encontrada! Verifique se o arquivo .env existe e tem a chave.")
    
    os.environ["GOOGLE_API_KEY"] = api_key
    
 

class MasterJudge:
    def __init__(self):
        configure()
        self.llm_juiz = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.0)

    def avaliar_turno_completo(self, acao_jogador, narracao_mestre, opcoes_mestre, contexto_pdf, tentativas=3):
        
        opcoes_str = "\n".join([f"[{i+1}] {op}" for i, op in enumerate(opcoes_mestre)])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor avançado de Sistemas RAG e Game Design de RPG.
            Avalie a interação do Mestre com base em 4 grandes eixos, dando notas de 1 a 5.
            
            CONTEXTO ORIGINAL DO PDF (GABARITO PARA AVALIAÇÃO DE ADERÊNCIA):
            {contexto_pdf}
            
            EIXOS DE AVALIAÇÃO (Baseados nas taxonomias NarraBench e RAGBench):
            1. STYLE_REV (Style & Revelation): A narração utiliza linguagem rica e descritiva para criar suspense e imersão na cena?
            2. EVENT_CAUS (Event Causality) - Composto por 3 rubricas:
               - EVENT_CAUS_D: As opções oferecem diversidade e caminhos distintos?
               - EVENT_CAUS_R: As opções são relevantes para o estado causal da cena?
               - EVENT_CAUS_C: As opções são claras e bem escritas?
            3. ADHERENCE (Aderência ao Lore): Compare a Narração EXCLUSIVAMENTE com o CONTEXTO DO PDF acima. O Mestre foi 100% aderente, sem inventar/alucinar nomes, itens ou monstros que não estão no texto? (Nota 5 = 100% fiel; penalize invenções).
            4. TIME_ORDER (Ordem Temporal): O Mestre resolveu a ação atual do jogador cronologicamente antes de dar novas opções (evitando loops lógicos ou cliffhangers injustificados)?
            
            Retorne APENAS um JSON válido seguindo exatamente este formato:
            {{
                "STYLE_REV": float,
                "EVENT_CAUS_D": float,
                "EVENT_CAUS_R": float,
                "EVENT_CAUS_C": float,
                "ADHERENCE": float,
                "TIME_ORDER": float,
                "justificativa": "Um parágrafo curto justificando as notas. Especifique se houve alguma alucinação ou detalhe inventado."
            }}"""),
            ("human", f"Ação do Jogador:\n{acao_jogador}\n\nNarração do Mestre:\n{narracao_mestre}\n\nOpções Dadas:\n{opcoes_str}")
        ])
        
        chain = prompt | self.llm_juiz
        
        for tentativa in range(tentativas):
            try:
                resultado = chain.invoke({"contexto_pdf": contexto_pdf})
                conteudo = resultado.content
                
                if isinstance(conteudo, list):
                    conteudo = conteudo[0].get('text', str(conteudo))
                
                texto_limpo = conteudo.replace("```json", "").replace("```", "").strip()
                dados = json.loads(texto_limpo)
                return dados
                
            except Exception as e:
                if "429" in str(e):
                    tempo_espera = 20 * (tentativa + 1)
                    print(f"⏳ Cota excedida (Erro 429). A aguardar {tempo_espera}s para tentar novamente...")
                    time.sleep(tempo_espera)
                else:
                    print(f"Erro ao converter JSON do Juiz: {e}")
                    break 
                    
        # Fallback atualizado para as novas chaves
        return {"STYLE_REV": 1, "EVENT_CAUS_D": 1, "EVENT_CAUS_R": 1, "EVENT_CAUS_C": 1, "ADHERENCE": 1, "TIME_ORDER": 1, "justificativa": "Erro na avaliação."}

def run_benchmark_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "db", "benchmark_results")
    
    arquivos_transcript = [f for f in os.listdir(results_dir) if f.startswith("transcript_")]
    if not arquivos_transcript:
        print("❌ Nenhum transcript encontrado! Execute a simulação (Fase 3) primeiro.")
        return
        
    arquivo_alvo = os.path.join(results_dir, arquivos_transcript[-1])
    nome_base_arquivo = os.path.basename(arquivo_alvo) # Para salvar no CSV de onde veio o dado
    
    with open(arquivo_alvo, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    print("\n" + "="*60)
    print("⚖️ TRIBUNAL MULTI-TURN RPGBENCH (COM GABARITO DO PDF E NARRABENCH)")
    print("="*60)

    juiz = MasterJudge()
    relatorio_final = []
    
    somas = {"STYLE_REV": 0, "EVENT_CAUS": 0, "ADHERENCE": 0, "TIME_ORDER": 0}

    for turno in transcript:
        num_turno = turno['turno']
        print(f"\n🔍 A avaliar Turno {num_turno}...")
        
        acao_jogador = turno['acao_solicitada']
        mestre_data = turno['resposta_mestre']
        narracao = mestre_data.get('narracao', '')
        opcoes = mestre_data.get('opcoes', [])
        contexto_pdf = mestre_data.get('contexto_usado', 'Nenhum contexto do PDF foi recuperado.')

        notas_brutas = juiz.avaliar_turno_completo(acao_jogador, narracao, opcoes, contexto_pdf)
        
        style_norm = (notas_brutas['STYLE_REV'] - 1) / 4.0
        media_event = (notas_brutas['EVENT_CAUS_D'] + notas_brutas['EVENT_CAUS_R'] + notas_brutas['EVENT_CAUS_C']) / 3.0
        event_norm = (media_event - 1) / 4.0
        adh_norm = (notas_brutas['ADHERENCE'] - 1) / 4.0
        time_norm = (notas_brutas['TIME_ORDER'] - 1) / 4.0

        somas["STYLE_REV"] += style_norm
        somas["EVENT_CAUS"] += event_norm
        somas["ADHERENCE"] += adh_norm
        somas["TIME_ORDER"] += time_norm

        relatorio_final.append({
            "turno": num_turno,
            "metricas": {
                "STYLE_REV": round(style_norm, 3),
                "EVENT_CAUS": round(event_norm, 3),
                "ADHERENCE": round(adh_norm, 3),
                "TIME_ORDER": round(time_norm, 3)
            },
            "justificativa": notas_brutas['justificativa']
        })
        
        print(f"✅ Notas do Turno -> STYLE_REV: {style_norm:.2f} | EVENT_CAUS: {event_norm:.2f} | ADHERENCE: {adh_norm:.2f} | TIME_ORDER: {time_norm:.2f}")

    total = len(transcript)
    medias_globais = {k: round(v / total, 3) for k, v in somas.items()}

    resultado_geral = {
        "medias_globais": medias_globais,
        "detalhes_por_turno": relatorio_final
    }

    # 1. Salva o JSON detalhado
    nome_relatorio = arquivo_alvo.replace("transcript_", "eval_grounded_")
    with open(nome_relatorio, 'w', encoding='utf-8') as f:
        json.dump(resultado_geral, f, ensure_ascii=False, indent=4)

    # ==========================================
    # 2. SALVA OS DADOS NO CSV (COM IDs DE DADOS)
    # ==========================================
    caminho_csv = os.path.join(results_dir, "metrics_history.csv")
    arquivo_existe = os.path.isfile(caminho_csv)
    
    # Gerar ID da Sessão baseado no nome do arquivo (limpando prefixo e extensão)
    id_sessao = nome_base_arquivo.replace("transcript_", "").replace(".json", "")

    # Usando delimitador ';' que é mais amigável para o Excel/PowerBI em PT-BR
    with open(caminho_csv, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        
        # Se for a primeira vez criando o arquivo, escreve o cabeçalho com os IDs
        if not arquivo_existe:
            writer.writerow(["id_turno", "id_sessao", "arquivo_origem", "turno", "style_rev", "event_caus", "adherence", "time_order"])
            
        # Adiciona uma linha para cada turno avaliado
        for turno_info in relatorio_final:
            num_turno = turno_info["turno"]
            # Chave Primária Única para a linha
            id_turno = f"{id_sessao}_T{num_turno}" 
            
            writer.writerow([
                id_turno,
                id_sessao,
                nome_base_arquivo,
                num_turno,
                turno_info["metricas"]["STYLE_REV"],
                turno_info["metricas"]["EVENT_CAUS"],
                turno_info["metricas"]["ADHERENCE"],
                turno_info["metricas"]["TIME_ORDER"]
            ])

    print("\n" + "="*60)
    print(f"📊 MÉDIAS FINAIS DA AVENTURA (Escala 0.000 a 1.000):")
    print(f"✨ Estilo e Revelação  (STYLE_REV)  : {medias_globais['STYLE_REV']:.3f}")
    print(f"🎯 Causalidade Eventos (EVENT_CAUS) : {medias_globais['EVENT_CAUS']:.3f}")
    print(f"🛡️ Aderência ao Lore   (ADHERENCE)  : {medias_globais['ADHERENCE']:.3f}")
    print(f"🔗 Ordem Temporal      (TIME_ORDER) : {medias_globais['TIME_ORDER']:.3f}")
    print(f"💾 Relatório JSON guardado em: {nome_relatorio}")
    print(f"📈 Dados CSV adicionados em: {caminho_csv}")
    print("="*60)
if __name__ == "__main__":
    run_benchmark_evaluation()