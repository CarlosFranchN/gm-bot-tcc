import os
import csv
import json
import time
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# ==========================================
# CONFIGURAÇÕES DA AVALIAÇÃO
# ==========================================
TARGET_MODEL = "meu-bot-rag-v1" 
# JUDGE_MODEL = "gemini-3.1-flash-lite-preview" 
JUDGE_MODEL = "gemini-3-flash-preview" 

# Carrega o .env automaticamente
load_dotenv(find_dotenv(), override=True)
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("❌ ERRO: GOOGLE_API_KEY não encontrada!")

# ==========================================
# SCHEMA DO JUIZ (Garante o formato exato)
# ==========================================
class AvaliacaoJuiz(BaseModel):
    STYLE_REV: float = Field(description="Nota de 1 a 5 para Estilo e Imersão")
    EVENT_CAUS_D: float = Field(description="Nota de 1 a 5 para Diversidade de Opções")
    EVENT_CAUS_R: float = Field(description="Nota de 1 a 5 para Relevância das Opções")
    EVENT_CAUS_C: float = Field(description="Nota de 1 a 5 para Clareza das Opções")
    ADHERENCE: float = Field(description="Nota de 1 a 5 para Aderência ao PDF (penalize invenções)")
    TIME_ORDER: float = Field(description="Nota de 1 a 5 para Resolução Cronológica")
    justificativa: str = Field(description="Justificativa curta para as notas")

# ==========================================
# CLASSE DO JUIZ
# ==========================================
class MasterJudge:
    def __init__(self):
        self.llm_juiz = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0.0)
        self.parser = PydanticOutputParser(pydantic_object=AvaliacaoJuiz)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor avançado de Sistemas RAG e Game Design de RPG.
            Avalie a interação do Mestre com base nos eixos, dando notas de 1.0 a 5.0.
            
            CONTEXTO ORIGINAL DO PDF (GABARITO):
            {contexto_pdf}
            
            {format_instructions}"""),
            ("human", "Ação do Jogador:\n{acao_jogador}\n\nNarração:\n{narracao_mestre}\n\nOpções:\n{opcoes_str}")
        ])
        
        self.chain = self.prompt | self.llm_juiz | self.parser

    def avaliar_turno_completo(self, acao_jogador, narracao_mestre, opcoes_mestre, contexto_pdf, tentativas=3):
        opcoes_str = "\n".join([f"[{i+1}] {op}" for i, op in enumerate(opcoes_mestre)])
        
        for tentativa in range(tentativas):
            try:
                # O Parser garante que o retorno já seja um objeto Pydantic
                resultado: AvaliacaoJuiz = self.chain.invoke({
                    "contexto_pdf": contexto_pdf,
                    "acao_jogador": acao_jogador,
                    "narracao_mestre": narracao_mestre,
                    "opcoes_str": opcoes_str,
                    "format_instructions": self.parser.get_format_instructions()
                })
                return resultado.model_dump()
                
            except Exception as e:
                if "429" in str(e):
                    tempo_espera = 15 * (tentativa + 1)
                    print(f"⏳ Cota excedida. Aguardando {tempo_espera}s...")
                    time.sleep(tempo_espera)
                else:
                    print(f"❌ Erro de parsing do Juiz na tentativa {tentativa+1}: {e}")
                    
        # Se falhar todas as tentativas, retorna None para não sujar o CSV
        return None

# ==========================================
# FUNÇÃO PRINCIPAL DE BENCHMARK
# ==========================================
def run_benchmark_evaluation():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    results_dir = PROJECT_ROOT / "db" / "benchmark_results"
    
    # Pega os arquivos e ordena pela DATA DE MODIFICAÇÃO (pega o mais recente de verdade)
    arquivos = list(results_dir.glob("transcript_*.json"))
    if not arquivos:
        print("❌ Nenhum transcript encontrado! Execute a simulação primeiro.")
        return
        
    arquivo_alvo = max(arquivos, key=os.path.getmtime)
    nome_base_arquivo = arquivo_alvo.name
    
    with open(arquivo_alvo, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    print(f"\n⚖️ TRIBUNAL INICIADO | Avaliando arquivo: {nome_base_arquivo}")
    
    juiz = MasterJudge()
    relatorio_final = []
    somas = {"STYLE_REV": 0, "EVENT_CAUS": 0, "ADHERENCE": 0, "TIME_ORDER": 0}
    turnos_validos = 0 # Contador para a média excluir os que falharam

    for turno in transcript:
        num_turno = turno['turno']
        print(f"\n🔍 Avaliando Turno {num_turno}...")
        
        mestre_data = turno['resposta_mestre']
        notas = juiz.avaliar_turno_completo(
            acao_jogador=turno['acao_solicitada'],
            narracao_mestre=mestre_data.get('narracao', ''),
            opcoes_mestre=mestre_data.get('opcoes', []),
            contexto_pdf=mestre_data.get('contexto_usado', 'Nenhum contexto')
        )
        
        if not notas:
            print(f"⚠️ Turno {num_turno} falhou e foi ignorado nas métricas.")
            continue # Pula para o próximo turno
            
        turnos_validos += 1
        
        # Normalização matemática
        style_norm = (notas['STYLE_REV'] - 1) / 4.0
        media_event = (notas['EVENT_CAUS_D'] + notas['EVENT_CAUS_R'] + notas['EVENT_CAUS_C']) / 3.0
        event_norm = (media_event - 1) / 4.0
        adh_norm = (notas['ADHERENCE'] - 1) / 4.0
        time_norm = (notas['TIME_ORDER'] - 1) / 4.0

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
            "justificativa": notas['justificativa'] 
        })
        
        print(f"✅ Notas do Turno -> STYLE_REV: {style_norm:.2f} | EVENT_CAUS: {event_norm:.2f} | ADHERENCE: {adh_norm:.2f} | TIME_ORDER: {time_norm:.2f}")

    # ==========================================
    # FORA DO LOOP FOR (Cálculos finais e salvamento)
    # ==========================================
    print("\n" + "="*60)
    print("💾 GERANDO RELATÓRIOS FINAIS...")
    
    # Para a média não quebrar se houver erros (divisão por zero)
    medias_globais = {k: round(v / turnos_validos, 3) for k, v in somas.items()} if turnos_validos > 0 else somas

    resultado_geral = {
        "medias_globais": medias_globais,
        "detalhes_por_turno": relatorio_final
    }

    # 1. Salva o JSON detalhado
    nome_relatorio = str(arquivo_alvo).replace("transcript_", "eval_grounded_") 
    with open(nome_relatorio, 'w', encoding='utf-8') as f:
        json.dump(resultado_geral, f, ensure_ascii=False, indent=4)
        
    # 2. SALVA OS DADOS NO CSV 
    caminho_csv = results_dir / "metrics_history.csv"
    arquivo_existe = caminho_csv.is_file()
    id_sessao = nome_base_arquivo.replace("transcript_", "").replace(".json", "")
            
    # Adiciona uma linha para cada turno avaliado
    with open(caminho_csv, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        
        # Novo Cabeçalho com a coluna 'modelo'
        if not arquivo_existe:
            writer.writerow([
                "id_turno", 
                "id_sessao", 
                "modelo", 
                "arquivo_origem", 
                "turno", 
                "style_rev", 
                "event_caus", 
                "adherence", 
                "time_order"
            ])
            
        for turno_info in relatorio_final:
            num_turno_csv = turno_info["turno"]
            id_turno = f"{id_sessao}_T{num_turno_csv}" 
            
            writer.writerow([
                id_turno,
                id_sessao,
                TARGET_MODEL,
                nome_base_arquivo,
                num_turno_csv,
                turno_info["metricas"]["STYLE_REV"],
                turno_info["metricas"]["EVENT_CAUS"],
                turno_info["metricas"]["ADHERENCE"],
                turno_info["metricas"]["TIME_ORDER"]
            ])

    print(f"📈 Dados CSV (Modelo: {TARGET_MODEL}) adicionados em: {caminho_csv}")

    print("\n" + "="*60)
    print(f"📊 MÉDIAS FINAIS DA AVENTURA (Escala 0.000 a 1.000):")
    if turnos_validos > 0:
        print(f"✨ Estilo e Revelação  (STYLE_REV)  : {medias_globais['STYLE_REV']:.3f}")
        print(f"🎯 Causalidade Eventos (EVENT_CAUS) : {medias_globais['EVENT_CAUS']:.3f}")
        print(f"🛡️ Aderência ao Lore   (ADHERENCE)  : {medias_globais['ADHERENCE']:.3f}")
        print(f"🔗 Ordem Temporal      (TIME_ORDER) : {medias_globais['TIME_ORDER']:.3f}")
    else:
        print("⚠️ Nenhum turno pôde ser avaliado com sucesso.")
    print(f"💾 Relatório JSON guardado em: {nome_relatorio}")
    print(f"📈 Dados CSV adicionados em: {caminho_csv}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark_evaluation()