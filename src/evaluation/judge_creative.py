import os
import csv
import json
import time
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_openai import ChatOpenAI

@dataclass
class JudgeConfig:
    target_model: str = "modelo-desconhecido" 
    provedor_juiz: str = "" # Agora este valor será sobrescrito pelo parâmetro
    llm_juiz_nome: str = ""

    def __post_init__(self):
        if self.provedor_juiz == "google":
            self.llm_juiz_nome = "gemini-3.1-flash-lite-preview" 
        elif self.provedor_juiz == "openai":
            self.llm_juiz_nome = "gpt-4o"
        elif self.provedor_juiz == "openrouter":
            self.llm_juiz_nome = "openai/gpt-4o"
        else:
            raise ValueError(f"Provedor '{self.provedor_juiz}' inválido no JudgeConfig.")

# Carrega o .env automaticamente
load_dotenv(find_dotenv(), override=True)
# if not os.getenv("GOOGLE_API_KEY"):
#     raise ValueError("❌ ERRO: GOOGLE_API_KEY não encontrada!")

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
    def __init__(self, config: JudgeConfig = None):
        self.cfg = config if config else JudgeConfig()
        
        print(f"⚖️ TRIBUNAL INICIADO")
        print(f"👨‍⚖️ Juiz: {self.cfg.llm_juiz_nome} (via {self.cfg.provedor_juiz.upper()})")
        print(f"🎯 Avaliando o modelo alvo: {self.cfg.target_model}")

        # 🧠 A MÁGICA DA ESCOLHA DO JUIZ
        if self.cfg.provedor_juiz == "google":
            self.llm_juiz = ChatGoogleGenerativeAI(
                model=self.cfg.llm_juiz_nome, 
                temperature=0.0,
                google_api_key=os.getenv("GOOGLE_API_KEY_JUIZ")
            )
            
        elif self.cfg.provedor_juiz == "openai":
            self.llm_juiz = ChatOpenAI(
                model=self.cfg.llm_juiz_nome, 
                temperature=0.0
            )
            
        elif self.cfg.provedor_juiz == "openrouter":
            self.llm_juiz = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=self.cfg.llm_juiz_nome,
                temperature=0.0
            )

        self.parser = PydanticOutputParser(pydantic_object=AvaliacaoJuiz)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor avançado de Sistemas RAG e Game Design de RPG.
            Avalie a interação do Mestre com base nos eixos, dando notas de 1.0 a 5.0.
            
            CONTEXTO ORIGINAL DO PDF (GABARITO):
            {contexto_pdf}
            
            EIXOS DE AVALIAÇÃO (Baseados nas taxonomias NarraBench e RAGBench):
            1. STYLE_REV (Style & Revelation): A narração utiliza linguagem rica e descritiva para criar suspense e imersão na cena (Imageability)?
            2. EVENT_CAUS (Event Causality): Avalie as opções dadas ao jogador em 3 frentes:
               - Diversidade (D): As opções oferecem caminhos mecânicos distintos?
               - Relevância (R): As opções fazem sentido causal com o que acabou de acontecer?
               - Clareza (C): O texto das opções é direto e fácil de entender?
            3. ADHERENCE (Aderência ao Lore): Compare a Narração EXCLUSIVAMENTE com o CONTEXTO DO PDF acima. O Mestre foi 100% aderente, sem inventar nomes, itens ou monstros? (Nota 5.0 = 100% fiel; penalize invenções).
            4. TIME_ORDER (Ordem Temporal): O Mestre resolveu a ação atual do jogador no presente antes de dar novas opções (evitando loops lógicos ou avançar o tempo sem permissão)?
            
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

# def run_benchmark_evaluation():
#     config = JudgeConfig()
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
#     results_dir = PROJECT_ROOT / "db" / "benchmark_results"
    
#     # 👉 NOVA VARIÁVEL: Aponta especificamente para a subpasta
#     transcripts_dir = results_dir / "transcript" 
    
#     # Pega TODOS os transcripts gerados DENTRO da subpasta
#     arquivos = list(transcripts_dir.glob("transcript_*.json"))
    
#     if not arquivos:
#         print(f"❌ Nenhum transcript encontrado em {transcripts_dir}! Execute a simulação primeiro.")
#         return
        
#     juiz = MasterJudge(config)
#     arquivos_julgados_agora = 0

#     # 🔄 LOOP MÁGICO: Passa por todos os arquivos da pasta
#     for arquivo_alvo in arquivos:
#         nome_base_arquivo = arquivo_alvo.name
        
#         # Onde vamos salvar o relatório? Na pasta results_dir (do lado de fora)
#         nome_relatorio = nome_base_arquivo.replace("transcript_", "eval_grounded_")
#         caminho_relatorio = results_dir / nome_relatorio
        
#         # 1. Checa se este arquivo JÁ FOI avaliado antes
#         if caminho_relatorio.exists():
#             print(f"⏩ Pulando: {nome_base_arquivo} (Já avaliado anteriormente)")
#             continue

#         print("\n" + "="*60)
#         print(f"⚖️ TRIBUNAL INICIADO | Lendo o Processo: {nome_base_arquivo}")
        
#         # 2. MÁGICA: Extrai o nome do modelo-alvo a partir do nome do arquivo!
#         partes_nome = nome_base_arquivo.replace(".json", "").split("_")
#         modelo_inferido = "_".join(partes_nome[4:-1]) if len(partes_nome) >= 6 else config.target_model
#         print(f"🎯 Réu identificado automaticamente: {modelo_inferido}")

#         with open(arquivo_alvo, 'r', encoding='utf-8') as f:
#             transcript = json.load(f)

#         relatorio_final = []
#         somas = {"STYLE_REV": 0, "EVENT_CAUS": 0, "ADHERENCE": 0, "TIME_ORDER": 0}
#         turnos_validos = 0 

#         for turno in transcript:
#             num_turno = turno['turno']
#             print(f"\n🔍 Avaliando Turno {num_turno}...")
            
#             mestre_data = turno['resposta_mestre']
#             notas = juiz.avaliar_turno_completo(
#                 acao_jogador=turno['acao_solicitada'],
#                 narracao_mestre=mestre_data.get('narracao', ''),
#                 opcoes_mestre=mestre_data.get('opcoes', []),
#                 contexto_pdf=mestre_data.get('contexto_usado', 'Nenhum contexto')
#             )
            
#             if not notas:
#                 print(f"⚠️ Turno {num_turno} falhou e foi ignorado nas métricas.")
#                 continue 
                
#             turnos_validos += 1
            
#             # Normalização matemática
#             style_norm = (notas['STYLE_REV'] - 1) / 4.0
#             media_event = (notas['EVENT_CAUS_D'] + notas['EVENT_CAUS_R'] + notas['EVENT_CAUS_C']) / 3.0
#             event_norm = (media_event - 1) / 4.0
#             adh_norm = (notas['ADHERENCE'] - 1) / 4.0
#             time_norm = (notas['TIME_ORDER'] - 1) / 4.0

#             somas["STYLE_REV"] += style_norm
#             somas["EVENT_CAUS"] += event_norm
#             somas["ADHERENCE"] += adh_norm
#             somas["TIME_ORDER"] += time_norm

#             relatorio_final.append({
#                 "turno": num_turno, 
#                 "metricas": {
#                     "STYLE_REV": round(style_norm, 3),
#                     "EVENT_CAUS": round(event_norm, 3),
#                     "ADHERENCE": round(adh_norm, 3),
#                     "TIME_ORDER": round(time_norm, 3)
#                 },
#                 "justificativa": notas['justificativa'] 
#             })
            
#             print(f"✅ Notas -> STYLE_REV: {style_norm:.2f} | EVENT_CAUS: {event_norm:.2f} | ADHERENCE: {adh_norm:.2f} | TIME_ORDER: {time_norm:.2f}")

#         # ==========================================
#         # SALVANDO OS DADOS (Fora da pasta transcript)
#         # ==========================================
#         medias_globais = {k: round(v / turnos_validos, 3) for k, v in somas.items()} if turnos_validos > 0 else somas

#         resultado_geral = {
#             "medias_globais": medias_globais,
#             "detalhes_por_turno": relatorio_final
#         }

#         # 1. Salva o JSON de avaliação (usando o caminho_relatorio que criamos lá em cima)
#         with open(caminho_relatorio, 'w', encoding='utf-8') as f:
#             json.dump(resultado_geral, f, ensure_ascii=False, indent=4)
            
#         # 2. SALVA NO CSV 
#         caminho_csv = results_dir / "metrics_history.csv"
#         arquivo_existe = caminho_csv.is_file()
#         id_sessao = nome_base_arquivo.replace("transcript_", "").replace(".json", "")
                
#         with open(caminho_csv, mode='a', newline='', encoding='utf-8-sig') as f:
#             writer = csv.writer(f, delimiter=';')
#             if not arquivo_existe:
#                 writer.writerow(["id_turno", "id_sessao", "modelo", "arquivo_origem", "turno", "style_rev", "event_caus", "adherence", "time_order"])
                
#             for turno_info in relatorio_final:
#                 num_turno_csv = turno_info["turno"]
#                 writer.writerow([
#                     f"{id_sessao}_T{num_turno_csv}",
#                     id_sessao,
#                     modelo_inferido,
#                     nome_base_arquivo,
#                     num_turno_csv,
#                     turno_info["metricas"]["STYLE_REV"],
#                     turno_info["metricas"]["EVENT_CAUS"],
#                     turno_info["metricas"]["ADHERENCE"],
#                     turno_info["metricas"]["TIME_ORDER"]
#                 ])

#         print(f"💾 Relatório JSON e CSV salvos com sucesso para o modelo: {modelo_inferido}")
#         arquivos_julgados_agora += 1
        
#         time.sleep(10)

#     if arquivos_julgados_agora == 0:
#         print("\n✅ O Tribunal está com a mesa limpa! Todos os transcripts da pasta já foram avaliados.")

# if __name__ == "__main__":
#     run_benchmark_evaluation()