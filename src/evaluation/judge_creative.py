import os
import sys
import json
import time
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

    # 👉 Agora a função recebe o contexto do PDF!
    def avaliar_turno_completo(self, acao_jogador, narracao_mestre, opcoes_mestre, contexto_pdf, tentativas=3):
        
        opcoes_str = "\n".join([f"[{i+1}] {op}" for i, op in enumerate(opcoes_mestre)])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor avançado de Sistemas RAG e Game Design de RPG.
            Avalie a interação do Mestre com base em 4 grandes eixos, dando notas de 1 a 5.
            
            CONTEXTO ORIGINAL DO PDF (GABARITO PARA AVALIAÇÃO DE FIDELIDADE):
            {contexto_pdf}
            
            EIXOS DE AVALIAÇÃO:
            1. INT (Interestingness): A narração é engajadora, criativa e imersiva?
            2. ACT (Action Quality) - Composto por 3 rubricas:
               - ACT_D: As opções oferecem diversidade e caminhos distintos?
               - ACT_R: As opções são relevantes para a cena atual?
               - ACT_C: As opções são claras e bem escritas?
            3. FID (Fidelidade/Groundedness): Compare a Narração do Mestre EXCLUSIVAMENTE com o CONTEXTO DO PDF acima. O Mestre inventou nomes, monstros, itens ou locais que NÃO estão no texto? (Nota 5 = 100% fiel ao texto; penalize invenções).
            4. REL (Relevância Narrativa): O Mestre resolveu a ação do jogador antes de dar novas opções?
            
            Retorne APENAS um JSON válido seguindo exatamente este formato:
            {{
                "INT": float,
                "ACT_D": float,
                "ACT_R": float,
                "ACT_C": float,
                "FID": float,
                "REL": float,
                "justificativa": "Um parágrafo curto justificando as notas. Especifique se houve alguma alucinação ou detalhe inventado."
            }}"""),
            ("human", f"Ação do Jogador:\n{acao_jogador}\n\nNarração do Mestre:\n{narracao_mestre}\n\nOpções Dadas:\n{opcoes_str}")
        ])
        
        chain = prompt | self.llm_juiz
        
        for tentativa in range(tentativas):
            try:
                # 👉 Passamos o PDF para o template aqui
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
                    
        return {"INT": 1, "ACT_D": 1, "ACT_R": 1, "ACT_C": 1, "FID": 1, "REL": 1, "justificativa": "Erro na avaliação."}

def run_benchmark_evaluation():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "db", "benchmark_results")
    
    arquivos_transcript = [f for f in os.listdir(results_dir) if f.startswith("transcript_")]
    if not arquivos_transcript:
        print("❌ Nenhum transcript encontrado! Execute a simulação (Fase 3) primeiro.")
        return
        
    arquivo_alvo = os.path.join(results_dir, arquivos_transcript[-1])
    
    with open(arquivo_alvo, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    print("\n" + "="*60)
    print("⚖️ TRIBUNAL MULTI-TURN RPGBENCH (COM GABARITO DO PDF)")
    print("="*60)

    juiz = MasterJudge()
    relatorio_final = []
    
    somas = {"INT": 0, "ACT": 0, "FID": 0, "REL": 0}

    for turno in transcript:
        num_turno = turno['turno']
        print(f"\n🔍 A avaliar Turno {num_turno}...")
        
        acao_jogador = turno['acao_solicitada']
        mestre_data = turno['resposta_mestre']
        narracao = mestre_data.get('narracao', '')
        opcoes = mestre_data.get('opcoes', [])
        
        # 👉 Extraímos o contexto que foi guardado no turno!
        contexto_pdf = mestre_data.get('contexto_usado', 'Nenhum contexto do PDF foi recuperado.')

        notas_brutas = juiz.avaliar_turno_completo(acao_jogador, narracao, opcoes, contexto_pdf)
        
        int_norm = (notas_brutas['INT'] - 1) / 4.0
        media_act = (notas_brutas['ACT_D'] + notas_brutas['ACT_R'] + notas_brutas['ACT_C']) / 3.0
        act_norm = (media_act - 1) / 4.0
        fid_norm = (notas_brutas['FID'] - 1) / 4.0
        rel_norm = (notas_brutas['REL'] - 1) / 4.0

        somas["INT"] += int_norm
        somas["ACT"] += act_norm
        somas["FID"] += fid_norm
        somas["REL"] += rel_norm

        relatorio_final.append({
            "turno": num_turno,
            "metricas": {
                "INT": round(int_norm, 3),
                "ACT": round(act_norm, 3),
                "FID": round(fid_norm, 3),
                "REL": round(rel_norm, 3)
            },
            "justificativa": notas_brutas['justificativa']
        })
        
        print(f"✅ Notas do Turno -> INT: {int_norm:.2f} | ACT: {act_norm:.2f} | FID: {fid_norm:.2f} | REL: {rel_norm:.2f}")

    total = len(transcript)
    medias_globais = {k: round(v / total, 3) for k, v in somas.items()}

    resultado_geral = {
        "medias_globais": medias_globais,
        "detalhes_por_turno": relatorio_final
    }

    nome_relatorio = arquivo_alvo.replace("transcript_", "eval_grounded_")
    with open(nome_relatorio, 'w', encoding='utf-8') as f:
        json.dump(resultado_geral, f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print(f"📊 MÉDIAS FINAIS DA AVENTURA (Escala 0.000 a 1.000):")
    print(f"✨ Grau de Interesse (INT) : {medias_globais['INT']:.3f}")
    print(f"🎯 Qualidade Ações   (ACT) : {medias_globais['ACT']:.3f}")
    print(f"🛡️ Fidelidade Lore   (FID) : {medias_globais['FID']:.3f} (Agora Real!)")
    print(f"🔗 Relevância Resposta (REL): {medias_globais['REL']:.3f}")
    print(f"💾 Relatório detalhado guardado em: {nome_relatorio}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark_evaluation()