import time
import json
from pathlib import Path
from judge_creative import MasterJudge, JudgeConfig
import eval_utils as utils

def executar_tribunal_completo(provedor: str = "google"):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    # Pasta Base onde tudo acontece
    base_results_dir = PROJECT_ROOT / "db" / "benchmark_results"
    
    # 👉 Onde os transcripts BRUTOS estão (A entrada)
    transcripts_dir = base_results_dir / "transcript"
    
    # 👉 Onde as avaliações JSON vão ficar (A saída)
    eval_folder = base_results_dir / "eval"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    config = JudgeConfig(provedor_juiz=provedor)
    juiz = MasterJudge(config)
    
    
    transcripts = list(transcripts_dir.glob("transcript_*.json"))
    
    print(f"📂 Vasculhando pasta: {transcripts_dir}")
    print(f"📄 Arquivos encontrados: {len(transcripts)}")
    processados = 0

    for arquivo_path in transcripts:
        nome_arquivo = arquivo_path.name
        nome_saida = f"eval_{provedor}_{nome_arquivo.replace('transcript_', '')}"
        caminho_relatorio_json = eval_folder / nome_saida
        
        if caminho_relatorio_json.exists():
            print(f"⏩ {nome_arquivo} já possui avaliação pelo juiz {provedor}. Pulando...")
            continue

        print(f"\n⚖️ ANALISANDO PROCESSO: {nome_arquivo}")
        modelo_alvo = utils.extrair_modelo_do_nome(nome_arquivo, config.target_model)
        
        with open(arquivo_path, 'r', encoding='utf-8') as f:
            turnos_jogo = json.load(f)

        relatorio_do_arquivo = []
        somas_norm = {"STYLE_REV": 0, "EVENT_CAUS": 0, "ADHERENCE": 0, "TIME_ORDER": 0}
        contagem_valida = 0

        # 3. Loop de Avaliação por Turno
        for turno in turnos_jogo:
            print(f"  🔍 Julgando Turno {turno['turno']}...")
            
            resp = turno['resposta_mestre']
            notas_brutas = juiz.avaliar_turno_completo(
                acao_jogador=turno['acao_solicitada'],
                narracao_mestre=resp.get('narracao', ''),
                opcoes_mestre=resp.get('opcoes', []),
                contexto_pdf=resp.get('contexto_usado', 'Nenhum')
            )

            if notas_brutas:
                contagem_valida += 1
                notas_n = utils.normalizar_notas(notas_brutas)
                
                for k in somas_norm: somas_norm[k] += notas_n[k]

                relatorio_do_arquivo.append({
                    "turno": turno['turno'],
                    "metricas": notas_n,
                    "justificativa": notas_brutas['justificativa']
                })

        # 4. Finalização do Arquivo
        if contagem_valida > 0:
            # Salva o JSON individual de conferência
            resumo_final = {
                "modelo_avaliado": modelo_alvo,
                "medias": {k: round(v/contagem_valida, 3) for k, v in somas_norm.items()},
                "detalhes": relatorio_do_arquivo
            }
            with open(caminho_relatorio_json, 'w', encoding='utf-8') as f:
                json.dump(resumo_final, f, indent=4, ensure_ascii=False)

            id_sessao = nome_arquivo.replace("transcript_", "").replace(".json", "")
            utils.salvar_no_csv(
                caminho_csv= eval_folder / "metrics_history.csv", 
                id_sessao=id_sessao, 
                modelo=modelo_alvo, 
                juiz_utilizado=provedor, 
                nome_arquivo=nome_arquivo, 
                relatorio=relatorio_do_arquivo
            )
            print(f"✅ Concluído: {modelo_alvo} avaliado com sucesso.")
            print(f"✅ Concluído: {modelo_alvo} julgado por {provedor}.")
            processados += 1
            time.sleep(10) # Proteção contra rate limit entre arquivos

    if processados == 0:
        print("\n✨ Tudo em ordem! Não há novos transcripts para avaliar.")

if __name__ == "__main__":
    # juizes_para_rodar = ["google","openrouter"] 
    juizes_para_rodar = ["google"] 
    
    for p in juizes_para_rodar:
        print(f"\n{'#'*60}")
        print(f"⚖️  INICIANDO SESSÃO DO TRIBUNAL COM: {p.upper()}")
        print(f"{'#'*60}")
        
        try:
            executar_tribunal_completo(provedor=p)
        except Exception as e:
            print(f"❌ Erro na sessão do juiz {p}: {e}")