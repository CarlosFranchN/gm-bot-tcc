import time
import json
from pathlib import Path
from judge_creative import MasterJudge, JudgeConfig
import eval_utils as utils

class TribunalOrchestrator:
    """
    Orquestra o fluxo de leitura dos transcripts, envio para o Juiz e salvamento dos resultados.
    """
    def __init__(self, provedor: str = "groq"):
        self.provedor = provedor
        
        # 1. Configuração de Caminhos (Paths)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.base_results_dir = self.project_root / "db" / "benchmark_results"
        self.transcripts_dir = self.base_results_dir / "transcript"
        self.eval_folder = self.base_results_dir / "eval"
        self.eval_folder.mkdir(parents=True, exist_ok=True)
        
        # 2. Inicialização do Juiz e Gabaritos
        self.config = JudgeConfig(provedor_juiz=provedor)
        self.juiz = MasterJudge(self.config)
        self.gabaritos = self._carregar_gabaritos()
    
    def _carregar_gabaritos(self) -> dict:
        """Lê o scenarios.json e cria um dicionário rápido de gabaritos para o Juiz."""
        caminho_scenarios = self.project_root / "datasets" / "scenarios.json"
        gabaritos = {}
        
        if caminho_scenarios.exists():
            with open(caminho_scenarios, 'r', encoding='utf-8') as f:
                cenarios = json.load(f)
                for cena in cenarios:
                    gabaritos[cena['id']] = cena.get('contexto_oficial_para_avaliacao', "Gabarito não fornecido.")
        return gabaritos
    def _extrair_metadados(self, nome_arquivo: str):
        """Extrai o modelo alvo, modo de teste (RAG/Baseline) e ID do cenário a partir do nome do arquivo."""
        modelo_alvo = utils.extrair_modelo_do_nome(nome_arquivo, self.config.target_model)
        modo_teste = "Baseline (Sem RAG)" if "_baseline_" in nome_arquivo else "RAG Ligado"
        
        # Extrai o ID do cenário (ex: transcript_adv_desert_01_gemini... -> adv_desert_01)
        partes = nome_arquivo.split("_")
        id_cenario = f"{partes[1]}_{partes[2]}_{partes[3]}" if len(partes) > 3 else "cenario_desconhecido"
        
        return modelo_alvo, modo_teste, id_cenario

    def _avaliar_unico_turno(self, turno: dict, modo_teste: str, id_cenario: str):
        """Prepara o contexto correto e envia um único turno para o Juiz avaliar."""
        print(f"  🔍 Julgando Turno {turno['turno']}...")
        resp = turno['resposta_mestre']
        
        # O Fim do "Juiz Cego": Injetando a verdade absoluta se for Baseline
        if modo_teste == "Baseline (Sem RAG)":
            contexto_para_o_juiz = self.gabaritos.get(id_cenario, "Gabarito não encontrado.")
        else:
            contexto_para_o_juiz = resp.get('contexto_usado', 'Nenhum')

        try:
            notas_brutas = self.juiz.avaliar_turno_completo(
                acao_jogador=turno['acao_solicitada'],
                narracao_mestre=resp.get('narracao', ''),
                opcoes_mestre=resp.get('opcoes', []),
                contexto_pdf=contexto_para_o_juiz
            )
            return notas_brutas
        except Exception as e:
            print(f"   ❌ Erro local ao processar o turno {turno['turno']}: {e}")
            return None

    def _processar_arquivo(self, arquivo_path: Path):
        """Abre um arquivo transcript, avalia todos os seus turnos e salva os resultados."""
        nome_arquivo = arquivo_path.name
        nome_saida = f"eval_{self.provedor}_{nome_arquivo.replace('transcript_', '')}"
        caminho_relatorio_json = self.eval_folder / nome_saida
        
        if caminho_relatorio_json.exists():
            print(f"⏩ {nome_arquivo} já avaliado. Pulando...")
            return False # Indica que não processou arquivo novo

        print(f"\n⚖️ ANALISANDO PROCESSO: {nome_arquivo}")
        modelo_alvo, modo_teste, id_cenario = self._extrair_metadados(nome_arquivo)
        print(f"🎯 Réu: {modelo_alvo} | Modo: {modo_teste}")
        
        with open(arquivo_path, 'r', encoding='utf-8') as f:
            turnos_jogo = json.load(f)

        relatorio_do_arquivo = []
        somas_norm = {"STYLE_REV": 0, "EVENT_CAUS": 0, "ADHERENCE": 0, "TIME_ORDER": 0}
        contagem_valida = 0

        # Loop de Turnos
        for turno in turnos_jogo:
            notas_brutas = self._avaliar_unico_turno(turno, modo_teste, id_cenario)
            
            if notas_brutas is None:
                print(f"  ⚠️ Turno {turno['turno']} ignorado nas métricas (Falha do Juiz).")
                continue 

            contagem_valida += 1
            notas_n = utils.normalizar_notas(notas_brutas)
            for k in somas_norm: somas_norm[k] += notas_n[k]

            relatorio_do_arquivo.append({
                "turno": turno['turno'],
                "metricas": notas_n,
                "justificativa": notas_brutas['justificativa']
            })
            time.sleep(10) # Respiro para Rate Limit da Groq

        # Finalização e Salvamento
        if contagem_valida > 0:
            self._salvar_resultados(nome_arquivo, modelo_alvo, modo_teste, contagem_valida, somas_norm, relatorio_do_arquivo, caminho_relatorio_json)
            
        return True # Indica que processou um arquivo novo

    def _salvar_resultados(self, nome_arquivo, modelo_alvo, modo_teste, contagem, somas, relatorio, caminho_json):
        """Salva o JSON final e adiciona os dados no CSV."""
        resumo_final = {
            "modelo_avaliado": modelo_alvo,
            "juiz_utilizado": self.provedor,
            "modo_teste": modo_teste,
            "medias": {k: round(v/contagem, 3) for k, v in somas.items()},
            "detalhes": relatorio
        }
        with open(caminho_json, 'w', encoding='utf-8') as f:
            json.dump(resumo_final, f, indent=4, ensure_ascii=False)

        id_sessao = nome_arquivo.replace("transcript_", "").replace(".json", "")
        utils.salvar_no_csv(
            caminho_csv=self.eval_folder / "metrics_history.csv", 
            id_sessao=id_sessao, 
            modelo=modelo_alvo, 
            juiz_utilizado=self.provedor, 
            modo_rag=modo_teste,
            nome_arquivo=nome_arquivo, 
            relatorio=relatorio
        )
        print(f"✅ Concluído: {modelo_alvo} julgado por {self.provedor}.")

    def run(self):
        """Método principal que inicia a varredura da pasta e julga tudo o que estiver pendente."""
        transcripts = list(self.transcripts_dir.glob("transcript_*.json"))
        print(f"📂 Vasculhando pasta: {self.transcripts_dir}")
        print(f"📄 Arquivos encontrados: {len(transcripts)}")
        
        processados = 0
        for arquivo_path in transcripts:
            se_processou = self._processar_arquivo(arquivo_path)
            if se_processou:
                processados += 1
                time.sleep(10) # Proteção extra entre arquivos grandes

        if processados == 0:
            print("\n✨ Tudo em ordem! Não há novos transcripts para avaliar.")

# ==========================================
# EXECUÇÃO DO SCRIPT
# ==========================================
if __name__ == "__main__":
    juizes_para_rodar = ["groq"] 
    # juizes_para_rodar = ["google"] 
    
    for provedor in juizes_para_rodar:
        print(f"\n{'#'*60}")
        print(f"⚖️  INICIANDO SESSÃO DO TRIBUNAL COM: {provedor.upper()}")
        print(f"{'#'*60}")
        
        try:
            orquestrador = TribunalOrchestrator(provedor=provedor)
            orquestrador.run()
        except Exception as e:
            print(f"❌ Erro crítico na sessão do juiz {provedor}: {e}")