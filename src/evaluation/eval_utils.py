import csv
import json
from pathlib import Path

def extrair_modelo_do_nome(nome_arquivo: str, modelo_padrao: str) -> str:
    """Extrai o nome do modelo a partir da nomenclatura do arquivo JSON."""
    partes = nome_arquivo.replace(".json", "").split("_")
    # Tenta pegar o que está entre o ID do cenário e o timestamp
    if len(partes) >= 6:
        return "_".join(partes[4:-1])
    return modelo_padrao

def normalizar_notas(notas: dict) -> dict:
    """Transforma a escala Likert (1-5) para a escala decimal (0.0-1.0)."""
    # Métrica composta para Causalidade de Eventos
    media_event = (notas['EVENT_CAUS_D'] + notas['EVENT_CAUS_R'] + notas['EVENT_CAUS_C']) / 3.0
    
    return {
        "STYLE_REV": round((notas['STYLE_REV'] - 1) / 4.0, 3),
        "EVENT_CAUS": round((media_event - 1) / 4.0, 3),
        "ADHERENCE": round((notas['ADHERENCE'] - 1) / 4.0, 3),
        "TIME_ORDER": round((notas['TIME_ORDER'] - 1) / 4.0, 3)
    }

def salvar_no_csv(caminho_csv: Path, id_sessao: str, modelo: str, juiz_utilizado : str , nome_arquivo: str, relatorio: list):
    """Garante a persistência dos dados no histórico global CSV."""
    arquivo_existe = caminho_csv.is_file()
    
    with open(caminho_csv, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        
        if not arquivo_existe:
            writer.writerow(["id_turno", "id_sessao", "modelo", "juiz", "arquivo_origem", "turno", "style_rev", "event_caus", "adherence", "time_order"])
            
        for turno in relatorio:
            n_turno = turno["turno"]
            writer.writerow([
                f"{id_sessao}_T{n_turno}", id_sessao, modelo, 
                juiz_utilizado, # 👉 REGISTRAMOS QUEM DEU A NOTA
                nome_arquivo, n_turno,
                turno["metricas"]["STYLE_REV"], turno["metricas"]["EVENT_CAUS"],
                turno["metricas"]["ADHERENCE"], turno["metricas"]["TIME_ORDER"]
            ])