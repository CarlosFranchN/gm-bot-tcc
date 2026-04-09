# GM BOT TCC 🎲🤖

Projeto de Trabalho de Conclusão de Curso (TCC) focado na avaliação empírica da criatividade narrativa de Large Language Models (LLMs) atuando como Mestres de RPG de mesa autônomos.

O sistema utiliza a arquitetura **RAG (Retrieval-Augmented Generation)** para garantir fidelidade ao cenário (material base em PDF) e **Chain-of-Thought (CoT)** estruturado em JSON para manter o controle mecânico e o estado do jogo, mitigando o problema de perda de contexto e loops lógicos.

---

## 🏗️ Arquitetura do Sistema

O projeto foi construído de forma modular para isolar a geração de texto, a simulação do jogador e a avaliação acadêmica:

1. **Motor do Jogo (`src/core/`)**: O "Cérebro" do Mestre. Utiliza LangChain e ChromaDB (banco vetorial) para recuperar o *lore* exato da cena. Emprega um prompt de sistema avançado que força a IA a calcular o estado do jogo antes de gerar a prosa literária e as opções do jogador.
2. **Simulador Autônomo (`src/simulation/`)**: O "Robô Jogador". Um script que lê diretrizes de um arquivo `scenarios.json`, inicia a sessão e toma decisões aleatórias baseadas nas opções fornecidas pelo Mestre, gerando um *Transcript* completo (log da partida) de forma automatizada.
3. **Tribunal de Avaliação (`src/evaluation/`)**: O "Avaliador". Baseado no conceito de *LLM-as-a-Judge*, este módulo lê o transcript gerado e o compara com o PDF original, atribuindo notas de 0.00 a 1.00 baseadas no framework RPGBench.

---

## 📊 Métricas de Avaliação

O sistema extrai dados quantitativos para a pesquisa com base em quatro eixos:

* **INT (Interestingness / Grau de Interesse):** Mede a imersão, o uso de detalhes sensoriais e a qualidade literária da narração.
* **ACT (Action Quality / Qualidade das Ações):** Avalia se as opções dadas ao jogador são diversas, relevantes para a cena e claras.
* **FID (Fidelidade / Groundedness):** Mede a taxa de alucinação. Penaliza o modelo se ele inventar nomes, monstros ou locais que não constam no material de referência (RAG).
* **REL (Relevância Narrativa):** Avalia se o modelo resolveu efetivamente a ação solicitada pelo jogador, evitando *cliffhangers* injustificados ou loops lógicos.

---

## 🛠️ Stack Tecnológico

* **Linguagem:** Python 3.10+
* **Framework IA:** LangChain
* **Modelos LLM:** Google Gemini (via API)
* **Banco de Dados Vetorial:** ChromaDB (Embeddings locais/RAG)
* **Estruturação de Dados:** JSON estruturado (para prompts e logs)
* **Padrões de Projeto IA:** RAG (Retrieval-Augmented Generation), CoT (Chain-of-Thought), LLM-as-a-Judge.


## 📂 Estrutura do Repositório

```text
├── data/
│   └── The_Desert_Wellspring-final.pdf   # Aventura base usada como gabarito
├── db/
│   ├── benchmark_results/                # Transcripts e Evals gerados pelo simulador
│   └── chroma_dnd/                       # Banco de dados vetorial
├── src/
│   ├── core/
│   │   ├── engine.py                     # Motor RAG e Chain-of-Thought
│   │   └── memory.py                     # Gerenciamento de memória (Janela deslizante)
│   ├── datasets/
│   │   └── scenarios.json                # Roteiro de cenas, dicas RAG e regras do Diretor
│   ├── evaluation/
│   │   └── judge_creative.py             # Script de avaliação (LLM-as-a-Judge)
│   ├── simulation/
│   │   └── runner.py                     # Script do jogador autônomo
│   ├── ingest.py                         # Script para processar o PDF no ChromaDB
│   └── test_retrieval.py                 # Teste de consistência do banco vetorial
└── requirements.txt
```

## 🚀 Como Executar
Pré-requisitos
Python 3.10+

Chave de API de um modelo LLM compatível (ex: Gemini) configurada no ambiente.

Passos de Instalação e Uso
1. Clone o repositório e crie o ambiente virtual:
```Bash
git clone [https://github.com/SEU_USUARIO/gm-bot-tcc.git](https://github.com/SEU_USUARIO/gm-bot-tcc.git)
cd gm-bot-tcc
python -m venv venv
venv\Scripts\activate  # No Windows
``` 

2. Instale as dependências:
``` Bash
pip install -r requirements.txt
```

3. Crie um arquivo .env na pasta src/ com a sua chave de API:
``` Plaintext
GEMINI_API_KEY=sua_chave_aqui
```

4. Crie o banco de dados vetorial ingerindo o PDF:
```bash
python src/ingest.py
```

5. Rode a simulação automatizada (Geração de Dados):
```Bash
python src/simulation/runner.py
```

5. Avalie o resultado gerado (O Tribunal):
```Bash
python src/evaluation/judge_creative.py
```

## ✒️ Autor e Contato

* **Autor:** Carlos Neto
* **Contexto:** Trabalho de Conclusão de Curso (TCC)
* **Ano:** 2026

---
*Este projeto é de cunho acadêmico e utiliza o cenário "The Desert Wellspring" estritamente como material de validação (Gabarito de RAG).*