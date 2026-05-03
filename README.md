# GM BOT TCC 🎲🤖

Projeto de Trabalho de Conclusão de Curso (TCC) focado na avaliação empírica da criatividade narrativa de Large Language Models (LLMs) atuando como Mestres de RPG de mesa autônomos.


## O Problema e a Solução (Exoesqueleto Cognitivo)
Modelos de IA frequentemente sofrem de perda de contexto e alucinações quando tentam narrar jogos complexos. Para isolar e medir puramente a criatividade da IA, este projeto fornece um Exoesqueleto Cognitivo ao modelo:

1- **RAG (Retrieval-Augmented Generation):** Injeta o lore exato da cena no momento necessário para garantir fidelidade ao cenário (material base em PDF), removendo o fardo da memorização.

2- **Chain-of-Thought (CoT) Estruturado**: Força a IA a raciocinar sobre o estado mecânico do jogo em formato JSON antes de escrever a prosa literária. Isso separa a "lógica" da "arte", mitigando loops lógicos e perda de contexto.

---

## Exemplo de Funcionamento
**Ação do Jogador:** "Entro no santuário escuro com minha tocha."

**Resposta Gerada pelo Motor (Estrutura JSON Obrigatória):**
```JSON
{
  "raciocinio_estado": "O jogador entrou no santuário. O RAG informa que há morcegos no teto. A luz da tocha vai assustá-los e iniciar um encontro.",
  "narracao": "A luz trêmula da sua tocha revela paredes de pedra úmida. Subitamente, o calor do fogo desperta dezenas de sombras no teto que começam a voar em sua direção...",
  "opcoes": [
    "Levantar a tocha para tentar afastar os morcegos com o fogo.",
    "Correr para o fundo do santuário em busca de abrigo.",
    "Apagar a tocha imediatamente e se jogar no chão."
  ]
}

```

---

## Arquitetura do Sistema

O projeto foi construído de forma modular para isolar a geração de texto, a simulação do jogador e a avaliação acadêmica:

1. **Motor do Jogo (`src/core/`)**: O "Cérebro" do Mestre. Utiliza LangChain e ChromaDB (banco vetorial) para recuperar o *lore* exato da cena. Emprega um prompt de sistema avançado que força a IA a calcular o estado do jogo antes de gerar a prosa literária e as opções do jogador.
2. **Simulador Autônomo (`src/simulation/`)**: O "Robô Jogador". Um script que lê diretrizes de um arquivo `scenarios.json`, inicia a sessão e toma decisões aleatórias baseadas nas opções fornecidas pelo Mestre, gerando um *Transcript* completo (log da partida) de forma automatizada.
3. **Tribunal de Avaliação (`src/evaluation/`)**: O "Avaliador". Baseado no conceito de LLM-as-a-Judge, este módulo lê o transcript gerado e o compara com o PDF original, atribuindo notas de 0.00 a 1.00 baseadas no framework RPGBench.

---

## Métricas de Avaliação

Baseado nas taxonomias NarraBench e RAGBench, o juiz avalia as interações em quatro eixos, atribuindo notas normalizadas (0.00 a 1.00):

- **INT (Interestingness / Estilo)**: Qualidade literária e imersão.

- **ACT (Action Quality)**: Diversidade, Relevância e Clareza das opções dadas ao jogador.

- **FID (Fidelidade / Adherence)**: Mede a taxa de alucinação baseada estritamente no material original (RAG).

- **REL (Relevância Narrativa / Time Order)**: Verifica se a ação solicitada foi resolvida temporalmente sem loops.

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
│   │   ├── judge_creative.py             # Script de avaliação (LLM-as-a-Judge)
│   │   └── run_eval.py                   # Executor em lote das avaliações
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
python src/evaluation/run_eval.py
```

## ✒️ Autor e Contato

* **Autor:** Carlos Neto
* **Contexto:** Trabalho de Conclusão de Curso (TCC)
* **Ano:** 2026

---
*Este projeto é de cunho acadêmico e utiliza o cenário "The Desert Wellspring" estritamente como material de validação (Gabarito de RAG).*