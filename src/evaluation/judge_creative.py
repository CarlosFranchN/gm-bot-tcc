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

load_dotenv(find_dotenv(), override=True)

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
        elif self.provedor_juiz == "llama":
            # 👉 MUDANÇA AQUI: String exata do modelo Llama no OpenRouter
            # Dica: Se quiser o Juiz mais inteligente do mundo, troque para "meta-llama/llama-3.1-405b-instruct"
            self.llm_juiz_nome = "meta-llama/llama-3.3-70b-instruct"
        elif self.provedor_juiz == "groq":
            self.llm_juiz_nome = "llama-3.3-70b-versatile"
        else:
            raise ValueError(f"Provedor '{self.provedor_juiz}' inválido no JudgeConfig.")


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
        elif self.cfg.provedor_juiz == "groq":
            self.llm_juiz = ChatOpenAI(
                api_key=os.getenv("GROQ_API_KEY"), 
                base_url="https://api.groq.com/openai/v1", 
                model=self.cfg.llm_juiz_nome, 
                temperature=0.0, 
                model_kwargs={
                    "response_format": {"type": "json_object"} 
                }
            )
        elif self.cfg.provedor_juiz == "llama":
            # 👉 MUDANÇA AQUI: Configuração limpa e direta para o OpenRouter
            self.llm_juiz = ChatOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
                model=self.cfg.llm_juiz_nome,
                temperature=0.0
            )
        self.parser = PydanticOutputParser(pydantic_object=AvaliacaoJuiz)
        
        self.prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um auditor avançado de Sistemas RAG e Crítico Chefe de Game Design de RPG. VOCÊ É EXTREMAMENTE SEVERO E METICULOSO.
    Sua função primária é ENCONTRAR FALHAS nas respostas do Mestre. Não seja benevolente. Notas 4.0 ou 5.0 devem ser raras e reservadas APENAS para genialidade absoluta.
    
    REGRA DE OURO: Presuma que a resposta é medíocre. O Mestre deve PROVAR seu valor através de precisão técnica e qualidade narrativa para subir sua nota.

    🚨 ATENÇÃO: SEPARAÇÃO ESTRITA DE MÉTRICAS (Combate ao Viés de Verbosidade)
    É crucial que você avalie o mérito literário de forma independente do progresso narrativo:
    - Se o texto for rico e imersivo, a nota alta pertence APENAS ao STYLE_REV.
    - Se esse texto bonito não fizer a história avançar, ignorar a ação do jogador ou mantê-lo em loop (Efeito Esteira), você DEVE punir severamente a nota de TIME_ORDER (Nota 1.0 ou 2.0). 
    - JAMAIS deixe a beleza da prosa mascarar falhas de progresso ou erros de Lore.

    CONTEXTO ORIGINAL DO PDF (GABARITO - VERDADE ABSOLUTA):
    {contexto_pdf}

    EIXOS DE AVALIAÇÃO (Dê notas com precisão decimal, ex: 2.5, 3.8. Use toda a escala de 1.0 a 5.0):

    1. STYLE_REV (Style & Revelation - Arte):
    - Avalie a qualidade da prosa e o apelo sensorial (visão, olfato, tato, audição).
    - PUNIÇÃO: Dê notas baixas (1.0 a 2.5) se o texto for mecânico, seco ou puramente funcional.
    - ELOGIO: Notas altas (4.0 a 5.0) para descrições viscerais que criem imagens mentais fortes e mantenham o tom da aventura.

    2. EVENT_CAUS (Event Causality - Opções ao Jogador):
    - EVENT_CAUS_D (Diversidade): PUNA se as 3 opções forem muito parecidas ou levarem ao mesmo resultado prático.
    - EVENT_CAUS_R (Relevância): PUNA se a opção ignorar o contexto imediato ou a urgência da cena.
    - EVENT_CAUS_C (Clareza): PUNA se a opção for confusa, longa demais ou mecanicamente ambígua.

    3. ADHERENCE (Aderência ao Lore / Anti-Alucinação): 
    - REGRA DE OURO: Compare a Narração EXCLUSIVAMENTE com o CONTEXTO DO PDF acima.
    - SEVERIDADE MÁXIMA: Qualquer invenção de nomes, itens, cores, texturas ou arquitetura que NÃO estejam no gabarito é ALUCINAÇÃO CRÍTICA (Nota 1.0 ou 2.0).
    - OMISSÃO: Se o gabarito destacar um elemento crucial (ex: uma fonte ou símbolo) e o Mestre o ignorar completamente, a nota deve ser penalizada.

    4. TIME_ORDER (Ordem Temporal e Progresso - Mecânica): 
    - CRÍTICO: O Mestre resolveu a intenção do jogador? Se houver muita descrição, mas a ação do jogador não teve desfecho claro, PUNA COM NOTA 1.0. O estado do jogo deve mudar.
    - AGÊNCIA: PUNA com nota 1.0 se o Mestre decidir pelo jogador, rolar dados por ele ou narrar ações futuras do personagem sem dar escolha.

    CRÍTICO: Seu retorno deve ser ÚNICA E EXCLUSIVAMENTE um objeto JSON válido.
    Inicie a justificativa apontando os ERROS e FALHAS primeiro. Seja direto e técnico.

    {{
        "STYLE_REV": <nota>,
        "EVENT_CAUS_D": <nota>,
        "EVENT_CAUS_R": <nota>,
        "EVENT_CAUS_C": <nota>,
        "ADHERENCE": <nota>,
        "TIME_ORDER": <nota>,
        "justificativa": "<Erros encontrados primeiro. Acertos depois.>"
    }}
    {format_instructions}"""),
    ("human", "Ação do Jogador:\n{acao_jogador}\n\nNarração do Mestre:\n{narracao_mestre}\n\nOpções Oferecidas:\n{opcoes_str}")
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
                erro_msg = str(e)
                # 👉 CORREÇÃO: Agora o Juiz também sobrevive ao Erro 503 (Servidor Lotado)
                if "429" in erro_msg or "503" in erro_msg:
                    tempo_espera = 15 * (tentativa + 1)
                    print(f"⏳ Instabilidade na API ({'429' if '429' in erro_msg else '503'}). Aguardando {tempo_espera}s...")
                    time.sleep(tempo_espera)
                else:
                    print(f"❌ Erro de parsing do Juiz na tentativa {tentativa+1}: {e}")
                    
        return None
