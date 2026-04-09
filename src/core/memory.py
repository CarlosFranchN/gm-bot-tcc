from langchain_core.prompts import ChatPromptTemplate

class ConversationMemory:
    def __init__(self, llm_resumidor, max_turnos_recentes=2):
        """
        Gerenciador de Memória Híbrida.
        :param llm_resumidor: Instância do LLM (Gemini Flash) usada para resumir.
        :param max_turnos_recentes: Quantos turnos (perguntas/respostas) manter crus antes de resumir.
        """
        self.llm = llm_resumidor
        self.max_turnos = max_turnos_recentes
        self.resumo_geral = ""
        self.historico_recente = []  # Lista de dicionários: [{"role": "user", "content": "..."}, ...]

    def adicionar_turno(self, acao_jogador, resposta_mestre):
        """Adiciona uma nova interação e verifica se precisa comprimir a memória."""
        self.historico_recente.append({"role": "user", "content": acao_jogador})
        self.historico_recente.append({"role": "assistant", "content": resposta_mestre})
        
        # Um "turno" equivale a 2 mensagens (1 do user, 1 do assistant).
        # Se passamos do limite, ativamos a compressão.
        if len(self.historico_recente) > (self.max_turnos * 2):
            self._comprimir_memoria()

    def _comprimir_memoria(self):
        """Pega os turnos mais antigos, mescla com o resumo atual e gera um novo resumo."""
        # Pega as duas mensagens mais antigas (1 turno inteiro) para resumir
        turno_para_resumir = self.historico_recente[:2]
        texto_para_resumir = f"Jogador: {turno_para_resumir[0]['content']}\nMestre: {turno_para_resumir[1]['content']}"
        
        prompt_compressao = ChatPromptTemplate.from_messages([
            ("system", """Você é um assistente de Mestre de RPG (D&D).
            Sua tarefa é comprimir o histórico da aventura em um resumo executivo curto (máx 150 palavras).
            
            REGRAS VITAIS DO RESUMO:
            1. Atualize o estado atual da missão.
            2. NUNCA omita itens coletados ou perdidos.
            3. NUNCA omita NPCs encontrados ou derrotados.
            4. Mantenha o local exato onde o jogador está agora.
            
            Resumo Antigo:
            {resumo_antigo}"""),
            ("human", "Novos acontecimentos para incluir no resumo:\n{novos_eventos}")
        ])
        
        chain = prompt_compressao | self.llm
        resultado = chain.invoke({
            "resumo_antigo": self.resumo_geral if self.resumo_geral else "A aventura acabou de começar.",
            "novos_eventos": texto_para_resumir
        })
        
        # Atualiza o resumo oficial
        self.resumo_geral = resultado.content.strip()
        
        # Remove o turno resumido da memória de curto prazo (os 2 primeiros itens)
        self.historico_recente = self.historico_recente[2:]
        
        print("\n⚙️ [SISTEMA] Memória comprimida para economizar tokens.")

    def obter_contexto_formatado(self):
        """
        Retorna a memória pronta para ser injetada no Prompt do Mestre.
        """
        contexto = ""
        if self.resumo_geral:
            contexto += f"=== RESUMO DA HISTÓRIA ATÉ AQUI ===\n{self.resumo_geral}\n===================================\n\n"
            
        if self.historico_recente:
            contexto += "=== ÚLTIMOS ACONTECIMENTOS ===\n"
            for msg in self.historico_recente:
                ator = "JOGADOR" if msg["role"] == "user" else "MESTRE"
                contexto += f"{ator}: {msg['content']}\n"
            contexto += "==============================\n"
            
        return contexto