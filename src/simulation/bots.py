import random

class PlayerBot:
    """Um bot unificado que pode assumir diferentes personas de jogo na simulação."""
    
    def __init__(self, style: str = "explorer", fallback_action="Eu continuo explorando com cautela."):
        self.style = style.lower()
        self.fallback_action = fallback_action

    def escolher_acao(self, opcoes: list[str]) -> str:
        if not opcoes:
            return self.fallback_action

        # COMPORTAMENTO 1: O Explorador (Força a progressão)
        if self.style == "explorer":
            palavras_chave = ["ir", "caminhar", "avançar", "explorar", "seguir", "entrar", "aproximar", "correr", "subir", "descer", "dirigir"]
            
            opcoes_de_movimento = []
            for opcao in opcoes:
                if any(palavra in opcao.lower() for palavra in palavras_chave):
                    opcoes_de_movimento.append(opcao)
                    
            if opcoes_de_movimento:
                return random.choice(opcoes_de_movimento)
            return opcoes[0] # Se o mestre não der opção de andar, escolhe a primeira

        # COMPORTAMENTO 2: O Caótico/Aleatório (Teste de stress)
        elif self.style == "random":
            return random.choice(opcoes)
            
        # Padrão de segurança
        return random.choice(opcoes)