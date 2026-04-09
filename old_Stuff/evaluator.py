import os
import json
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from mestre import get_engine # Reaproveitamos o motor do Dia 2

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

def avaliar_resposta(pergunta, resposta_mestre, resposta_esperada):
    # Usando o Pro para ser um juiz mais criterioso
    llm_juiz = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
    
# Prompt ultra-enxuto para economia de tokens
    prompt_juiz = ChatPromptTemplate.from_messages([
        ("system", """Aja como um auditor de RAG para RPG. 
        Compare a RESPOSTA DO MESTRE com o GABARITO.
        Atribua notas de 0.0 a 5.0 para:
        f: Fidelidade (Não inventou nada?)
        r: Recuperação (Citou todos os pontos do gabarito?)
        rel: Relevância (A resposta é útil para o jogador?)
        
        Responda APENAS um JSON: {"f": float, "r": float, "rel": float, "j": "string curta"}"""),
        ("human", f"GABARITO: {resposta_esperada}\nMESTRE: {resposta_mestre}")
    ])
    
    chain = prompt_juiz | llm_juiz
    
    try:
        resultado = chain.invoke({})
        conteudo = resultado.content
        
        # Limpeza de Markdown se houver
        if "```json" in conteudo:
            conteudo = conteudo.split("```json")[1].split("```")[0]
        
        notas = json.loads(conteudo.strip())
        
        # --- CÁLCULO MATEMÁTICO NO PYTHON (Economia de Tokens) ---
        # Fórmula: (Fidelidade * 0.5) + (Recuperação * 0.3) + (Relevância * 0.2)
        f, r, rel = notas['f'], notas['r'], notas['rel']
        nota_final = (f * 0.5) + (r * 0.3) + (rel * 0.2)
        
        return {
            "nota_final": round(nota_final, 2),
            "metricas": {
                "fidelidade": round(f, 2),
                "recuperacao": round(r, 2),
                "relevancia": round(rel, 2)
            },
            "justificativa": notas['j']
        }
    except Exception as e:
        return {"nota_final": 0.0, "metricas": {}, "justificativa": f"Erro no processamento: {e}"}

def run_evaluation():
    configure()
    # Carrega o motor do mestre (ChromaDB + Gemini Flash)
    vectorstore, llm_mestre = get_engine()
    
    caminho_gabarito = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    with open(caminho_gabarito, "r", encoding="utf-8") as f:
        gabaritos = json.load(f)

    relatorio = []

    print("\n" + "="*50)
    print("🧪 AVALIAÇÃO TÉCNICA ")
    print("="*50 + "\n")

    for i, item in enumerate(gabaritos):
        print(f"[{i+1}/{len(gabaritos)}] Avaliando: {item['pergunta'][:50]}...")
        
        # 1. Recuperação (k=2 para economizar)
        docs = vectorstore.similarity_search(item['pergunta'], k=2)
        contexto = "\n\n".join([d.page_content for d in docs])
        
        # 2. Geração da Resposta
        prompt_mestre = f"Use o contexto: {contexto}\n\nResponda: {item['pergunta']}"
        resposta_mestre = llm_mestre.invoke(prompt_mestre).content
        
        # 3. Descanso de Cota (45 segundos para evitar o erro 429)
        if i > 0:
            print("⏳ Aguardando cooldown da API...")
            time.sleep(45)
            
        # 4. Juiz Híbrido
        avaliacao = avaliar_resposta(item['pergunta'], resposta_mestre, item['resposta_esperada'])
        
        relatorio.append({
            "pergunta": item['pergunta'],
            "mestre": resposta_mestre,
            "gabarito": item['resposta_esperada'],
            "nota_final": avaliacao['nota_final'],
            "metricas": avaliacao['metricas'],
            "justificativa": avaliacao['justificativa']
        })
        print(f"✅ Nota Final: {avaliacao['nota_final']}/5.00\n")

    # Salva o resultado final para o TCC
    caminho_output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "resultado_avaliacao.json")
    with open(caminho_output, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=4, ensure_ascii=False)
    
    print(f"🏁 Relatório detalhado salvo em: {caminho_output}")

if __name__ == "__main__":
    run_evaluation()