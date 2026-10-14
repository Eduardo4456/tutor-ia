import re
import ollama
from config import *
from retrieve import buscar

import sys

# Garante saída UTF-8 no Windows para evitar caracteres corrompidos
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESPOSTAS_FIXAS = (                              # frases que indicam recusa/fora de escopo
    "contate a coordenação",
    "não tenho essa informação",
    "não encontrei essa informação",
    "não posso ajudar",
    "só posso ajudar com dúvidas sobre estágio",
    "fora do escopo",
)

NAO_SEI = ("Infelizmente não tenho essa informação sobre estágio no manual. "
           "Por favor, contate a coordenação pelo portal do aluno.")

def eh_saudacao(p):
    limpo = re.sub(r"[^\w\s]", "", p.lower()).strip()
    termos_saudacao = {"oi", "ola", "olá", "bom dia", "boa tarde", "boa noite", "e ai", "e aí", "opa", "eae"}
    termos_apresentacao = ["quem e voce", "quem é você", "quem e vc", "quem é vc", "o que voce faz", "o que você faz", "quem voce e", "quem você é"]
    return limpo in termos_saudacao or any(t in limpo for t in termos_apresentacao)

def responder(pergunta):
    # Se for apenas saudação ou apresentação, responde diretamente sem poluir com trechos do manual
    if eh_saudacao(pergunta):
        r = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": pergunta}])
        return r["message"]["content"].strip()

    achados = buscar(pergunta)

    # Curto-circuito: nenhum trecho atingiu o limiar mínimo de similaridade
    if not achados:
        return NAO_SEI

    contexto = "\n\n".join(f"[{n}] {c.texto}" for n, (c, _) in enumerate(achados, 1))

    # Instrução explícita no user message — modelos pequenos seguem melhor aqui do que no SYSTEM
    prompt_usuario = (
        f"TRECHOS DO MATERIAL OFICIAL:\n{contexto}\n\n"
        f"PERGUNTA DO ALUNO: {pergunta}\n\n"
        "INSTRUÇÃO OBRIGATÓRIA: Responda SOMENTE com informações que estejam ESCRITAS nos TRECHOS acima. "
        "Se a resposta para a pergunta NÃO estiver nos trechos, responda EXATAMENTE: "
        "'Infelizmente não tenho essa informação sobre estágio. Por favor, contate a coordenação pelo portal do aluno.' "
        "NÃO invente, NÃO use conhecimento próprio, NÃO complete com informações externas."
    )

    r = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt_usuario}])
    texto = r["message"]["content"].strip()

    # Detecção de resposta de recusa — retorna limpo, sem fontes
    if any(f in texto.lower() for f in RESPOSTAS_FIXAS):
        return NAO_SEI

    # Verificação pós-resposta: se o modelo não citou NENHUM trecho [N],
    # é forte indício de que inventou a resposta em vez de usar o material
    citados = sorted({int(n) for n in re.findall(r"\[(\d+)\]", texto)
                      if 1 <= int(n) <= len(achados)})

    if not citados:
        # Modelo não citou fonte nenhuma — provavelmente alucinação
        return NAO_SEI

    titulo = "Fontes:"
    linhas = []
    for n in citados:
        c, s = achados[n - 1]
        linhas.append(f"[{n}] {c.doc}, p. {c.pagina} ({c.tipo}, sim. {s:.2f}): \"{c.texto[:150]}...\"")
    return texto + "\n\n" + titulo + "\n" + "\n".join(linhas)

if __name__ == "__main__":
    print("Tutor de Estágio (digite 'sair' para encerrar)")
    while (p := input("\nAluno: ").strip()).lower() != "sair":
        print("\nTutor:", responder(p))