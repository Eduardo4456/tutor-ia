import re
import ollama
from config import *
from retrieve import buscar

import sys

# Garante saída UTF-8 no Windows para evitar caracteres corrompidos
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESPOSTAS_FIXAS = ("contate a coordenação",)   # fora de escopo / sem informação

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

    # Curto-circuito inteligente se nenhum trecho atingir o limiar mínimo de similaridade
    if not achados:
        return "Infelizmente não tenho essa informação sobre estágio no manual. Por favor, contate a coordenação pelo portal do aluno."


    contexto = "\n\n".join(f"[{n}] {c.texto}" for n, (c, _) in enumerate(achados, 1))
    r = ollama.chat(model=LLM_MODEL, messages=[{
        "role": "user",
        "content": f"TRECHOS:\n{contexto}\n\nPERGUNTA DO ALUNO: {pergunta}"}])
    texto = r["message"]["content"].strip()

    if any(f in texto.lower() for f in RESPOSTAS_FIXAS):
        return texto                                   # sem fontes

    citados = sorted({int(n) for n in re.findall(r"\[(\d+)\]", texto)
                      if 1 <= int(n) <= len(achados)})
    titulo = "Fontes:"
    if not citados:                                    # modelo não citou
        citados, titulo = list(range(1, len(achados) + 1)), "Trechos consultados (sem citação do modelo):"

    linhas = []
    for n in citados:
        c, s = achados[n - 1]
        linhas.append(f"[{n}] {c.doc}, p. {c.pagina} ({c.tipo}, sim. {s:.2f}): \"{c.texto[:150]}...\"")
    return texto + "\n\n" + titulo + "\n" + "\n".join(linhas)

if __name__ == "__main__":
    print("Tutor de Estágio (digite 'sair' para encerrar)")
    while (p := input("\nAluno: ").strip()).lower() != "sair":
        print("\nTutor:", responder(p))