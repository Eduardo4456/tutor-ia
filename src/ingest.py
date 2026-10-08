"""
ingest.py
=========
Pipeline de ingestão de PDFs para a base vetorial.

Etapas por página:
  1. Extração de texto  — via PyMuPDF (pag.get_text), limpeza e chunking.
  2. Extração de imagens — via image_processor.descrever_imagens_da_pagina(),
     que usa um modelo de visão (LLaVA) para gerar descrições textuais.
  3. Todos os chunks (texto + descrições de imagens) recebem embedding
     via ollama.embed e são salvos no CSV da base vetorial.

A coluna "tipo" indica a origem do chunk:
  - "texto"  → trecho extraído diretamente do PDF
  - "imagem" → descrição gerada pelo modelo de visão
"""

import re, json, unicodedata
from pathlib import Path
import pymupdf
import pandas as pd, ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import *
from image_processor import descrever_imagens_da_pagina


def limpar(t):
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"-\n(\w)", r"\1", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    return re.sub(r"^\d(?:\s?\d)?\s+", "", t)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end")

linhas = []

for pdf in Path("data/raw").glob("*.pdf"):
    print(f"\nProcessando: {pdf.name}")
    doc = pymupdf.open(pdf)

    for n, pag in enumerate(doc, 1):

        # ------------------------------------------------------------------
        # 1. TEXTO: extração e chunking normais
        # ------------------------------------------------------------------
        texto = limpar(pag.get_text("text"))
        for pedaco in splitter.split_text(texto):
            if len(pedaco) < MIN_CHUNK:
                continue
            linhas.append({
                "doc": pdf.name,
                "pagina": n,
                "tipo": "texto",
                "texto": pedaco,
            })

        # ------------------------------------------------------------------
        # 2. IMAGENS: o image_processor detecta imagens embutidas na página,
        #    filtra as decorativas e pede ao modelo de visão uma descrição
        #    em linguagem natural. Cada descrição vira um chunk independente.
        # ------------------------------------------------------------------
        descricoes = descrever_imagens_da_pagina(pag)
        for desc in descricoes:
            if len(desc) < MIN_CHUNK:
                continue
            print(f"  [imagem] p.{n} -> {desc[:80]}...")
            linhas.append({
                "doc": pdf.name,
                "pagina": n,
                "tipo": "imagem",
                "texto": desc,           # descrição tratada como texto normal
            })

    doc.close()

df = pd.DataFrame(linhas)
df.insert(0, "id", range(len(df)))

# Libera o modelo de visao da memoria antes de carregar o de embedding
# (evita Compute Error 500 por falta de VRAM/RAM)
print("\nLiberando modelo de visao da memoria...")
try:
    ollama.chat(model=VISION_MODEL, messages=[], keep_alive=0)
except Exception:
    pass

# Prefixo de origem no texto enviado ao modelo de embedding
print("Gerando embeddings...")
entradas = [f"{Path(d).stem}, p. {p} [{tp}]: {t}"
            for d, p, tp, t in zip(df.doc, df.pagina, df.tipo, df.texto)]

embs = []
for i in range(0, len(entradas), 32):
    embs += ollama.embed(model=EMB_MODEL, input=entradas[i:i+32])["embeddings"]

df["embedding"] = [json.dumps(e) for e in embs]
df.to_csv(CSV_PATH, index=False)

n_texto  = (df.tipo == "texto").sum()
n_imagem = (df.tipo == "imagem").sum()
print(f"\n{len(df)} chunks salvos -- {n_texto} de texto, {n_imagem} de imagem.")