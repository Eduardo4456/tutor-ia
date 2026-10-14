"""
image_processor.py
==================
Extrai imagens embutidas em páginas de PDF e produz descrições textuais
usando um modelo de visão (multimodal) via Ollama.

Fluxo por página:
  1. PyMuPDF lista todos os objetos de imagem da página (xrefs).
  2. Cada xref é convertido em um Pixmap (bitmap em memória).
  3. Imagens muito pequenas (ícones, linhas decorativas) são ignoradas
     com base em MIN_IMG_PIXELS definido em config.py.
  4. Imagens com canal alpha (RGBA) são achatadas sobre fundo branco
     para evitar artefatos ao codificar em JPEG.
  5. O bitmap é codificado em PNG dentro de um buffer em memória
     (sem gravar em disco) e convertido para base64.
  6. A string base64 é enviada junto com um prompt ao modelo de visão
     via ollama.chat() — o mesmo cliente já usado no projeto.
  7. A resposta do modelo (descrição em português) é retornada como
     texto simples, pronto para ser inserido na base vetorial.

Por que multimodal em vez de OCR puro?
  - OCR (Tesseract) só lê texto em imagens; não entende gráficos,
    fluxogramas, tabelas visuais ou fotos.
  - Um modelo de visão (LLaVA, llama3.2-vision) descreve qualquer
    conteúdo visual em linguagem natural, tornando-o pesquisável.
"""

import base64
import io
import pymupdf
import ollama
from config import VISION_MODEL, MIN_IMG_PIXELS

PROMPT_DESCRICAO = (
    "Descreva em português o que você vê nesta imagem. "
    "Liste textos visíveis, dados de tabelas, etapas de fluxogramas, "
    "ícones e qualquer elemento visual presente. Seja objetivo e direto."
)

# Prefixos que indicam que o modelo recusou descrever a imagem
PREFIXOS_RECUSA = (
    "desculpe",
    "não posso",
    "não consigo",
    "não é possível",
    "não tenho acesso",
    "não está disponível",
    "não foi carregada",
    "unable to",
    "i cannot",
    "i can't",
)


def _pixmap_para_base64(pix: pymupdf.Pixmap) -> str:
    """
    Converte um Pixmap do PyMuPDF para uma string base64 (PNG).

    Se a imagem tiver canal alpha (ex.: PNG com transparência),
    ela é convertida para RGB primeiro para evitar erros de codec.
    """
    if pix.alpha:                          # RGBA → RGB sobre fundo branco
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

    buf = io.BytesIO(pix.tobytes("png"))   # codifica em PNG sem tocar o disco
    return base64.standard_b64encode(buf.getvalue()).decode()


def descrever_imagens_da_pagina(pagina: pymupdf.Page) -> list[str]:
    """
    Recebe uma página PyMuPDF e retorna uma lista de strings, onde cada
    string é a descrição textual de uma imagem encontrada nessa página.

    Parâmetros
    ----------
    pagina : pymupdf.Page
        Objeto de página aberto pelo PyMuPDF.

    Retorna
    -------
    list[str]
        Lista de descrições (pode ser vazia se não houver imagens válidas).
    """
    doc = pagina.parent                    # documento pai para acessar xrefs
    descricoes = []

    for img_info in pagina.get_images(full=True):
        xref = img_info[0]                 # identificador interno da imagem

        try:
            pix = pymupdf.Pixmap(doc, xref)
        except Exception:
            continue                       # xref corrompido ou não suportado

        # --- filtro de tamanho -------------------------------------------------
        # Ignora ícones, linhas e outros elementos decorativos minúsculos.
        if pix.width * pix.height < MIN_IMG_PIXELS:
            continue

        b64 = _pixmap_para_base64(pix)

        try:
            resposta = ollama.chat(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": PROMPT_DESCRICAO,
                    "images": [b64],
                }],
            )
            texto = resposta["message"]["content"].strip()

            # Descarta respostas de recusa do modelo
            if texto and not any(texto.lower().startswith(p) for p in PREFIXOS_RECUSA):
                descricoes.append(texto)
        except Exception as e:
            print(f"  [imagem] erro ao descrever xref {xref}: {e}")

    return descricoes
