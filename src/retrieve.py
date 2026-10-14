import json, numpy as np, pandas as pd, ollama
from config import *

df = pd.read_csv(CSV_PATH)
M = np.array([json.loads(e) for e in df.embedding])
M /= np.linalg.norm(M, axis=1, keepdims=True)

def buscar(pergunta, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    v = np.array(ollama.embed(model=EMB_MODEL, input=pergunta)["embeddings"][0])
    v /= np.linalg.norm(v)
    sims = M @ v
    idx = sims.argsort()[::-1][:k]
    return [(df.iloc[i], float(sims[i])) for i in idx if float(sims[i]) >= threshold]


if __name__ == "__main__":
    import sys
    for c, s in buscar(" ".join(sys.argv[1:])):
        print(f"[{s:.2f}] {c.doc}, p. {c.pagina}\n{c.texto[:300]}\n")