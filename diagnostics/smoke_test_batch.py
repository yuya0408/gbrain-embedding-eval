import json
import os
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/embed"
CHUNKS_PATH = os.environ["CHUNKS_PATH"]
N = 200
BATCH = 32

def load_chunks(n):
    texts = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            texts.append(json.loads(line)["chunk_text"])
            if len(texts) >= n:
                break
    return texts

def embed_batch(model, texts):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    return body["embeddings"]

if __name__ == "__main__":
    model = sys.argv[1]
    texts = load_chunks(N)
    print(f"model={model} chunks={len(texts)} batch={BATCH}", flush=True)
    t0 = time.perf_counter()
    dim = None
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        vecs = embed_batch(model, batch)
        dim = len(vecs[0])
    elapsed = time.perf_counter() - t0
    print(f"model={model} total={elapsed:.2f}s per_chunk={elapsed/len(texts)*1000:.1f}ms throughput={len(texts)/elapsed:.2f} chunks/s dim={dim}", flush=True)
