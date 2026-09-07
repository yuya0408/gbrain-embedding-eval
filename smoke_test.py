import json
import os
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/embed"
CHUNKS_PATH = os.environ["CHUNKS_PATH"]
N = 200

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

def embed_one(model, text):
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    return body["embeddings"][0]

def run(model, texts):
    t0 = time.perf_counter()
    dim = None
    for t in texts:
        vec = embed_one(model, t)
        dim = len(vec)
    elapsed = time.perf_counter() - t0
    return elapsed, dim

if __name__ == "__main__":
    model = sys.argv[1]
    texts = load_chunks(N)
    print(f"model={model} chunks={len(texts)}")
    elapsed, dim = run(model, texts)
    print(f"model={model} total={elapsed:.2f}s per_chunk={elapsed/len(texts)*1000:.1f}ms throughput={len(texts)/elapsed:.2f} chunks/s dim={dim}")
