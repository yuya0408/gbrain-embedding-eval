# -*- coding: utf-8 -*-
"""Embed the eval corpus (documents) and the frozen query sets (queries)
with every candidate model, applying each model's official prefix scheme.
Batched (BATCH_SIZE per Ollama /api/embed call) per the throughput finding.
Outputs, per model key, under OUT_DIR:
  {key}.chunks.json   -> {"ids": [chunk_id,...], "vectors": [[...], ...]}
  {key}.queries.json  -> {"ids": [query_id,...], "vectors": [[...], ...]}
"""
import json
import os
import time
import urllib.request

from model_config import MODELS, BATCH_SIZE, OLLAMA_URL, apply_fmt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks_eval.jsonl")
CURATED_Q_PATH = os.path.join(DATA_DIR, "curated_queries.jsonl")
AUTO_Q_PATH = os.path.join(DATA_DIR, "auto_queries.jsonl")
OUT_DIR = DATA_DIR


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def embed_batch(tag, texts):
    payload = json.dumps({"model": tag, "input": texts}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    return body["embeddings"]


def embed_all(tag, items, text_key, fmt, char_limit, label):
    ids = []
    vectors = []
    t0 = time.time()
    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        texts = [apply_fmt(fmt, it[text_key], char_limit) for it in batch]
        vecs = embed_batch(tag, texts)
        vectors.extend(vecs)
        ids.extend([it.get("chunk_id", it.get("query_id")) for it in batch])
        print(f"  [{label}] {min(i+BATCH_SIZE, len(items))}/{len(items)}", flush=True)
    print(f"  [{label}] done in {time.time()-t0:.1f}s", flush=True)
    return ids, vectors


def main():
    chunks = load_jsonl(CHUNKS_PATH)
    queries = load_jsonl(CURATED_Q_PATH) + load_jsonl(AUTO_Q_PATH)
    print(f"corpus={len(chunks)} chunks, queries={len(queries)}", flush=True)

    for key, cfg in MODELS.items():
        print(f"=== {key} ({cfg['tag']}) ===", flush=True)
        t0 = time.time()

        c_ids, c_vecs = embed_all(cfg["tag"], chunks, "chunk_text", cfg["doc_fmt"], cfg["char_limit"], "docs")
        with open(f"{OUT_DIR}\\{key}.chunks.json", "w", encoding="utf-8") as f:
            json.dump({"ids": c_ids, "vectors": c_vecs}, f)

        q_ids, q_vecs = embed_all(cfg["tag"], queries, "query_text", cfg["query_fmt"], None, "queries")
        with open(f"{OUT_DIR}\\{key}.queries.json", "w", encoding="utf-8") as f:
            json.dump({"ids": q_ids, "vectors": q_vecs}, f)

        print(f"=== {key} total {time.time()-t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
