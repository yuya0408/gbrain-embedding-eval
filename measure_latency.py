# -*- coding: utf-8 -*-
"""Finals tiebreaker: latency/throughput measurement for the 4 finalist models.

Two things measured, matching what a personal RAG's UX actually depends on:
  - single-query embedding latency (median of 15 consecutive calls, one short query)
  - batch ingest throughput (200 chunks, batch=32, from the real corpus)
"""
import json
import os
import statistics
import time
import urllib.request

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OLLAMA_URL = "http://localhost:11434/api/embed"

FINALISTS = {
    "embeddinggemma": {"tag": "embeddinggemma:300m", "params": "308M", "dim": 768},
    "qwen3-embedding": {"tag": "qwen3-embedding:0.6b", "params": "596M", "dim": 1024},
    "snowflake-arctic-embed2": {"tag": "snowflake-arctic-embed2", "params": "567M", "dim": 1024},
    "bge-m3": {"tag": "bge-m3", "params": "567M", "dim": 1024},
}

SAMPLE_QUERY = "gbrainのMCPサーバーをローカルで設定する手順は？"
N_SINGLE = 15
N_BATCH_CHUNKS = 200
BATCH = 32


def load_chunks(n):
    texts = []
    with open(f"{BASE}\\chunks_eval.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            texts.append(json.loads(line)["chunk_text"])
            if len(texts) >= n:
                break
    return texts


def embed(model, texts):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    return body["embeddings"]


def measure_single(tag):
    # warm up once (model load into memory), then time N_SINGLE consecutive calls
    embed(tag, [SAMPLE_QUERY])
    times = []
    for _ in range(N_SINGLE):
        t0 = time.perf_counter()
        embed(tag, [SAMPLE_QUERY])
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def measure_batch(tag, texts):
    t0 = time.perf_counter()
    for i in range(0, len(texts), BATCH):
        embed(tag, texts[i:i + BATCH])
    elapsed = time.perf_counter() - t0
    return elapsed, len(texts) / elapsed


def main():
    texts = load_chunks(N_BATCH_CHUNKS)
    print(f"batch test: {len(texts)} chunks, batch_size={BATCH}\n")
    print(f"{'model':28s} {'params':>7s} {'dim':>5s} {'single(ms,med)':>15s} {'batch_total(s)':>15s} {'throughput(chunks/s)':>21s}")
    results = []
    for key, cfg in FINALISTS.items():
        single_ms = measure_single(cfg["tag"])
        batch_s, throughput = measure_batch(cfg["tag"], texts)
        results.append((key, cfg, single_ms, batch_s, throughput))
        print(f"{key:28s} {cfg['params']:>7s} {cfg['dim']:>5d} {single_ms:15.1f} {batch_s:15.2f} {throughput:21.2f}")

    print("\n=== ranked by single-query latency (median, ms) ===")
    for key, cfg, single_ms, batch_s, throughput in sorted(results, key=lambda r: r[2]):
        print(f"{key:28s} {single_ms:.1f}ms")


if __name__ == "__main__":
    main()
