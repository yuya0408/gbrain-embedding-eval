# -*- coding: utf-8 -*-
"""Stage-1 screening: Recall@10 and MRR per model, no significance testing.
Hit rule: synthetic queries (gold_chunk_id present) -> exact chunk_id match.
          manual queries (gold_chunk_id absent)     -> chunk's slug == gold_slug.
"""
import json
import math
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10

from model_config import MODELS  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cosine_matrix_topk(query_vecs, chunk_vecs, k):
    # chunk_vecs: list of vectors; normalize once, then dot product ranking.
    def norm(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v

    chunk_norm = [norm(v) for v in chunk_vecs]
    results = []
    for qv in query_vecs:
        qn = norm(qv)
        sims = [sum(a * b for a, b in zip(qn, cv)) for cv in chunk_norm]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
        results.append(ranked)
    return results


def main():
    chunks = load_jsonl(f"{BASE}\\chunks_eval.jsonl")
    chunk_id_to_slug = {c["chunk_id"]: c["slug"] for c in chunks}

    queries = load_jsonl(f"{BASE}\\curated_queries.jsonl") + load_jsonl(f"{BASE}\\auto_queries.jsonl")

    report = []
    for key in MODELS:
        with open(f"{BASE}\\{key}.chunks.json", "r", encoding="utf-8") as f:
            cdata = json.load(f)
        with open(f"{BASE}\\{key}.queries.json", "r", encoding="utf-8") as f:
            qdata = json.load(f)

        chunk_ids = cdata["ids"]
        chunk_vecs = cdata["vectors"]
        query_ids = qdata["ids"]
        query_vecs = qdata["vectors"]
        qid_to_query = {q["query_id"]: q for q in queries}

        topk_lists = cosine_matrix_topk(query_vecs, chunk_vecs, K)

        hits = 0
        rr_sum = 0.0
        n = 0
        for qid, topk_idx in zip(query_ids, topk_lists):
            q = qid_to_query.get(qid)
            if q is None:
                continue
            n += 1
            top_chunk_ids = [chunk_ids[i] for i in topk_idx]
            gold_chunk_id = q.get("gold_chunk_id")
            gold_slug = q.get("gold_slug")

            rank = None
            for pos, cid in enumerate(top_chunk_ids, start=1):
                is_hit = (gold_chunk_id is not None and cid == gold_chunk_id) or (
                    gold_chunk_id is None and chunk_id_to_slug.get(cid) == gold_slug
                )
                if is_hit:
                    rank = pos
                    break
            if rank is not None:
                hits += 1
                rr_sum += 1.0 / rank

        recall_at_k = hits / n
        mrr = rr_sum / n
        report.append((key, recall_at_k, mrr, n))
        print(f"{key:28s} Recall@{K}={recall_at_k:.3f}  MRR={mrr:.3f}  (n={n})", flush=True)

    report.sort(key=lambda r: r[2], reverse=True)
    print("\n=== ranked by MRR ===", flush=True)
    for key, r, m, n in report:
        print(f"{key:28s} Recall@{K}={r:.3f}  MRR={m:.3f}", flush=True)


if __name__ == "__main__":
    main()
