# -*- coding: utf-8 -*-
import json
import math
import os

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


chunks = load_jsonl(f"{BASE}\\chunks_eval.jsonl")
chunk_id_to_len = {c["chunk_id"]: len(c["chunk_text"]) for c in chunks}
chunk_id_to_slug = {c["chunk_id"]: c["slug"] for c in chunks}

queries = load_jsonl(f"{BASE}\\curated_queries.jsonl") + load_jsonl(f"{BASE}\\auto_queries.jsonl")
qid_to_query = {q["query_id"]: q for q in queries}

for key in ["ruri", "bge-m3"]:
    with open(f"{BASE}\\{key}.chunks.json", encoding="utf-8") as f:
        cdata = json.load(f)
    with open(f"{BASE}\\{key}.queries.json", encoding="utf-8") as f:
        qdata = json.load(f)

    chunk_ids = cdata["ids"]
    chunk_vecs = [norm(v) for v in cdata["vectors"]]

    # split queries by whether their gold chunk (if known) was truncated
    trunc_hits, trunc_n = 0, 0
    untrunc_hits, untrunc_n = 0, 0
    trunc_rr, untrunc_rr = 0.0, 0.0

    for qid, qv in zip(qdata["ids"], qdata["vectors"]):
        q = qid_to_query.get(qid)
        if q is None:
            continue
        gold_chunk_id = q.get("gold_chunk_id")
        if gold_chunk_id is None:
            continue  # only look at auto queries here (exact chunk gold)
        is_trunc = chunk_id_to_len.get(gold_chunk_id, 0) > 700

        qn = norm(qv)
        sims = [sum(a * b for a, b in zip(qn, cv)) for cv in chunk_vecs]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:K]
        top_ids = [chunk_ids[i] for i in ranked]

        rank = None
        for pos, cid in enumerate(top_ids, start=1):
            if cid == gold_chunk_id:
                rank = pos
                break

        if is_trunc:
            trunc_n += 1
            if rank:
                trunc_hits += 1
                trunc_rr += 1.0 / rank
        else:
            untrunc_n += 1
            if rank:
                untrunc_hits += 1
                untrunc_rr += 1.0 / rank

    print(f"=== {key} ===")
    print(f"  truncated gold chunks:   n={trunc_n:3d}  Recall@10={trunc_hits/trunc_n:.3f}  MRR={trunc_rr/trunc_n:.3f}")
    print(f"  untruncated gold chunks: n={untrunc_n:3d}  Recall@10={untrunc_hits/untrunc_n:.3f}  MRR={untrunc_rr/untrunc_n:.3f}")
