# -*- coding: utf-8 -*-
"""The numbers behind the README that screen.py and finals.py do not print.
All of these were computed after the selection, and none of them decided it.

  - MRR and hits by query type (manual / synthetic)
  - how far duplicate chunks move the result (identical text ties in cosine)
  - how far the screening line moves the result (top 3 / top 5 instead of top 4)
  - paired differences between embeddinggemma and the other finalists
  - the before/after check on gbrain itself, by query type

Ties follow screen.py: a stable sort, so the lower chunk position comes first.

Reads the embeddings from embed_all.py, plus the before/after results in
data/gbrain_query_*.json: one row per query, with "query_id" and "rank"
(1-based rank of the gold chunk or page in gbrain's top 10, null if absent).

Usage: python breakdown.py
"""
import json
import os

import numpy as np
from scipy.stats import binomtest, chi2

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10
FINALISTS = ["embeddinggemma", "qwen3-embedding", "snowflake-arctic-embed2", "bge-m3"]
MODELS = FINALISTS + ["nomic-embed-text", "all-minilm", "ruri"]
BEFORE = "gbrain_query_baseline_nomic_full185.json"
AFTER = "gbrain_query_post_embeddinggemma_full185.json"
N_BOOT = 10000
SEED = 42


def load_jsonl(name):
    with open(os.path.join(BASE, name), "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(name):
    # PowerShell wrote the before/after files with a BOM.
    with open(os.path.join(BASE, name), "r", encoding="utf-8-sig") as f:
        return json.load(f)


def full_ranking(key):
    """(chunk ids, query ids, every chunk's position for every query, best first)."""
    cdata = load_json(f"{key}.chunks.json")
    qdata = load_json(f"{key}.queries.json")
    chunks = np.array(cdata["vectors"], dtype=float)
    queries = np.array(qdata["vectors"], dtype=float)
    chunks /= np.linalg.norm(chunks, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    order = np.argsort(-(queries @ chunks.T), axis=1, kind="stable")
    return cdata["ids"], qdata["ids"], order


def gold_set(query, chunk_slug):
    """Chunks that count as correct: the source chunk, or any chunk of the gold page."""
    if query.get("gold_chunk_id") is not None:
        return {query["gold_chunk_id"]}
    return {cid for cid, slug in chunk_slug.items() if slug == query["gold_slug"]}


def rank_of(ranked_ids, gold, text, tie):
    """1-based rank of the first correct chunk in the top K, or None.

    tie="first" keeps the sort order, as screen.py does.
    tie="last" puts copies of a correct chunk's text that are not themselves
    correct ahead of it. tie="same" counts those copies as correct.
    """
    gold_texts = {text[c] for c in gold}
    for pos, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            if tie == "last":
                pos += sum(1 for c in ranked_ids[pos:] if text[c] == text[cid] and c not in gold)
            return pos if pos <= K else None
        if tie == "same" and text[cid] in gold_texts:
            return pos if pos <= K else None
    return None


def cochrans_q(matrix):
    k = matrix.shape[1]
    cols = matrix.sum(axis=0)
    rows = matrix.sum(axis=1)
    denominator = k * rows.sum() - (rows ** 2).sum()
    q = k * (k - 1) * ((cols - cols.mean()) ** 2).sum() / denominator
    return q, chi2.sf(q, k - 1)


def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)


def hits(ranks, k):
    return sum(1 for r in ranks if r and r <= k)


def hit_matrix(ranks_by_model, models, query_ids, k):
    return np.array([[1 if ranks_by_model[m][q] and ranks_by_model[m][q] <= k else 0 for m in models]
                     for q in query_ids])


def main():
    chunks = load_jsonl("chunks_eval.jsonl")
    chunk_slug = {c["chunk_id"]: c["slug"] for c in chunks}
    text = {c["chunk_id"]: c["chunk_text"] for c in chunks}
    queries = load_jsonl("curated_queries.jsonl") + load_jsonl("auto_queries.jsonl")
    by_id = {q["query_id"]: q for q in queries}
    query_ids = [q["query_id"] for q in queries]
    groups = [("manual", [q["query_id"] for q in queries if q.get("gold_chunk_id") is None]),
              ("synthetic", [q["query_id"] for q in queries if q.get("gold_chunk_id") is not None])]
    copies = {}
    for c in chunks:
        copies.setdefault(c["chunk_text"], []).append(c["chunk_id"])
    dup_groups = [v for v in copies.values() if len(v) > 1]
    golds = {qid: gold_set(by_id[qid], chunk_slug) for qid in query_ids}
    n_dup_gold = sum(1 for g in groups[1][1] if len(copies[text[by_id[g]["gold_chunk_id"]]]) > 1)
    print(f"chunks={len(chunks)} queries={len(queries)} pages={len(set(chunk_slug.values()))}")
    print(f"duplicate texts: {len(dup_groups)} groups, {sum(map(len, dup_groups))} chunks; "
          f"{n_dup_gold} synthetic queries have a copy of their gold chunk")

    ranks = {tie: {} for tie in ("first", "last", "same")}
    for key in MODELS:
        chunk_ids, qids, order = full_ranking(key)
        for tie in ranks:
            ranks[tie][key] = {}
        for qid, row in zip(qids, order):
            ranked = [chunk_ids[i] for i in row]
            for tie in ranks:
                ranks[tie][key][qid] = rank_of(ranked, golds[qid], text, tie)
    base = ranks["first"]

    print("\n=== by query type (ties as in screen.py) ===")
    print(f"{'model':26s} {'all MRR':>8s} {'hit@1':>7s} | {'manual MRR':>10s} {'hit@1':>6s} {'hit@10':>7s} | {'synth MRR':>9s} {'hit@1':>6s} {'hit@10':>7s}")
    for key in MODELS:
        all_r = [base[key][q] for q in query_ids]
        man = [base[key][q] for q in groups[0][1]]
        syn = [base[key][q] for q in groups[1][1]]
        print(f"{key:26s} {mrr(all_r):8.3f} {hits(all_r, 1):7d} | {mrr(man):10.3f} {hits(man, 1):6d} {hits(man, K):7d} | "
              f"{mrr(syn):9.3f} {hits(syn, 1):6d} {hits(syn, K):7d}")

    print("\n=== duplicate chunks: finalists under three tie rules ===")
    for tie, label in (("first", "lower position first (screen.py)"), ("last", "gold after its copies"), ("same", "a copy counts as correct")):
        values = "  ".join(f"{key}={mrr([ranks[tie][key][q] for q in query_ids]):.3f}" for key in FINALISTS)
        q10, p10 = cochrans_q(hit_matrix(ranks[tie], FINALISTS, query_ids, K))
        print(f"{label:36s} {values}  Q(hit@10)={q10:.4f} p={p10:.4f}")

    print("\n=== Cochran's Q over the finalists ===")
    for k in (K, 1):
        matrix = hit_matrix(base, FINALISTS, query_ids, k)
        q, p = cochrans_q(matrix)
        split = int(((matrix.sum(axis=1) > 0) & (matrix.sum(axis=1) < 4)).sum())
        print(f"hit@{k}: Q={q:.4f} p={p:.4f}, models disagree on {split}/{len(query_ids)} queries")

    print("\n=== the screening line: Cochran's Q (hit@10) with other cuts ===")
    for label, models in (("top 3", FINALISTS[:3]), ("top 4", FINALISTS), ("top 5", FINALISTS + ["nomic-embed-text"])):
        q, p = cochrans_q(hit_matrix(base, models, query_ids, K))
        print(f"{label}: Q={q:.4f} p={p:.3g}")

    print(f"\n=== embeddinggemma minus each finalist (query-level bootstrap, {N_BOOT} draws, seed {SEED}) ===")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(query_ids), size=(N_BOOT, len(query_ids)))
    rr = {m: np.array([1 / base[m][q] if base[m][q] else 0.0 for q in query_ids]) for m in FINALISTS}
    for other in FINALISTS[1:]:
        diff = rr["embeddinggemma"] - rr[other]
        lo, hi = np.percentile(diff[picks].mean(axis=1), [2.5, 97.5])
        pair = hit_matrix(base, ["embeddinggemma", other], query_ids, K)
        only_a = int(((pair[:, 0] == 1) & (pair[:, 1] == 0)).sum())
        only_b = int(((pair[:, 0] == 0) & (pair[:, 1] == 1)).sum())
        print(f"{other:26s} MRR diff={diff.mean():+.3f} 95% [{lo:+.3f}, {hi:+.3f}]  hit@10 only one side: {only_a} vs {only_b}")

    print("\n=== gbrain itself, before and after the migration ===")
    before = {r["query_id"]: r["rank"] for r in load_json(BEFORE)}
    after = {r["query_id"]: r["rank"] for r in load_json(AFTER)}
    for label, ids in [("all", query_ids)] + groups:
        for name, table in (("before", before), ("after", after)):
            r = [table[q] for q in ids]
            print(f"{label:10s} {name:7s} n={len(r):3d} hit@1={hits(r, 1):3d} hit@3={hits(r, 3):3d} hit@10={hits(r, K):3d} MRR={mrr(r):.3f}")
    only_after = sum(1 for q in query_ids if after[q] and not before[q])
    only_before = sum(1 for q in query_ids if before[q] and not after[q])
    p = binomtest(only_before, only_after + only_before, 0.5).pvalue
    print(f"hit@10 only one side: after {only_after}, before {only_before}, exact McNemar p={p:.2g}")


if __name__ == "__main__":
    main()
