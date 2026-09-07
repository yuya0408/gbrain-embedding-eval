# -*- coding: utf-8 -*-
"""Finals: Cochran's Q test (once) across the 4 finalist models on the same
185-query binary hit@10 outcomes; if significant, follow up with all 6 pairwise
exact McNemar tests (Holm-corrected).

Finalists (screening tie-break, MRR 0.700-0.741, break to 5th place at 0.395):
embeddinggemma, qwen3-embedding, snowflake-arctic-embed2, bge-m3

Hit rule mirrors screen.py: auto queries match by exact chunk_id; curated
queries match by slug (page-level gold label).
"""
import json
import math
import os

import numpy as np
from scipy.stats import chi2, binomtest

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
K = 10
FINALISTS = ["embeddinggemma", "qwen3-embedding", "snowflake-arctic-embed2", "bge-m3"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cosine_topk(query_vecs, chunk_vecs, k):
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


def hits_for_model(key, queries, chunk_id_to_slug, qid_order):
    with open(f"{BASE}\\{key}.chunks.json", "r", encoding="utf-8") as f:
        cdata = json.load(f)
    with open(f"{BASE}\\{key}.queries.json", "r", encoding="utf-8") as f:
        qdata = json.load(f)

    chunk_ids = cdata["ids"]
    chunk_vecs = cdata["vectors"]
    query_ids = qdata["ids"]
    query_vecs = qdata["vectors"]
    qid_to_query = {q["query_id"]: q for q in queries}

    topk_lists = cosine_topk(query_vecs, chunk_vecs, K)
    qid_to_hit = {}
    for qid, topk_idx in zip(query_ids, topk_lists):
        q = qid_to_query.get(qid)
        if q is None:
            continue
        top_chunk_ids = [chunk_ids[i] for i in topk_idx]
        gold_chunk_id = q.get("gold_chunk_id")
        gold_slug = q.get("gold_slug")
        hit = 0
        for cid in top_chunk_ids:
            is_hit = (gold_chunk_id is not None and cid == gold_chunk_id) or (
                gold_chunk_id is None and chunk_id_to_slug.get(cid) == gold_slug
            )
            if is_hit:
                hit = 1
                break
        qid_to_hit[qid] = hit

    # align to the common query order, error if any query is missing for this model
    return [qid_to_hit[qid] for qid in qid_order]


def cochrans_q(binary_matrix):
    """binary_matrix: n_subjects x k_treatments array of 0/1."""
    n, k = binary_matrix.shape
    col_sums = binary_matrix.sum(axis=0)  # Cj
    row_sums = binary_matrix.sum(axis=1)  # Ri
    numerator = k * (k - 1) * np.sum((col_sums - col_sums.mean()) ** 2)
    denominator = k * row_sums.sum() - np.sum(row_sums ** 2)
    if denominator == 0:
        return 0.0, 1.0, k - 1
    Q = numerator / denominator
    df = k - 1
    p = chi2.sf(Q, df)
    return Q, p, df


def exact_mcnemar(a, b):
    """a, b: binary arrays for two models over the same subjects.
    Returns (b_count, c_count, p_value) using exact binomial test on discordant pairs."""
    b_count = int(np.sum((a == 1) & (b == 0)))  # a wins
    c_count = int(np.sum((a == 0) & (b == 1)))  # b wins
    n_disc = b_count + c_count
    if n_disc == 0:
        return b_count, c_count, 1.0
    result = binomtest(min(b_count, c_count), n_disc, 0.5, alternative="two-sided")
    return b_count, c_count, result.pvalue


def holm_correct(pvalues):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in original order."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main():
    chunks = load_jsonl(f"{BASE}\\chunks_eval.jsonl")
    chunk_id_to_slug = {c["chunk_id"]: c["slug"] for c in chunks}
    queries = load_jsonl(f"{BASE}\\curated_queries.jsonl") + load_jsonl(f"{BASE}\\auto_queries.jsonl")

    # Common query order: queries present in every finalist's queries.json, intersected.
    qid_sets = []
    for key in FINALISTS:
        with open(f"{BASE}\\{key}.queries.json", "r", encoding="utf-8") as f:
            qdata = json.load(f)
        qid_sets.append(set(qdata["ids"]))
    common_qids = set.intersection(*qid_sets)
    qid_order = [q["query_id"] for q in queries if q["query_id"] in common_qids]
    print(f"N queries used = {len(qid_order)} (expected 185)")

    hit_matrix = {}
    for key in FINALISTS:
        hit_matrix[key] = hits_for_model(key, queries, chunk_id_to_slug, qid_order)
        recall = sum(hit_matrix[key]) / len(hit_matrix[key])
        print(f"{key:28s} Recall@{K}={recall:.3f}  (n={len(hit_matrix[key])})")

    M = np.array([hit_matrix[key] for key in FINALISTS]).T  # n_queries x 4

    print("\n=== Cochran's Q test (4 models, once) ===")
    Q, p, df = cochrans_q(M)
    print(f"Q = {Q:.4f}, df = {df}, p = {p:.6f}")

    alpha = 0.05
    if p >= alpha:
        print(f"\np >= {alpha}: no significant difference among the 4 finalists.")
        print("Stopping here per the pre-registered design -- proceed to the throughput/latency tiebreaker.")
        return

    print(f"\np < {alpha}: significant difference detected -- running post-hoc pairwise McNemar (Holm-corrected).")
    pairs = [(i, j) for i in range(len(FINALISTS)) for j in range(i + 1, len(FINALISTS))]
    raw_results = []
    for i, j in pairs:
        a = np.array(hit_matrix[FINALISTS[i]])
        b = np.array(hit_matrix[FINALISTS[j]])
        b_count, c_count, pval = exact_mcnemar(a, b)
        raw_results.append((FINALISTS[i], FINALISTS[j], b_count, c_count, pval))

    raw_pvals = [r[4] for r in raw_results]
    adj_pvals = holm_correct(raw_pvals)

    print("\n=== Post-hoc pairwise McNemar (exact, Holm-corrected) ===")
    print(f"{'model A':28s} {'model B':28s} {'A-only':>7s} {'B-only':>7s} {'p-raw':>10s} {'p-holm':>10s} sig")
    for (a_key, b_key, b_count, c_count, pval), adj in zip(raw_results, adj_pvals):
        sig = "*" if adj < alpha else ""
        print(f"{a_key:28s} {b_key:28s} {b_count:7d} {c_count:7d} {pval:10.4f} {adj:10.4f} {sig}")


if __name__ == "__main__":
    main()
