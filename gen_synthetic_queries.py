# -*- coding: utf-8 -*-
import json
import os
import random
import time
import urllib.request

CHUNKS_PATH = os.environ["CHUNKS_PATH"]
OUT_PATH = os.environ["OUT_PATH"]
TARGET_TOTAL = 150
SEED = 42
MODEL = "qwen3.5:4b"
MIN_TOKENS = 40  # skip near-empty chunks (headers etc.)

PROMPT_TMPL = """次の文章の内容について、検索エンジンやチャットボットに入力するような、短く自然な日本語の質問文を1つだけ作ってください。
- 文章の内容を要約・言い換えるのではなく、その文章を読めば答えられるような「質問」にしてください
- 質問文以外は一切出力しないでください(前置き・説明・カギ括弧は不要)

文章:
{chunk}

質問:"""


def load_chunks():
    rows = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows, target_total, seed):
    rng = random.Random(seed)
    by_slug = {}
    for r in rows:
        if r["token_count"] < MIN_TOKENS:
            continue
        by_slug.setdefault(r["slug"], []).append(r)

    slugs = sorted(by_slug.keys())
    n_slugs = len(slugs)
    base_quota = max(1, target_total // n_slugs)

    selected = []
    for slug in slugs:
        pool = by_slug[slug]
        rng.shuffle(pool)
        k = min(len(pool), base_quota)
        selected.extend(pool[:k])

    # top up to target_total from remaining pool (weighted by leftover chunks), capped
    if len(selected) < target_total:
        selected_ids = {r["chunk_id"] for r in selected}
        remaining = [r for slug in slugs for r in by_slug[slug] if r["chunk_id"] not in selected_ids]
        rng.shuffle(remaining)
        need = target_total - len(selected)
        selected.extend(remaining[:need])

    rng.shuffle(selected)
    return selected[:target_total]


def generate_query(chunk_text):
    prompt = PROMPT_TMPL.format(chunk=chunk_text)
    payload = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "think": False}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body.get("response", "").strip()


def main():
    rows = load_chunks()
    sample = stratified_sample(rows, TARGET_TOTAL, SEED)
    print(f"sampled {len(sample)} chunks across {len(set(r['slug'] for r in sample))} pages", flush=True)

    out = []
    t0 = time.time()
    for i, r in enumerate(sample):
        try:
            q = generate_query(r["chunk_text"])
        except Exception as e:
            print(f"[{i+1}/{len(sample)}] ERROR chunk_id={r['chunk_id']}: {e}", flush=True)
            continue
        out.append(
            {
                "query_id": f"auto_{r['chunk_id']}",
                "query_text": q,
                "gold_chunk_id": r["chunk_id"],
                "gold_slug": r["slug"],
                "source": "auto",
            }
        )
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(sample) - (i + 1)) / rate
            print(f"[{i+1}/{len(sample)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} synthetic queries -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
