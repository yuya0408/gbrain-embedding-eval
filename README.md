# gbrain-embedding-eval

[gbrain](https://github.com/garrytan/gbrain) の埋め込みモデルを、既定の `nomic-embed-text` から
統計的検証に基づいて選び直すための比較ハーネス。

解説記事: [個人用RAG基盤gbrainの埋め込みモデルを、統計的検証で選び直した記録](https://zenn.dev/yuya0408/articles/gbrain-embedding-eval)

## 検証の流れ

| Stage | 内容 | スクリプト |
| --- | --- | --- |
| 0 | gbrainのPGLiteからチャンクを書き出し、合成クエリを生成 | `export_chunks.ts`, `gen_synthetic_queries.py` |
| 1 | 全候補でコーパス・クエリを埋め込み、Recall@10 / MRR で足切り | `embed_all.py`, `screen.py` |
| 2 | 決勝4モデルの hit@10 に Cochran's Q 検定(有意なら Holm 補正つき exact McNemar) | `finals.py` |
| 3 | 決勝4モデルの単発レイテンシ・バッチスループットを計測 | `measure_latency.py` |

候補は Ollama 公式ライブラリで提供される埋め込みモデル6つ。`model_config.py` に残っている
`ruri`(と `diagnose_ruri.py`)は、Ollama ではサードパーティ提供版しか使えないため候補から外した。

## 再現手順

前提: Ollama が `localhost:11434` で起動しており、`model_config.py` の各モデルと `qwen3.5:4b` を pull 済み。

```bash
# Stage 0
GBRAIN_DIR=<gbrainのPGLiteデータディレクトリ> npx tsx export_chunks.ts data/chunks.jsonl
# 評価対象から外したいページ(個人的なメモ等)の行を除いて data/chunks_eval.jsonl を作る
# (今回は1ページ・14チャンクを除外し、535→521チャンク)
CHUNKS_PATH=data/chunks_eval.jsonl OUT_PATH=data/auto_queries.jsonl python gen_synthetic_queries.py

# Stage 1-3
python embed_all.py
python screen.py
python finals.py
python measure_latency.py
```

手動作成クエリは `data/curated_queries.jsonl` に `{"query_id", "query_text", "gold_slug"}` 形式で置く。
正解判定は、合成クエリは `gold_chunk_id` の完全一致、手動クエリはチャンクの slug 一致。

## データについて

`data/` は個人のナレッジベース由来のため公開していない(`.gitignore` 済み)。
自分の gbrain コーパスに対して上記手順を実行すれば同じ評価を再現できる。
