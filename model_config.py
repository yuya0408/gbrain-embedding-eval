# -*- coding: utf-8 -*-
# Model configs for the embedding comparison harness.
# query_fmt/doc_fmt are applied per the official recommendation of each model.
MODELS = {
    "ruri": {
        "tag": "kun432/cl-nagoya-ruri-large",
        "query_fmt": "クエリ: {text}",
        "doc_fmt": "文章: {text}",
        "char_limit": 700,  # ~512 token ctx; conservative char-level truncation for JA text
    },
    "bge-m3": {
        "tag": "bge-m3",
        "query_fmt": "{text}",
        "doc_fmt": "{text}",
        "char_limit": None,
    },
    "qwen3-embedding": {
        "tag": "qwen3-embedding:0.6b",
        "query_fmt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{text}",
        "doc_fmt": "{text}",
        "char_limit": None,
    },
    "embeddinggemma": {
        "tag": "embeddinggemma:300m",
        "query_fmt": "task: search result | query: {text}",
        "doc_fmt": "title: none | text: {text}",
        "char_limit": 2800,  # ~2048 token ctx
    },
    "snowflake-arctic-embed2": {
        "tag": "snowflake-arctic-embed2",
        "query_fmt": "query: {text}",
        "doc_fmt": "{text}",
        "char_limit": None,
    },
    "nomic-embed-text": {
        "tag": "nomic-embed-text",
        "query_fmt": "search_query: {text}",
        "doc_fmt": "search_document: {text}",
        "char_limit": 2800,  # ~2048 token ctx
    },
    "all-minilm": {
        "tag": "all-minilm",
        "query_fmt": "{text}",
        "doc_fmt": "{text}",
        "char_limit": 700,  # ~512 token ctx (sentence-transformers default)
    },
}

BATCH_SIZE = 32
OLLAMA_URL = "http://localhost:11434/api/embed"


def apply_fmt(fmt, text, char_limit):
    if char_limit is not None and len(text) > char_limit:
        text = text[:char_limit]
    return fmt.format(text=text)
