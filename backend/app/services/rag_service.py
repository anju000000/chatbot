import pickle
from pathlib import Path

from app.core.config import Settings

# BM25 インデックスはプロセス起動時に1回だけ読み込む（リクエストごとの I/O を避ける）
_bm25_cache: tuple | None = None


def _load_bm25(settings: Settings) -> tuple:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    index_path = Path(settings.bm25_index_path)
    docs_path = Path(settings.bm25_docs_path)
    if not index_path.exists() or not docs_path.exists():
        raise FileNotFoundError(
            f"BM25インデックスが見つかりません: {index_path}\n"
            "daihatsu_rag/ingest.py を実行してください。"
        )
    with open(index_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    _bm25_cache = (bm25, docs)
    return _bm25_cache


def _bm25_search(bm25, docs: list[dict], query: str, top_k: int) -> list[dict]:
    tokens = list(query)
    scores = bm25.get_scores(tokens)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in scored[:top_k]:
        if score <= 0:
            continue
        d = docs[idx]
        results.append({
            "score": round(float(score), 4),
            "mode": "BM25",
            "text": d["text"],
            "pdf_name": d.get("pdf_name", ""),
            "chapter": d.get("chapter", ""),
            "article_number": d.get("article_number", ""),
            "article_title": d.get("article_title", ""),
        })
    return results


def _vector_search(settings: Settings, query: str, top_k: int) -> list[dict]:
    try:
        import chromadb
    except ImportError as e:
        raise ImportError("chromadb 未インストール: pip install chromadb") from e

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    collection = client.get_collection(settings.chroma_collection)

    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i] if res.get("distances") else 0.0
        results.append({
            "score": round(max(0.0, 1.0 - dist), 4),
            "mode": "VECTOR",
            "text": doc,
            "pdf_name": meta.get("pdf_name", ""),
            "chapter": meta.get("chapter", ""),
            "article_number": meta.get("article_number", ""),
            "article_title": meta.get("article_title", ""),
        })
    return results


def retrieve(query: str, settings: Settings) -> list[dict]:
    """クエリに関連するチャンクを返す。rag_mode に応じて BM25 / vector / hybrid を切り替え。"""
    mode = settings.rag_mode.lower()
    top_k = settings.rag_top_k

    results: list[dict] = []

    if mode in ("bm25", "hybrid"):
        # BM25 は必須パスなので例外をそのまま上げる
        bm25, docs = _load_bm25(settings)
        results.extend(_bm25_search(bm25, docs, query, top_k))

    if mode in ("vector", "hybrid"):
        # vector は任意パス（ChromaDB 未起動でも BM25 結果を返せるよう握りつぶす）
        try:
            results.extend(_vector_search(settings, query, top_k))
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("vector_search skipped: %s", e)

    if mode == "hybrid":
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["text"][:60]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped[:top_k]

    return results
