#!/usr/bin/env python3
"""
search_test.py — Daihatsu RAG 検索テスト

検索モード:
  bm25   : BM25 キーワード検索（デフォルト）← rank-bm25 のみ必要
  vector : ChromaDB ベクトル検索            ← chromadb 必要
  hybrid : 両方実行して結果をまとめて表示

Usage:
    python search_test.py                        # 対話モード（BM25）
    python search_test.py --query "試用期間"     # 1回検索して終了
    python search_test.py --query "第30条" --mode bm25
    python search_test.py --query "育児休業" --mode vector
    python search_test.py --query "懲戒" --mode hybrid
    python search_test.py --query "賃金" --top-k 5
    python search_test.py --query "出張" --filter-file 国内出張旅費規程.pdf
"""

import sys
import pickle
import argparse
from pathlib import Path

# ─── Config（ingest.py と合わせる）──────────────────────────────────────────
CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8000
COLLECTION_NAME = "daihatsu_regulations"
BM25_INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"
BM25_DOCS_PATH  = Path(__file__).parent / "bm25_docs.pkl"
DEFAULT_TOP_K   = 3


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 検索
# ═══════════════════════════════════════════════════════════════════════════════

def load_bm25():
    """保存済み BM25 インデックスとチャンクを読み込む。"""
    if not BM25_INDEX_PATH.exists() or not BM25_DOCS_PATH.exists():
        print(f"[ERROR] BM25 インデックスが見つかりません: {BM25_INDEX_PATH}")
        print("  → 先に  python ingest.py  を実行してください")
        sys.exit(1)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return bm25, docs


def bm25_search(bm25, docs: list[dict], query: str, top_k: int,
                filter_file: str | None = None) -> list[dict]:
    """
    BM25 キーワード検索。日本語は文字単位トークナイズ。
    filter_file 指定時は該当 pdf_name のみ対象。
    """
    # filter_file 指定 → 対象ドキュメントを絞り込み
    if filter_file:
        filtered = [(i, d) for i, d in enumerate(docs) if d.get("pdf_name") == filter_file]
        if not filtered:
            return []
        indices, subdocs = zip(*filtered)
    else:
        indices = list(range(len(docs)))
        subdocs = docs

    tokens = list(query)                    # 文字単位（形態素解析なし）
    all_scores = bm25.get_scores(tokens)    # 全チャンクのスコア取得

    # filter_file 指定時は絞り込んだインデックスのスコアのみ使用
    if filter_file:
        scored = [(orig_i, all_scores[orig_i], docs[orig_i]) for orig_i in indices]
    else:
        scored = [(i, all_scores[i], docs[i]) for i in indices]

    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for _, score, d in scored[:top_k]:
        if score <= 0:
            continue
        results.append({
            "score":          round(float(score), 4),
            "mode":           "BM25",
            "text":           d["text"],
            "pdf_name":       d.get("pdf_name", ""),
            "chapter":        d.get("chapter", ""),
            "article_number": d.get("article_number", ""),
            "article_title":  d.get("article_title", ""),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB ベクトル検索
# ═══════════════════════════════════════════════════════════════════════════════

def load_chroma_collection():
    """認証なしで ChromaDB コレクションを取得する。"""
    try:
        import chromadb
    except ImportError:
        print("[ERROR] chromadb 未インストール: pip install chromadb")
        sys.exit(1)

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
    except Exception as e:
        print(f"[ERROR] ChromaDB 接続失敗: {e}")
        print(f"  → docker-compose up -d でサーバーを起動してください")
        sys.exit(1)

    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in names:
        print(f"[ERROR] コレクション '{COLLECTION_NAME}' が存在しません")
        print("  → 先に  python ingest.py  を実行してください")
        sys.exit(1)

    return client.get_collection(COLLECTION_NAME)


def vector_search(collection, query: str, top_k: int,
                  filter_file: str | None = None) -> list[dict]:
    """ChromaDB ベクトル検索（where フィルタ対応）。"""
    where = {"pdf_name": filter_file} if filter_file else None

    kwargs = dict(query_texts=[query], n_results=top_k, include=["documents", "metadatas", "distances"])
    if where:
        kwargs["where"] = where

    try:
        res = collection.query(**kwargs)
    except Exception as e:
        print(f"[WARN] ベクトル検索エラー: {e}")
        return []

    results = []
    for i, doc in enumerate(res["documents"][0]):
        meta  = res["metadatas"][0][i]
        dist  = res["distances"][0][i] if res.get("distances") else 0.0
        score = round(max(0.0, 1.0 - dist), 4)
        results.append({
            "score":          score,
            "mode":           "VECTOR",
            "text":           doc,
            "pdf_name":       meta.get("pdf_name", ""),
            "chapter":        meta.get("chapter", ""),
            "article_number": meta.get("article_number", ""),
            "article_title":  meta.get("article_title", ""),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 表示
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results: list[dict], query: str, mode: str) -> None:
    """検索結果を整形表示する。"""
    print(f"\n{'═' * 60}")
    print(f"  🔍 クエリ: 「{query}」  モード: {mode.upper()}  件数: {len(results)}")
    print(f"{'═' * 60}")

    if not results:
        print("  ヒットなし。キーワードを変えてみてください。")
        print()
        return

    for i, r in enumerate(results, 1):
        print(f"\n  {'─' * 56}")
        print(f"  #{i}  [{r['mode']}]  score = {r['score']}")
        print(f"       📄 {r['pdf_name']}")
        print(f"       📂 {r['chapter']}")
        print(f"       🔖 {r['article_number']}  {r['article_title']}")
        print(f"  {'─' * 56}")
        body = r["text"].replace("\n", " ").strip()
        print(f"  {body[:260]}{'…' if len(body) > 260 else ''}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 検索実行
# ═══════════════════════════════════════════════════════════════════════════════

def do_search(query: str, mode: str, top_k: int,
              bm25=None, bm25_docs: list | None = None,
              collection=None, filter_file: str | None = None) -> None:
    """指定モードで検索を実行し、結果を表示する。"""
    results = []

    if mode in ("bm25", "hybrid") and bm25 is not None:
        results.extend(bm25_search(bm25, bm25_docs, query, top_k, filter_file))

    if mode in ("vector", "hybrid") and collection is not None:
        results.extend(vector_search(collection, query, top_k, filter_file))

    if mode == "hybrid":
        # スコア降順 + 重複除去（本文先頭60文字で判定）
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["text"][:60]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped

    print_results(results, query, mode)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daihatsu RAG 検索テスト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python search_test.py                            # 対話モード（BM25）
  python search_test.py --query "試用期間"         # 1回だけ検索
  python search_test.py --query "第30条" --mode bm25
  python search_test.py --query "育児休業" --mode vector
  python search_test.py --query "懲戒" --mode hybrid --top-k 5
  python search_test.py --query "出張" --filter-file 国内出張旅費規程.pdf
""",
    )
    p.add_argument("--query", "-q", default=None,
                   help="検索クエリ（省略時は対話モード）")
    p.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                   help=f"返却件数 (デフォルト: {DEFAULT_TOP_K})")
    p.add_argument("--mode", "-m",
                   choices=["bm25", "vector", "hybrid"], default="bm25",
                   help="検索モード (デフォルト: bm25)")
    p.add_argument("--filter-file", default=None, metavar="FILENAME",
                   help="pdf_name で絞り込み（例: 就業規則.pdf）")
    return p.parse_args()


def main() -> None:
    # Windows CP932 コンソールでも日本語・記号を正しく出力する
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # ── リソース初期化 ────────────────────────────────────────────
    bm25, bm25_docs, collection = None, None, None

    if args.mode in ("bm25", "hybrid"):
        print("BM25 インデックス読み込み中...")
        bm25, bm25_docs = load_bm25()
        print(f"  ✓ {len(bm25_docs)} チャンク")

    if args.mode in ("vector", "hybrid"):
        print("ChromaDB 接続中...")
        collection = load_chroma_collection()
        print(f"  ✓ コレクション: {COLLECTION_NAME}  ({collection.count()} チャンク)")

    print("  準備完了\n")

    # ── 1回だけ検索して終了 ───────────────────────────────────────
    if args.query:
        do_search(args.query, args.mode, args.top_k,
                  bm25, bm25_docs, collection, args.filter_file)
        return

    # ── 対話モード ────────────────────────────────────────────────
    print(f"対話モード  mode={args.mode.upper()}  top_k={args.top_k}")
    if args.filter_file:
        print(f"絞り込み  : {args.filter_file}")
    print("'q' または空 Enter で終了\n")

    while True:
        try:
            query = input("🔍 検索: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        if not query or query.lower() == "q":
            print("終了します。")
            break

        do_search(query, args.mode, args.top_k,
                  bm25, bm25_docs, collection, args.filter_file)


if __name__ == "__main__":
    main()
