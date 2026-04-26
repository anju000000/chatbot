#!/usr/bin/env python3
"""
search_test.py — Daihatsu RAG 検索テスト（メタデータフィルタ強化版）

検索モード:
  bm25   : BM25 キーワード検索（デフォルト）← rank-bm25 のみ必要
  vector : ChromaDB ベクトル検索            ← chromadb 必要
  hybrid : 両方実行して結果をまとめて表示

Usage:
  python search_test.py                                      # 対話モード（BM25）
  python search_test.py --query "試用期間"                   # 1回検索して終了
  python search_test.py --query "休暇" --filter-article 13  # 第13条に絞り込み
  python search_test.py --query "休暇" --filter-article "第13条"
  python search_test.py --query "育児" --filter-chapter "育児・介護"
  python search_test.py --query "出張" --filter-pdf "旅費"
  python search_test.py --query "給与" --filter-article 15 --filter-chapter "賃金"
"""

import re
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

# ─── 同義語辞書（将来の拡張用）────────────────────────────────────────────
# キー: 正規化後の用語, 値: 同義語リスト
# 例: SYNONYMS = {"休暇": ["休み", "有給", "年休"], "賃金": ["給与", "給料"]}
SYNONYMS: dict[str, list[str]] = {}


# ═══════════════════════════════════════════════════════════════════════════════
# フィルタ条件クラス
# ═══════════════════════════════════════════════════════════════════════════════

class FilterConditions:
    """
    メタデータフィルタ条件をまとめて管理するクラス。
    複数フィルタの AND 条件マッチを担う。
    """

    def __init__(
        self,
        article: str | None = None,
        chapter: str | None = None,
        pdf: str | None = None,
    ):
        self.article_raw = article    # 入力値そのまま（表示用）
        self.chapter_raw = chapter
        self.pdf_raw     = pdf

        # 条項番号は数字部分だけ抽出して正規化（"第13条"/"13条"/"13" → "13"）
        self.article_num: str | None = _normalize_article_number(article) if article else None

    def is_empty(self) -> bool:
        """フィルタが1つも指定されていない場合 True。"""
        return self.article_num is None and self.chapter_raw is None and self.pdf_raw is None

    def matches(self, doc: dict) -> bool:
        """
        ドキュメント1件がすべてのフィルタ条件を満たすか判定（AND条件）。
        """
        # 条項番号フィルタ: "第13条" → 数字 "13" で部分一致
        if self.article_num is not None:
            art = doc.get("article_number", "")
            art_num = _normalize_article_number(art)
            if art_num != self.article_num:
                return False

        # 章フィルタ: 部分一致（大文字小文字・全角半角は問わない）
        if self.chapter_raw is not None:
            chapter = doc.get("chapter", "")
            if self.chapter_raw.lower() not in chapter.lower():
                return False

        # PDF名フィルタ: 部分一致
        if self.pdf_raw is not None:
            pdf_name = doc.get("pdf_name", "")
            if self.pdf_raw.lower() not in pdf_name.lower():
                return False

        return True

    def describe(self) -> str:
        """フィルタ内容を人が読める文字列で返す（表示用）。"""
        parts = []
        if self.article_raw:
            parts.append(f"条項番号=「{self.article_raw}」→ 第{self.article_num}条")
        if self.chapter_raw:
            parts.append(f"章=「{self.chapter_raw}」（部分一致）")
        if self.pdf_raw:
            parts.append(f"PDF名=「{self.pdf_raw}」（部分一致）")
        return "  /  ".join(parts) if parts else "（なし）"


def _normalize_article_number(text: str | None) -> str | None:
    """
    "第13条", "13条", "13" のいずれの形式でも数字部分 "13" を返す。
    数字が見つからなければ None を返す。
    """
    if not text:
        return None
    m = re.search(r"\d+", text)
    return m.group() if m else None


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


def bm25_search(
    bm25,
    docs: list[dict],
    query: str,
    top_k: int,
    filters: FilterConditions,
) -> list[dict]:
    """
    BM25 キーワード検索。日本語は文字単位トークナイズ。
    filters に指定された条件に合致するチャンクのみを対象に絞り込む。
    top_k は絞り込み後の件数上限（スコア > 0 のもの）。
    """
    # フィルタ適用: 条件に合致するチャンクのインデックスを収集
    if not filters.is_empty():
        filtered_indices = [i for i, d in enumerate(docs) if filters.matches(d)]
    else:
        filtered_indices = list(range(len(docs)))

    total_filtered = len(filtered_indices)

    if not filtered_indices:
        return []

    tokens = list(query)                 # 文字単位トークナイズ（形態素解析なし）
    all_scores = bm25.get_scores(tokens) # 全チャンクに対するスコアを一括取得

    # 絞り込み済みインデックスのスコアのみ使用してランキング
    scored = [
        (orig_i, all_scores[orig_i], docs[orig_i])
        for orig_i in filtered_indices
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for _, score, d in scored:
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
            "_total_filtered": total_filtered,  # ヒット件数表示に使用
        })
        if len(results) >= top_k:
            break

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


def _build_chroma_where(filters: FilterConditions) -> dict | None:
    """
    FilterConditions を ChromaDB の where 句に変換する。
    ChromaDB は完全一致のみサポートするため、部分一致は後段でフィルタする。
    条項番号だけ where に渡し、章・PDF名は Python 側でフィルタする。
    """
    if filters.is_empty():
        return None

    # ChromaDB では article_number の完全一致のみ where で絞れる
    # 章・PDF名は部分一致なので Python 側フィルタに任せる
    where_conditions = []

    if filters.article_num is not None:
        # "第13条" 形式で格納されている想定
        where_conditions.append({"article_number": {"$eq": f"第{filters.article_num}条"}})

    if len(where_conditions) == 0:
        return None
    if len(where_conditions) == 1:
        return where_conditions[0]
    return {"$and": where_conditions}


def vector_search(
    collection,
    query: str,
    top_k: int,
    filters: FilterConditions,
) -> list[dict]:
    """
    ChromaDB ベクトル検索。
    条項番号は where で絞り込み、章・PDF名は Python 側でフィルタする。
    """
    where = _build_chroma_where(filters)
    # 章・PDF名フィルタが指定されている場合は多めに取得して後段で絞る
    fetch_k = top_k * 5 if (filters.chapter_raw or filters.pdf_raw) else top_k

    kwargs = dict(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    try:
        res = collection.query(**kwargs)
    except Exception as e:
        print(f"[WARN] ベクトル検索エラー: {e}")
        return []

    results = []
    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        dist = res["distances"][0][i] if res.get("distances") else 0.0

        # 章・PDF名のPython側フィルタ（ChromaDBで絞れなかった分）
        if not filters.matches(meta):
            continue

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
        if len(results) >= top_k:
            break

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 表示
# ═══════════════════════════════════════════════════════════════════════════════

# 表示幅（端末に合わせて変更可）
_WIDTH = 64

def print_results(
    results: list[dict],
    query: str,
    mode: str,
    filters: FilterConditions,
    top_k: int,
) -> None:
    """検索結果を整形表示する。フィルタ情報とヒット件数も表示。"""
    print(f"\n{'═' * _WIDTH}")
    print(f"  クエリ : 「{query}」")
    print(f"  モード : {mode.upper()}")

    # フィルタ条件を表示
    if not filters.is_empty():
        print(f"  フィルタ: {filters.describe()}")

    # ヒット件数（BM25 の場合 _total_filtered が付いている）
    total_filtered = next(
        (r["_total_filtered"] for r in results if "_total_filtered" in r), None
    )
    hit_label = f"{len(results)} 件表示"
    if total_filtered is not None and not filters.is_empty():
        hit_label = f"フィルタ後 {total_filtered} 件中 上位 {len(results)} 件を表示"

    print(f"  結果   : {hit_label}")
    print(f"{'═' * _WIDTH}")

    if not results:
        print("  ヒットなし。キーワードまたはフィルタ条件を変えてみてください。")
        print()
        return

    # ─── 結果テーブルヘッダー ───
    print(
        f"  {'#':>2}  {'スコア':>6}  {'条項番号':^8}  "
        f"{'章（抜粋）':<18}  PDF名（抜粋）"
    )
    print(f"  {'─' * (_WIDTH - 2)}")

    for i, r in enumerate(results, 1):
        chapter_short  = r["chapter"][:16] + "…" if len(r["chapter"]) > 16 else r["chapter"]
        pdf_short      = r["pdf_name"][:18] + "…" if len(r["pdf_name"]) > 18 else r["pdf_name"]
        article_label  = r["article_number"] or "—"

        print(
            f"  {i:>2}  {r['score']:>6.4f}  {article_label:^8}  "
            f"{chapter_short:<18}  {pdf_short}"
        )

    # ─── 各結果の本文プレビュー ───
    for i, r in enumerate(results, 1):
        print(f"\n  ── #{i}  {r['article_number']} {r['article_title']} ──")
        body = r["text"].replace("\n", " ").strip()
        print(f"  {body[:280]}{'…' if len(body) > 280 else ''}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 検索実行（共通ディスパッチャ）
# ═══════════════════════════════════════════════════════════════════════════════

def do_search(
    query: str,
    mode: str,
    top_k: int,
    filters: FilterConditions,
    bm25=None,
    bm25_docs: list | None = None,
    collection=None,
) -> None:
    """指定モードで検索を実行し、結果を表示する。"""
    results = []

    if mode in ("bm25", "hybrid") and bm25 is not None:
        results.extend(bm25_search(bm25, bm25_docs, query, top_k, filters))

    if mode in ("vector", "hybrid") and collection is not None:
        results.extend(vector_search(collection, query, top_k, filters))

    if mode == "hybrid":
        # スコア降順 + 重複除去（本文先頭60文字で判定）
        seen: set[str] = set()
        deduped: list[dict] = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = r["text"][:60]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        results = deduped[:top_k]

    print_results(results, query, mode, filters, top_k)


# ═══════════════════════════════════════════════════════════════════════════════
# 対話モード入力パーサ
# ═══════════════════════════════════════════════════════════════════════════════

def parse_interactive_input(raw: str) -> tuple[str, FilterConditions]:
    """
    対話モードの入力を解析する。
    通常クエリ:  "育児休業"
    フィルタ付き: "育児休業  filter-article:13  filter-chapter:育児  filter-pdf:就業規則"

    フィルタキーワードは大文字小文字を問わない。
    フィルタとクエリの区切りは半角スペース。
    フィルタ値にスペースを含める場合はクォートで囲む（簡易対応）。
    """
    # filter-xxx:値 を抽出
    pattern = re.compile(
        r"filter-(article|chapter|pdf):([^\s]+)", re.IGNORECASE
    )
    matches = pattern.findall(raw)

    article = chapter = pdf = None
    for key, val in matches:
        key = key.lower()
        if key == "article":
            article = val
        elif key == "chapter":
            chapter = val
        elif key == "pdf":
            pdf = val

    # フィルタ部分を除去してクエリを取り出す
    query = pattern.sub("", raw).strip()

    return query, FilterConditions(article=article, chapter=chapter, pdf=pdf)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daihatsu RAG 検索テスト — メタデータフィルタ強化版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
フィルタ使用例:
  # 条項番号フィルタ（"第13条" / "13条" / "13" すべて対応）
  python search_test.py --query "休暇" --filter-article 13
  python search_test.py --query "休暇" --filter-article "第13条"
  python search_test.py --query "休暇" --filter-article 13条

  # 章フィルタ（部分一致）
  python search_test.py --query "育児" --filter-chapter "育児・介護"
  python search_test.py --query "労働時間" --filter-chapter "労働時間"

  # PDF名フィルタ（部分一致）
  python search_test.py --query "出張" --filter-pdf "旅費"
  python search_test.py --query "給与" --filter-pdf "就業規則"

  # 複数フィルタ（AND条件）
  python search_test.py --query "給与" --filter-article 15 --filter-chapter "賃金"

  # 対話モード（起動後に filter-xxx: を付けて入力）
  python search_test.py
  🔍 検索: 育児休業  filter-chapter:育児・介護
  🔍 検索: 給与  filter-article:15  filter-pdf:就業規則
""",
    )
    p.add_argument("--query", "-q", default=None,
                   help="検索クエリ（省略時は対話モード）")
    p.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                   help=f"返却件数 (デフォルト: {DEFAULT_TOP_K})")
    p.add_argument("--mode", "-m",
                   choices=["bm25", "vector", "hybrid"], default="bm25",
                   help="検索モード (デフォルト: bm25)")

    # ─── メタデータフィルタ ───────────────────────────────────────
    p.add_argument(
        "--filter-article", default=None, metavar="ARTICLE",
        help=(
            "条項番号で絞り込み。"
            "「第13条」「13条」「13」のいずれの形式でも対応。"
            "例: --filter-article 13"
        ),
    )
    p.add_argument(
        "--filter-chapter", default=None, metavar="CHAPTER",
        help=(
            "章タイトルで絞り込み（部分一致）。"
            "例: --filter-chapter \"育児・介護\""
        ),
    )
    p.add_argument(
        "--filter-pdf", default=None, metavar="PDF_NAME",
        help=(
            "PDF ファイル名で絞り込み（部分一致）。"
            "例: --filter-pdf \"就業規則\""
        ),
    )

    # 旧 --filter-file との互換性を保持（非推奨）
    p.add_argument(
        "--filter-file", default=None, metavar="FILENAME",
        help="[非推奨] pdf_name 完全一致フィルタ。--filter-pdf を推奨。",
    )

    return p.parse_args()


def main() -> None:
    # Windows CP932 コンソールでも日本語・記号を正しく出力する
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # --filter-file（旧オプション）→ --filter-pdf に吸収
    filter_pdf = args.filter_pdf or args.filter_file

    # フィルタ条件オブジェクトを構築
    filters = FilterConditions(
        article=args.filter_article,
        chapter=args.filter_chapter,
        pdf=filter_pdf,
    )

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
        do_search(
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            filters=filters,
            bm25=bm25,
            bm25_docs=bm25_docs,
            collection=collection,
        )
        return

    # ── 対話モード ────────────────────────────────────────────────
    print(f"対話モード  mode={args.mode.upper()}  top_k={args.top_k}")
    if not filters.is_empty():
        print(f"フィルタ   : {filters.describe()}")
    print()
    print("  クエリだけ入力:      育児休業")
    print("  フィルタ付き入力:    育児休業  filter-article:13  filter-chapter:育児・介護")
    print("  filter-pdf:就業規則  のように PDF 名でも絞れます")
    print("  'q' または空 Enter で終了\n")

    while True:
        try:
            raw = input("🔍 検索: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n終了します。")
            break

        if not raw or raw.lower() == "q":
            print("終了します。")
            break

        # 対話入力のフィルタを解析（コマンドライン引数のフィルタとマージ）
        query, inline_filters = parse_interactive_input(raw)

        if not query:
            print("  [INFO] クエリが空です。フィルタのみの検索はできません。")
            continue

        # コマンドラインフィルタ（--filter-xxx）と行内フィルタをマージ
        # 行内フィルタが優先
        merged = FilterConditions(
            article=inline_filters.article_raw or filters.article_raw,
            chapter=inline_filters.chapter_raw or filters.chapter_raw,
            pdf=inline_filters.pdf_raw     or filters.pdf_raw,
        )

        do_search(
            query=query,
            mode=args.mode,
            top_k=args.top_k,
            filters=merged,
            bm25=bm25,
            bm25_docs=bm25_docs,
            collection=collection,
        )


if __name__ == "__main__":
    main()
