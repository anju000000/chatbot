#!/usr/bin/env python3
"""
search_test.py — Daihatsu RAG BM25 検索ツール

Usage:
  python search_test.py                                        # 対話モード
  python search_test.py -q "育児休業"                         # 1回検索して終了
  python search_test.py -q "休暇" --filter-article 13         # 第13条に絞り込み
  python search_test.py -q "育児" --filter-chapter "育児・介護"
  python search_test.py -q "出張" --filter-pdf "旅費"
  python search_test.py -q "給与" --filter-article 15 --filter-chapter "賃金"
"""

import re
import sys
import pickle
import argparse
from pathlib import Path

# ── パス設定（ingest.py と合わせる）─────────────────────────────────────────
BM25_INDEX_PATH = Path(__file__).parent / "bm25_index.pkl"
BM25_DOCS_PATH  = Path(__file__).parent / "bm25_docs.pkl"
DEFAULT_TOP_K   = 3

# ── 同義語辞書（将来の拡張ポイント）─────────────────────────────────────────
# 例: SYNONYMS = {"休暇": ["有給", "年休"], "賃金": ["給与", "給料"]}
SYNONYMS: dict[str, list[str]] = {}

# 表示幅（端末幅に合わせて調整可）
_WIDTH = 66


# ═══════════════════════════════════════════════════════════════════════════════
# フィルタ条件
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_article_num(text: str | None) -> str | None:
    """
    "第13条" / "13条" / "13" のいずれからも数字部分 "13" を返す。
    数字が見つからなければ None。
    """
    if not text:
        return None
    m = re.search(r"\d+", text)
    return m.group() if m else None


class Filters:
    """
    メタデータフィルタ条件をまとめて管理する。
    article / chapter / pdf の3種類を AND 条件で評価する。
    """

    def __init__(
        self,
        article: str | None = None,
        chapter: str | None = None,
        pdf:     str | None = None,
    ):
        self.article_raw = article      # 表示用（入力そのまま）
        self.chapter_raw = chapter
        self.pdf_raw     = pdf
        self.article_num = _extract_article_num(article)  # 正規化済み数字

    # ── 判定 ──────────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return self.article_num is None and self.chapter_raw is None and self.pdf_raw is None

    def matches(self, doc: dict) -> bool:
        """doc が全フィルタ条件を満たす場合 True（AND 条件）。"""
        if self.article_num is not None:
            # "第13条" などから数字を抽出して比較
            if _extract_article_num(doc.get("article_number", "")) != self.article_num:
                return False

        if self.chapter_raw is not None:
            # 章は部分一致（大文字小文字区別なし）
            if self.chapter_raw.lower() not in doc.get("chapter", "").lower():
                return False

        if self.pdf_raw is not None:
            # PDF名も部分一致
            if self.pdf_raw.lower() not in doc.get("pdf_name", "").lower():
                return False

        return True

    # ── 表示用 ────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        """フィルタ内容を人が読める形式で返す。"""
        parts = []
        if self.article_raw:
            parts.append(f"条項番号=「{self.article_raw}」(第{self.article_num}条)")
        if self.chapter_raw:
            parts.append(f"章=「{self.chapter_raw}」(部分一致)")
        if self.pdf_raw:
            parts.append(f"PDF=「{self.pdf_raw}」(部分一致)")
        return " AND ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 インデックス読み込み・検索
# ═══════════════════════════════════════════════════════════════════════════════

def load_bm25() -> tuple:
    """bm25_index.pkl と bm25_docs.pkl を読み込んで返す。"""
    if not BM25_INDEX_PATH.exists() or not BM25_DOCS_PATH.exists():
        print(f"[ERROR] BM25 インデックスが見つかりません: {BM25_INDEX_PATH}")
        print("  → 先に  python ingest.py  を実行してください")
        sys.exit(1)

    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    with open(BM25_DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return bm25, docs


def search(bm25, docs: list[dict], query: str, top_k: int, filters: Filters) -> list[dict]:
    """
    BM25 検索を実行して上位 top_k 件を返す。
    filters が指定されている場合は先にチャンクを絞り込んでからスコアリングする。

    戻り値の各 dict に _total_filtered（フィルタ後の総チャンク数）を含める。
    """
    # ① フィルタ適用：条件を満たすチャンクのインデックスだけ残す
    if filters.is_empty():
        target_indices = list(range(len(docs)))
    else:
        target_indices = [i for i, d in enumerate(docs) if filters.matches(d)]

    total_filtered = len(target_indices)

    if not target_indices:
        return []

    # ② BM25 スコア算出（文字単位トークナイズ、形態素解析なし）
    all_scores = bm25.get_scores(list(query))

    # ③ 絞り込み済みチャンクのみランキング
    ranked = sorted(
        ((all_scores[i], docs[i]) for i in target_indices),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, d in ranked:
        if score <= 0:
            break  # BM25 スコアは降順なので 0 以下になったら終了
        results.append({
            "score":           round(float(score), 4),
            "text":            d["text"],
            "pdf_name":        d.get("pdf_name", ""),
            "chapter":         d.get("chapter", ""),
            "article_number":  d.get("article_number", ""),
            "article_title":   d.get("article_title", ""),
            "_total_filtered": total_filtered,
        })
        if len(results) >= top_k:
            break

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 結果表示
# ═══════════════════════════════════════════════════════════════════════════════

def _truncate(text: str, length: int) -> str:
    return text[:length] + "…" if len(text) > length else text


def print_results(results: list[dict], query: str, filters: Filters) -> None:
    """検索結果をヘッダー・テーブル・本文プレビューの3段構成で表示する。"""

    # ── ヘッダー ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * _WIDTH}")
    print(f"  クエリ : 「{query}」")

    if not filters.is_empty():
        print(f"  フィルタ: {filters.describe()}")

    # ヒット件数ラベル
    total_filtered = results[0]["_total_filtered"] if results else 0
    if not filters.is_empty() and results:
        hit_label = f"フィルタ後 {total_filtered} 件中 上位 {len(results)} 件を表示"
    elif results:
        hit_label = f"上位 {len(results)} 件を表示"
    else:
        hit_label = "ヒットなし"

    print(f"  結果   : {hit_label}")
    print(f"{'═' * _WIDTH}")

    if not results:
        print("  キーワードまたはフィルタ条件を変えてみてください。")
        print()
        return

    # ── サマリーテーブル ───────────────────────────────────────────────────────
    print(f"  {'#':>2}  {'スコア':>6}  {'条項番号':^8}  {'章（抜粋）':<16}  PDF名（抜粋）")
    print(f"  {'─' * (_WIDTH - 2)}")

    for i, r in enumerate(results, 1):
        print(
            f"  {i:>2}  {r['score']:>6.4f}"
            f"  {(r['article_number'] or '—'):^8}"
            f"  {_truncate(r['chapter'], 14):<16}"
            f"  {_truncate(r['pdf_name'], 20)}"
        )

    # ── 本文プレビュー ────────────────────────────────────────────────────────
    for i, r in enumerate(results, 1):
        header = f"#{i}  {r['article_number']} {r['article_title']}".strip()
        print(f"\n  ── {header} ──")
        body = r["text"].replace("\n", " ").strip()
        print(f"  {_truncate(body, 300)}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 対話モード入力パーサ
# ═══════════════════════════════════════════════════════════════════════════════

# 対話モードで使えるインラインフィルタの書式:
#   育児休業  filter-article:13  filter-chapter:育児・介護  filter-pdf:就業規則
_INLINE_FILTER_RE = re.compile(r"filter-(article|chapter|pdf):(\S+)", re.IGNORECASE)


def parse_inline(raw: str) -> tuple[str, Filters]:
    """
    入力文字列からインラインフィルタを抽出し、(クエリ文字列, Filters) を返す。
    フィルタが無い場合は空の Filters を返す。
    """
    article = chapter = pdf = None
    for key, val in _INLINE_FILTER_RE.findall(raw):
        key = key.lower()
        if key == "article":
            article = val
        elif key == "chapter":
            chapter = val
        elif key == "pdf":
            pdf = val

    query = _INLINE_FILTER_RE.sub("", raw).strip()
    return query, Filters(article=article, chapter=chapter, pdf=pdf)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 引数
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daihatsu RAG — BM25 検索ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python search_test.py -q "育児休業"
  python search_test.py -q "休暇" --filter-article 13
  python search_test.py -q "休暇" --filter-article "第13条"
  python search_test.py -q "育児" --filter-chapter "育児・介護"
  python search_test.py -q "出張" --filter-pdf "旅費" -k 5
  python search_test.py -q "給与" --filter-article 15 --filter-chapter "賃金"

対話モードのインライン指定:
  python search_test.py
  🔍 検索: 育児休業  filter-chapter:育児・介護
  🔍 検索: 給与  filter-article:15  filter-pdf:就業規則
""",
    )
    p.add_argument("--query", "-q", default=None,
                   help="検索クエリ（省略時は対話モード）")
    p.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K,
                   help=f"表示件数（デフォルト: {DEFAULT_TOP_K}）")
    p.add_argument("--filter-article", default=None, metavar="ARTICLE",
                   help="条項番号フィルタ。「第13条」「13条」「13」すべて対応。例: 13")
    p.add_argument("--filter-chapter", default=None, metavar="CHAPTER",
                   help="章タイトルで絞り込み（部分一致）。例: \"育児・介護\"")
    p.add_argument("--filter-pdf", default=None, metavar="PDF",
                   help="PDFファイル名で絞り込み（部分一致）。例: \"就業規則\"")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# エントリーポイント
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Windows CP932 コンソールでも日本語を正しく出力する
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    # BM25 インデックス読み込み
    print("BM25 インデックス読み込み中...")
    bm25, docs = load_bm25()
    print(f"  ✓ {len(docs)} チャンク  準備完了\n")

    # コマンドライン引数からフィルタを構築
    cli_filters = Filters(
        article=args.filter_article,
        chapter=args.filter_chapter,
        pdf=args.filter_pdf,
    )

    # ── 1回だけ検索して終了 ───────────────────────────────────────────────────
    if args.query:
        results = search(bm25, docs, args.query, args.top_k, cli_filters)
        print_results(results, args.query, cli_filters)
        return

    # ── 対話モード ─────────────────────────────────────────────────────────────
    print(f"対話モード（top_k={args.top_k}）")
    if not cli_filters.is_empty():
        print(f"フィルタ（固定）: {cli_filters.describe()}")
    print("  通常入力:          育児休業")
    print("  インラインフィルタ: 育児休業  filter-article:13  filter-chapter:育児・介護")
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

        query, inline_filters = parse_inline(raw)

        if not query:
            print("  [INFO] クエリが空です。")
            continue

        # インラインフィルタが CLI フィルタより優先
        active_filters = Filters(
            article=inline_filters.article_raw or cli_filters.article_raw,
            chapter=inline_filters.chapter_raw or cli_filters.chapter_raw,
            pdf=inline_filters.pdf_raw         or cli_filters.pdf_raw,
        )

        results = search(bm25, docs, query, args.top_k, active_filters)
        print_results(results, query, active_filters)


if __name__ == "__main__":
    main()
