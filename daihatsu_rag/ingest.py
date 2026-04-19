#!/usr/bin/env python3
"""
ingest.py — Daihatsu Regulations RAG Ingestion Script

cleaned_final/*.md を解析し、以下を生成する:
  1. BM25 インデックス (bm25_index.pkl / bm25_docs.pkl)  ← 常に生成・検索のメイン
  2. ChromaDB コレクション (http://localhost:8000)         ← デフォルトON、--skip-chroma でOFF

ChromaDB接続は認証なし (HttpClient のみ、Settings/Token 一切不使用)。
langchain-chroma / langchain-ollama は不要。chromadb 直接API を使用。

Usage:
    python ingest.py               # BM25 + ChromaDB 両方
    python ingest.py --reset       # コレクション削除して再投入
    python ingest.py --skip-chroma # BM25 pkl のみ（ChromaDB 不要）
    python ingest.py --dry-run     # パース確認のみ（投入なし）
    python ingest.py -v            # デバッグログ表示
"""

import re
import sys
import pickle
import argparse
import logging
from pathlib import Path

from tqdm import tqdm

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
CLEANED_FINAL_DIR = BASE_DIR / "cleaned_final"
BM25_INDEX_PATH   = BASE_DIR / "bm25_index.pkl"
BM25_DOCS_PATH    = BASE_DIR / "bm25_docs.pkl"

CHROMA_HOST     = "localhost"
CHROMA_PORT     = 8000
COLLECTION_NAME = "daihatsu_regulations"

DEFAULT_BATCH_SIZE = 64
MIN_CHUNK_LENGTH   = 20

# ─── Regex ────────────────────────────────────────────────────────────────────
RE_CHAPTER     = re.compile(r'^第\d+章')
RE_ARTICLE_NUM = re.compile(r'^第(\d+)条')
RE_TITLE_ONLY  = re.compile(r'^[（(].+[）)]$')
RE_INLINE_ART  = re.compile(r'第(\d+)条')


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown Parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_markdown_to_chunks(md_path: Path) -> list[dict]:
    """
    Markdown を条項単位の dict リストへ変換する。

    各 dict のキー:
        text, pdf_name, doc_title, chapter,
        article_number, article_title, doc_type, file_path

    対応パターン:
      Pattern A: # 第X条（タイトル）     ← 就業規則スタイル
      Pattern B: # （タイトル）\n# 第X条  ← ハラスメント規程スタイル
      Pattern C: # （タイトル）\n第X条 … ← 安全衛生規程スタイル（インライン条番号）
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError as e:
        log.error(f"読み込みエラー: {md_path} — {e}")
        return []

    lines             = text.splitlines()
    pdf_name          = md_path.stem + ".pdf"
    doc_title         = ""
    current_chapter   = ""
    current_art_num   = ""
    current_art_title = ""
    pending_title     = ""      # Pattern B: （タイトル）を一時保持
    chunk_lines: list[str] = []
    chunks: list[dict]     = []

    def _flush():
        nonlocal chunk_lines, current_art_num, current_art_title, pending_title
        if not chunk_lines:
            return
        content = "\n".join(chunk_lines).strip()
        if len(content) < MIN_CHUNK_LENGTH:
            chunk_lines = []
            return
        # Pattern C: 条番号がメタデータ未設定 → 本文から抽出
        art_num = current_art_num
        if not art_num:
            m = RE_INLINE_ART.search(content)
            if m:
                art_num = f"第{m.group(1)}条"
        chunks.append({
            "text":           content,
            "pdf_name":       pdf_name,
            "doc_title":      doc_title,
            "chapter":        current_chapter,
            "article_number": art_num,
            "article_title":  current_art_title,
            "doc_type":       "regulation",
            "file_path":      str(md_path),
        })
        chunk_lines = []

    for line in lines:
        h = re.match(r'^(#{1,6})\s+(.+)$', line)
        if not h:
            chunk_lines.append(line)
            continue

        ht             = h.group(2).strip()
        is_chapter     = bool(RE_CHAPTER.match(ht))
        is_article_num = bool(RE_ARTICLE_NUM.match(ht))
        is_title_only  = bool(RE_TITLE_ONLY.match(ht)) and not is_article_num

        # ── 章見出し ────────────────────────────────────────────────
        if is_chapter:
            _flush()
            current_chapter   = ht
            current_art_num   = ""
            current_art_title = ""
            pending_title     = ""
            continue

        # ── 条番号見出し（第X条 / 第X条（タイトル）） ──────────────
        if is_article_num:
            m = RE_ARTICLE_NUM.match(ht)
            art_num = f"第{m.group(1)}条"
            if pending_title:
                # Pattern B: 直前の（タイトル）と同一チャンクにマージ
                current_art_num   = art_num
                current_art_title = f"{pending_title} {ht}"
                chunk_lines.append(line)
                pending_title = ""
            else:
                # Pattern A / 単独条番号
                _flush()
                current_art_num   = art_num
                current_art_title = ht
                pending_title     = ""
                chunk_lines       = [line]
            continue

        # ── タイトルのみ見出し （目的）など ──────────────────────────
        if is_title_only:
            _flush()
            current_art_num   = ""
            current_art_title = ht
            pending_title     = ht
            chunk_lines       = [line]
            continue

        # ── その他見出し（文書タイトル / 附則 / 別表 など）──────────
        if not doc_title:
            doc_title = ht   # 最初の非章・非条項見出し → 文書タイトル
        else:
            _flush()
            current_art_num   = ""
            current_art_title = ht
            pending_title     = ""
            chunk_lines       = [line]

    _flush()
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# BM25
# ═══════════════════════════════════════════════════════════════════════════════

def build_bm25_index(chunks: list[dict]) -> bool:
    """
    BM25 インデックスを pkl ファイルへ保存する。
    日本語は文字単位トークナイズ（形態素解析なし）。
    戻り値: 成功なら True
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        log.warning("rank-bm25 未インストール → BM25 スキップ (pip install rank-bm25)")
        return False

    log.info(f"BM25 インデックス構築: {len(chunks)} チャンク")
    tokenized = [list(c["text"]) for c in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    log.info(f"BM25 保存完了 → {BM25_INDEX_PATH.name} / {BM25_DOCS_PATH.name}")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# ChromaDB  (認証なし・chromadb 直接API・langchain 不使用)
# ═══════════════════════════════════════════════════════════════════════════════

def get_chroma_client():
    """
    認証なしの ChromaDB HTTP クライアントを返す。
    Settings / TokenAuthentication は一切使用しない。
    """
    try:
        import chromadb
    except ImportError:
        log.error("chromadb 未インストール: pip install chromadb")
        sys.exit(1)

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        log.info(f"ChromaDB 接続成功: http://{CHROMA_HOST}:{CHROMA_PORT}")
        return client
    except Exception as e:
        log.error(f"ChromaDB 接続失敗: {e}")
        log.error("  → docker-compose up -d でサーバーを起動してください")
        sys.exit(1)


def handle_reset(client, force_reset: bool) -> None:
    """既存コレクションの削除判断（--reset なら無条件削除）。"""
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        return

    if force_reset:
        client.delete_collection(COLLECTION_NAME)
        log.info(f"コレクション '{COLLECTION_NAME}' を削除しました（--reset）")
        return

    ans = input(f"\n⚠️  '{COLLECTION_NAME}' が存在します。削除して再投入しますか？ [y/N]: ").strip().lower()
    if ans == "y":
        client.delete_collection(COLLECTION_NAME)
        log.info("削除しました")
    else:
        log.info("既存コレクションへ追記します（重複注意）")


def ingest_to_chroma(client, chunks: list[dict], batch_size: int) -> None:
    """
    chromadb 直接API でチャンクを投入する。
    langchain-chroma / langchain-ollama は不使用。
    埋め込みは chromadb のデフォルト (all-MiniLM-L6-v2, 初回のみ自動DL)。
    """
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Daihatsu regulations chunks"},
    )

    total = len(chunks)
    log.info(f"ChromaDB 投入開始: {total} チャンク")

    with tqdm(total=total, desc="ChromaDB 投入", unit="chunk") as pbar:
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            ids      = [f"chunk_{i + j}" for j in range(len(batch))]
            docs     = [c["text"] for c in batch]
            # metadata は文字列値のみ許可（ChromaDB制約）
            metadatas = [
                {k: str(v) for k, v in c.items() if k != "text"}
                for c in batch
            ]
            try:
                collection.add(ids=ids, documents=docs, metadatas=metadatas)
            except Exception as e:
                log.error(f"バッチ投入エラー (offset={i}): {e}")
                raise
            pbar.update(len(batch))

    log.info(f"ChromaDB 投入完了: 合計 {collection.count()} チャンク")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daihatsu Regulations → BM25 + ChromaDB 投入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python ingest.py               # 通常投入（BM25 + ChromaDB）
  python ingest.py --reset       # コレクション削除して再投入
  python ingest.py --skip-chroma # BM25 pkl のみ生成
  python ingest.py --dry-run     # パース結果だけ確認
""",
    )
    p.add_argument("--reset",       action="store_true", help="既存コレクションを削除して再投入")
    p.add_argument("--skip-chroma", action="store_true", help="ChromaDB 投入をスキップ（BM25 のみ）")
    p.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE, metavar="N",
                   help=f"ChromaDB 投入バッチサイズ (デフォルト: {DEFAULT_BATCH_SIZE})")
    p.add_argument("--input-dir",   type=Path, default=CLEANED_FINAL_DIR, metavar="DIR",
                   help=f"Markdown ディレクトリ (デフォルト: {CLEANED_FINAL_DIR})")
    p.add_argument("--dry-run",     action="store_true", help="パース確認のみ（ChromaDB/BM25 投入なし）")
    p.add_argument("--verbose", "-v", action="store_true", help="デバッグログ表示")
    return p.parse_args()


def main() -> None:
    # Windows CP932 コンソールでも日本語・記号を正しく出力する
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 入力ディレクトリ確認 ──────────────────────────────────────
    input_dir: Path = args.input_dir
    if not input_dir.exists():
        log.error(f"ディレクトリが見つかりません: {input_dir}")
        sys.exit(1)

    md_files = sorted(input_dir.glob("*.md"))
    if not md_files:
        log.error(f"Markdown ファイルが見つかりません: {input_dir}/*.md")
        sys.exit(1)

    log.info(f"対象: {len(md_files)} ファイル")

    # ── Markdown パース ───────────────────────────────────────────
    all_chunks: list[dict] = []
    parse_errors = 0

    for md_path in tqdm(md_files, desc="Markdown パース", unit="file"):
        try:
            chunks = parse_markdown_to_chunks(md_path)
            all_chunks.extend(chunks)
            log.debug(f"  {md_path.name}: {len(chunks)} チャンク")
        except Exception as e:
            log.error(f"パースエラー: {md_path.name} — {e}")
            parse_errors += 1

    if not all_chunks:
        log.error("チャンクが1件も生成されませんでした")
        sys.exit(1)

    # ── パース結果サマリ ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  パース完了: {len(all_chunks)} チャンク / {len(md_files)} ファイル")
    if parse_errors:
        print(f"  ⚠️  エラー: {parse_errors} ファイル")
    print()
    file_counts: dict[str, int] = {}
    for c in all_chunks:
        fn = c.get("pdf_name", "?")
        file_counts[fn] = file_counts.get(fn, 0) + 1
    for fn, cnt in sorted(file_counts.items()):
        print(f"  {fn}: {cnt} チャンク")
    print("─" * 60 + "\n")

    # ── Dry-run ───────────────────────────────────────────────────
    if args.dry_run:
        log.info("--dry-run: 投入をスキップします")
        print("=== サンプル（先頭3件）===")
        for c in all_chunks[:3]:
            print(f"\n  [{c['article_number']}] {c['chapter']} | {c['article_title']}")
            print(f"  {c['text'][:150]}{'…' if len(c['text']) > 150 else ''}")
            print()
        return

    # ── BM25 インデックス構築 ─────────────────────────────────────
    bm25_ok = build_bm25_index(all_chunks)

    # ── ChromaDB 投入 ─────────────────────────────────────────────
    chroma_ok = False
    if not args.skip_chroma:
        client = get_chroma_client()
        handle_reset(client, args.reset)
        ingest_to_chroma(client, all_chunks, args.batch_size)
        chroma_ok = True
    else:
        log.info("--skip-chroma: ChromaDB 投入をスキップ")

    # ── 完了サマリ ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  ✓ 完了: {len(all_chunks)} チャンク")
    print(f"  BM25   : {'✓ ' + str(BM25_INDEX_PATH.name) if bm25_ok else '✗ スキップ'}")
    print(f"  ChromaDB: {'✓ ' + COLLECTION_NAME if chroma_ok else '✗ スキップ'}")
    print("═" * 60 + "\n")

    print("次のステップ:")
    print("  python search_test.py --query \"試用期間\"")
    print("  python search_test.py  # 対話モード\n")


if __name__ == "__main__":
    main()
