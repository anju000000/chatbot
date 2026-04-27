import logging
import pickle
from pathlib import Path

from app.core.config import Settings

logger = logging.getLogger(__name__)

# BM25 インデックスはプロセス起動時に1回だけ読み込む（リクエストごとの I/O を避ける）
_bm25_cache: tuple | None = None

# ── SudachiPy 初期化（ingest.py と同じロジックで必ずトークナイザを揃える）──────
_sudachi = None
_SUDACHI_MODE = None
try:
    from sudachipy import tokenizer as _st, dictionary as _sd
    _sudachi = _sd.Dictionary().create()
    _SUDACHI_MODE = _st.Tokenizer.SplitMode.C
    logger.info("SudachiPy: 形態素解析トークナイザを使用します")
except Exception:
    logger.info("SudachiPy: 未インストール — 文字ユニグラム+バイグラムにフォールバックします")


def _tokenize(text: str) -> list[str]:
    """ingest.py と同じトークナイザ。SudachiPy があれば形態素解析、なければ N-gram。"""
    if _sudachi is not None:
        return [m.surface() for m in _sudachi.tokenize(text, _SUDACHI_MODE) if m.surface().strip()]
    chars = [c for c in text if not c.isspace()]
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return chars + bigrams


# ── 社内規程向け同義語辞書（クエリ時のみ適用）────────────────────────────────
_SYNONYMS: dict[str, list[str]] = {
    "有給": ["年休", "年次有給休暇", "有給休暇"],
    "年休": ["有給", "年次有給休暇", "有給休暇"],
    "年次有給休暇": ["有給", "年休", "有給休暇"],
    "有給休暇": ["有給", "年休", "年次有給休暇"],
    "産休": ["産前産後休業", "産前休業", "産後休業"],
    "産前産後休業": ["産休", "産前休業", "産後休業"],
    "育休": ["育児休業", "育児・介護休業"],
    "育児休業": ["育休", "育児・介護休業"],
    "給与": ["賃金", "給料"],
    "賃金": ["給与", "給料"],
    "給料": ["給与", "賃金"],
    "残業": ["時間外労働", "時間外勤務"],
    "時間外労働": ["残業", "時間外勤務"],
    "解雇": ["退職", "雇用終了"],
    "退職": ["辞職", "退社"],
    "出張": ["旅費", "出張旅費"],
    "旅費": ["出張", "出張旅費"],
    "通勤": ["通勤手当", "交通費"],
    "通勤手当": ["通勤", "交通費"],
    "慶弔": ["慶弔見舞金", "弔慰金"],
    "見舞金": ["慶弔見舞金", "弔慰金"],
    "ハラスメント": ["セクハラ", "パワハラ"],
    "セクハラ": ["ハラスメント", "性的嫌がらせ"],
    "パワハラ": ["ハラスメント"],
    "安全衛生": ["健康管理", "安全管理"],
    "休職": ["休業"],
    "窓口": ["相談窓口", "連絡先", "相談先"],
    "相談窓口": ["窓口", "連絡先", "相談先"],
    "本採用": ["解雇", "不適格", "試用期間"],
    "見送り": ["解雇", "不適格"],
    "対象家族": ["配偶者", "父母", "祖父母", "兄弟姉妹"],
    "介護休暇": ["介護", "介護休業", "育児・介護休業"],
    "介護休業": ["介護休暇", "介護", "育児・介護休業"],
    "子の看護休暇": ["看護休暇", "子の看護", "育児・介護休業"],
    "看護休暇": ["子の看護休暇", "子の看護"],
    "懲戒": ["懲戒処分", "制裁", "譴責", "降格", "諭旨解雇", "懲戒解雇"],
    "懲戒処分": ["懲戒", "制裁"],
    "試用期間": ["試用", "本採用", "採用"],
    "フレックス": ["フレックスタイム", "フレキシブルタイム"],
    "フレックスタイム": ["フレックス", "フレキシブルタイム"],
    "所定労働時間": ["労働時間", "勤務時間"],
    "労働時間": ["所定労働時間", "勤務時間"],
    "深夜": ["深夜労働", "深夜割増"],
    "休日": ["法定休日", "所定休日", "公休"],
    "特別休暇": ["特休", "慶弔休暇"],
    "慶弔休暇": ["特別休暇", "特休"],
}


def _expand_tokens(tokens: list[str]) -> list[str]:
    """同義語を追加したトークンリストを返す（重複なし・順序保持）。"""
    seen: set[str] = set(tokens)
    expanded = list(tokens)
    for token in tokens:
        for syn in _SYNONYMS.get(token, []):
            if syn not in seen:
                seen.add(syn)
                expanded.append(syn)
    return expanded


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


def _bm25_search(
    bm25, docs: list[dict], query: str, top_k: int, apply_min_score: bool = True
) -> list[dict]:
    tokens = _expand_tokens(_tokenize(query))
    scores = bm25.get_scores(tokens)
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # apply_min_score=True（単独BM25モード）のときだけ低スコアを除外する。
    # hybrid モードでは RRF が最終絞り込みを担うため無効化する。
    top_score = scored[0][1] if scored else 0
    min_score = max(1.0, top_score * 0.5) if apply_min_score else 0.0

    results = []
    for idx, score in scored[:top_k]:
        if score < min_score:
            break
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


_RRF_K = 60  # RRF 標準定数（変更不要）


def _doc_key(r: dict) -> str:
    return f"{r.get('pdf_name', '')}|{r.get('article_number', '')}|{r['text'][:40]}"


def _rrf_merge(bm25_results: list[dict], vector_results: list[dict], top_k: int) -> list[dict]:
    """Reciprocal Rank Fusion で BM25 とベクトル検索結果を統合する。

    RRF は順位のみを使うためスコアのスケール差を回避できる。
    どちらか一方にしかヒットしない文書も適切に扱われる。
    """
    rrf_scores: dict[str, float] = {}
    key_to_doc: dict[str, dict] = {}

    for rank, r in enumerate(bm25_results, start=1):
        key = _doc_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        if key not in key_to_doc:
            key_to_doc[key] = {**r, "mode": "BM25"}

    for rank, r in enumerate(vector_results, start=1):
        key = _doc_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        if key in key_to_doc:
            key_to_doc[key]["mode"] = "HYBRID"  # 両検索にヒット → 最も信頼度が高い
        else:
            key_to_doc[key] = {**r, "mode": "VECTOR"}

    results = []
    for key in sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]:
        doc = {**key_to_doc[key], "score": round(rrf_scores[key], 6)}
        results.append(doc)

    return results


import re as _re

_YES_NO_PATTERNS = [
    (_re.compile(r'(.+)てはいけません(か？?)?$'), r'\1について規程を教えてください'),
    (_re.compile(r'(.+)することはあります(か？?)?$'), r'\1について規程を教えてください'),
    (_re.compile(r'(.+)[でき|でき]ます(か？?)?$'), r'\1について規程を教えてください'),
    (_re.compile(r'(.+)(は|も)あります(か？?)?$'), r'\1について規程を教えてください'),
]


def _normalize_query(query: str) -> str:
    """Yes/No型の疑問文を規程検索に適した形式に正規化する（BM25検索専用）。"""
    for pattern, replacement in _YES_NO_PATTERNS:
        normalized = pattern.sub(replacement, query.strip())
        if normalized != query.strip():
            logger.debug("Query normalized: %r -> %r", query, normalized)
            return normalized
    return query


def _get_definition_chunks(docs: list[dict]) -> list[dict]:
    """各規程の第2条（定義）チャンクを返す。対象家族・用語定義の取りこぼしを防ぐ。"""
    seen: set[str] = set()
    result = []
    for d in docs:
        if d.get("article_number") == "第2条":
            key = d.get("pdf_name", "")
            if key not in seen:
                seen.add(key)
                result.append({**d, "score": 0.0, "mode": "DEF"})
    return result


def retrieve(query: str, settings: Settings) -> list[dict]:
    """クエリに関連するチャンクを返す。rag_mode に応じて BM25 / vector / hybrid を切り替え。"""
    mode = settings.rag_mode.lower()
    top_k = settings.rag_top_k
    normalized_query = _normalize_query(query)

    if mode == "bm25":
        bm25, docs = _load_bm25(settings)
        search_results = _bm25_search(bm25, docs, normalized_query, top_k)
        def_chunks = _get_definition_chunks(docs)
        # 定義条文を末尾に追加（重複除去）
        existing_keys = {r["text"][:60] for r in search_results}
        for d in def_chunks:
            if d["text"][:60] not in existing_keys:
                search_results.append(d)
        return search_results

    if mode == "vector":
        try:
            return _vector_search(settings, normalized_query, top_k)
        except Exception as e:
            logger.warning("vector_search failed: %s", e)
            return []

    if mode == "hybrid":
        pool = top_k * 3

        bm25, docs = _load_bm25(settings)
        bm25_results = _bm25_search(bm25, docs, normalized_query, pool, apply_min_score=False)

        vector_results: list[dict] = []
        try:
            vector_results = _vector_search(settings, normalized_query, pool)
        except Exception as e:
            logger.warning("vector_search skipped (hybrid fallback to BM25 only): %s", e)

        results = _rrf_merge(bm25_results, vector_results, top_k)
        def_chunks = _get_definition_chunks(docs)
        existing_keys = {r["text"][:60] for r in results}
        for d in def_chunks:
            if d["text"][:60] not in existing_keys:
                results.append(d)
        return results

    raise ValueError(f"Unknown rag_mode: {settings.rag_mode!r}")
