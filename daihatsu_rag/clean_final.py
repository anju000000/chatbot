"""
CLAUDE.mdルールに従い cleaned_markdown_v2 → cleaned_final へクレンジング
"""
import os
import re

INPUT_DIR = os.path.join(os.path.dirname(__file__), "cleaned_markdown_v2")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cleaned_final")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_markdown(text: str) -> str:
    lines = text.splitlines()

    # ① 行頭・行末スペース除去
    lines = [l.rstrip() for l in lines]
    lines = [l.lstrip() for l in lines]

    # ② 見出しの **N** → N を解除し、## → # に統一、スペースゆれを吸収
    #    例: "## 第 **1** 章 総則"  → "# 第1章 総則"
    #    例: "## 第 **12** 条（目的）" → "# 第12条（目的）"
    fixed = []
    for l in lines:
        # **数字** → 数字
        l = re.sub(r'\*\*(\d+)\*\*', r'\1', l)
        # 見出し行: ## → #（ただし既に # 単独の行は維持）
        if l.startswith('## '):
            l = '# ' + l[3:]
        # 見出し内の「第 N 章」「第 N 条」の余分スペースを除去
        l = re.sub(r'(第)\s+(\d+)\s*(章|条|項)', r'\1\2\3', l)
        fixed.append(l)
    lines = fixed

    # ③ 連続した (見出しなし・箇条書きなし・空行なし) の行を結合
    #    日本語テキストで行末が句読点・閉じ括弧以外なら次行と結合
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 見出し・箇条書き・空行はそのまま
        if (line == '' or
                line.startswith('#') or
                re.match(r'^(\d+\.|[-*・])\s', line) or
                re.match(r'^\(?\d+\)', line)):
            merged.append(line)
            i += 1
            continue

        # 次行があり、次行も通常テキストなら結合候補
        while i + 1 < len(lines):
            next_line = lines[i + 1]
            # 次行が空・見出し・箇条書きなら結合しない
            if (next_line == '' or
                    next_line.startswith('#') or
                    re.match(r'^(\d+\.|[-*・])\s', next_line) or
                    re.match(r'^\(?\d+\)', next_line)):
                break
            # 現行が句読点・閉じ記号で終わるなら結合しない
            if re.search(r'[。、．，）」』】\]]\s*$', line):
                break
            # 結合
            line = line + next_line
            i += 1

        merged.append(line)
        i += 1
    lines = merged

    # ③' 空行をまたいだ単語分断を結合
    #   例: "...を重点的に行\n\nう。" → "...を重点的に行う。"
    #   条件: 通常本文行 + 行末が句読点なし + 空行1行 + 次行が短い続き（≤8文字）
    IS_LIST = lambda l: (re.match(r'^(\d+\.|[-*・])\s', l) or re.match(r'^\(?\d+\)', l))
    result2 = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (line != '' and
                not line.startswith('#') and
                not IS_LIST(line) and
                not re.search(r'[。、．，）」』】\]]\s*$', line) and
                i + 2 < len(lines) and
                lines[i + 1] == '' and
                lines[i + 2] != '' and
                not lines[i + 2].startswith('#') and
                not IS_LIST(lines[i + 2]) and
                len(lines[i + 2]) <= 8):
            result2.append(line + lines[i + 2])
            i += 3
        else:
            result2.append(line)
            i += 1
    lines = result2

    # ④ 行内の連続スペース（半角）を1つに統一
    lines = [re.sub(r'[ \t]{2,}', ' ', l) for l in lines]

    # ⑤ 全角スペースの連続を1つに統一（通常は不要だが念のため）
    lines = [re.sub(r'　{2,}', '　', l) for l in lines]

    # ⑥ 句読点前後の不要スペース除去
    lines = [re.sub(r'\s+([。、．，）」』】])', r'\1', l) for l in lines]
    lines = [re.sub(r'([（「『【])\s+', r'\1', l) for l in lines]

    # ⑦ 数字と日本語の間の不要スペース除去（例: "1 週間" → "1週間"）
    #    ただし箇条書き番号 "1. " は保持
    lines = [re.sub(r'(\d)\s+([\u3040-\u30ff\u4e00-\u9fff])', r'\1\2', l) for l in lines]
    lines = [re.sub(r'([\u3040-\u30ff\u4e00-\u9fff])\s+(\d)', r'\1\2', l) for l in lines]

    # ⑦' 本文行（見出し以外）のCJK文字間スペースを全て除去
    #   日本語本文にスペースは存在しないため、全てPDF抽出アーティファクト
    #   見出し行（# から始まる）は会社名とタイトルの区切りが意図的なため除外
    CJK = r'[\u3040-\u30ff\u4e00-\u9fff\uff00-\uffef\u300c-\u300f]'
    def remove_cjk_spaces(line: str) -> str:
        if line.startswith('#'):
            return line
        prev = None
        while prev != line:
            prev = line
            line = re.sub(f'({CJK}) ({CJK})', r'\1\2', line)
        return line
    lines = [remove_cjk_spaces(l) for l in lines]

    # ⑦'' 行頭の条番号直後の日本語テキストにスペースを復元
    #   例: "第3条この規程は" → "第3条 この規程は"（行頭のみ。文中の条番号は対象外）
    lines = [re.sub(r'^(第\d+[条項])([\u3040-\u30ff\u4e00-\u9fff])', r'\1 \2', l) for l in lines]

    # ⑧ 連続空行を最大2行に制限
    result = []
    blank_count = 0
    for l in lines:
        if l == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(l)
        else:
            blank_count = 0
            result.append(l)

    # ⑨ 末尾の余分な空行を除去
    while result and result[-1] == '':
        result.pop()

    return '\n'.join(result) + '\n'


def diff_summary(original: str, cleaned: str) -> dict:
    orig_lines = original.splitlines()
    clean_lines = cleaned.splitlines()
    return {
        "元行数": len(orig_lines),
        "後行数": len(clean_lines),
        "削減行数": len(orig_lines) - len(clean_lines),
    }


results = []
md_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.md')]

for filename in sorted(md_files):
    in_path = os.path.join(INPUT_DIR, filename)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(in_path, encoding='utf-8') as f:
        original = f.read()

    cleaned = clean_markdown(original)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    summary = diff_summary(original, cleaned)
    results.append((filename, summary))
    print(f"[OK] {filename}  {summary}")

print(f"\n完了: {len(results)} ファイルを cleaned_final に保存しました。")
