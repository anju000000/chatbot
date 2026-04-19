import re
import os
from pathlib import Path
from tqdm import tqdm

# ================== 設定 ==================
INPUT_FOLDER = "./output_markdown"
OUTPUT_FOLDER = "./cleaned_markdown_v2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_text(text: str) -> str:
    # 1. すべての空白（半角・全角・タブ）を一旦統一
    text = re.sub(r'[\u3000\t ]+', ' ', text)   # 全角スペース・タブ・半角スペース → 1つの半角スペース

    # 2. 行ごとに処理（これが重要）
    lines = text.split('\n')
    cleaned_lines = []
    prev_line_was_heading = False

    for line in lines:
        line = line.strip()                     # 行頭・行末の空白を完全削除

        if not line:                            # 空行は1つだけ残す
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
            continue

        # 見出し処理（# で始まる行）
        if line.startswith('#'):
            # #の後ろに必ず半角スペースを1つ入れる
            line = re.sub(r'^(#+)\s*', r'\1 ', line)
            prev_line_was_heading = True
        else:
            prev_line_was_heading = False

        # 3. 条項番号の正規化（より強力に）
        line = re.sub(r'第\s*([0-9０-９一二三四五六七八九十]+)\s*条', 
                     lambda m: f"第{normalize_number(m.group(1))}条", line)

        # 4. 文章内の不自然なスペース除去（句読点の前後）
        line = re.sub(r'\s+([、。；：）】」』])', r'\1', line)   # 句読点の前にスペースがあったら削除
        line = re.sub(r'([、。；：（【「『])\s+', r'\1', line)   # 句読点の後にスペースがあったら削除

        # 5. 連続する句読点を調整
        line = re.sub(r'([、。])\s*([、。])', r'\1\2', line)

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # 6. 連続空行を最大2つまでに制限
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 7. 全体の余分なスペースを最後に掃除
    text = re.sub(r' +', ' ', text)

    return text.strip()

def normalize_number(num: str) -> str:
    """全角数字・漢数字を半角に変換"""
    num = num.translate(str.maketrans('０１２３４５６７８９一二三四五六七八九十', '0123456789一二三四五六七八九十'))
    # 漢数字の簡単変換（必要に応じて拡張）
    return num

# ================== メイン ==================
if __name__ == "__main__":
    md_files = list(Path(INPUT_FOLDER).glob("*.md"))
    
    print(f"クレンジング v2 開始: {len(md_files)} ファイル\n")
    
    success = 0
    for md_file in tqdm(md_files):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            cleaned = clean_text(content)
            
            output_path = Path(OUTPUT_FOLDER) / md_file.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            
            success += 1
        except Exception as e:
            print(f"❌ エラー {md_file.name}: {e}")
    
    print(f"\n🎉 クレンジング v2 完了！")
    print(f"成功: {success} / {len(md_files)}")
    print(f"出力先 → {OUTPUT_FOLDER}")