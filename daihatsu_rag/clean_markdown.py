import re
import os
from pathlib import Path
from tqdm import tqdm

# ================== 設定 ==================
INPUT_FOLDER = "./output_markdown"      # PyMuPDF4LLMで出力したフォルダ
OUTPUT_FOLDER = "./cleaned_markdown"    # クレンジング後の出力先

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clean_text(text: str) -> str:
    # 1. 連続する空白（半角・全角）を1つにまとめる
    text = re.sub(r'[ \u3000\t]+', ' ', text)          # 半角スペース、全角スペース、タブ → 1つの半角スペース
    
    # 2. 行頭・行末の余分なスペースを削除
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()                            # 行頭・行末の空白削除
        
        # 見出し行（# で始まる）はスペースを1つだけ入れる
        if line.startswith('#'):
            line = re.sub(r'^#+\s*', lambda m: m.group(0).rstrip() + ' ', line)
        
        # 空行は1つだけ残す（連続空行を1つに）
        if not line and cleaned_lines and cleaned_lines[-1] == '':
            continue
        
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # 3. 条項番号の正規化（第１条 → 第1条、第 15 条 → 第15条 など）
    # 「第」+ 数字（全角・半角）+ 「条」
    text = re.sub(r'第\s*([0-9０-９]+)\s*条', lambda m: f"第{normalize_number(m.group(1))}条", text)
    
    # 4. 軽い文章内正規化（句読点の前後のスペース調整など）
    text = re.sub(r'([、。])\s+', r'\1', text)          # 「、 」や「。 」→「、」「。」
    text = re.sub(r'\s+([、。])', r'\1', text)
    
    # 5. 連続する改行を最大2つまでに制限（見出しの後などは残す）
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def normalize_number(num_str: str) -> str:
    """全角数字を半角に変換"""
    return num_str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))

# ================== メイン処理 ==================
if __name__ == "__main__":
    md_files = list(Path(INPUT_FOLDER).glob("*.md"))
    
    if not md_files:
        print(f"エラー: {INPUT_FOLDER} に .md ファイルが見つかりません。")
    else:
        print(f"クレンジング開始: {len(md_files)} ファイル\n")
        
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
        
        print(f"\n🎉 クレンジング完了！")
        print(f"成功: {success} / {len(md_files)} ファイル")
        print(f"出力先: {OUTPUT_FOLDER}")