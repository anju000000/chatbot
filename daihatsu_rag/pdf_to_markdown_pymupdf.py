import pymupdf4llm
import os
from pathlib import Path
from tqdm import tqdm   # 進捗バーを表示

# ================== 設定 ==================
PDF_FOLDER = "./pdfs"
OUTPUT_MD_FOLDER = "./output_markdown"

os.makedirs(OUTPUT_MD_FOLDER, exist_ok=True)

def convert_pdf_to_markdown(pdf_path: str):
    pdf_name = Path(pdf_path).stem
    print(f"処理中: {pdf_name}")
    
    try:
        md_text = pymupdf4llm.to_markdown(
            pdf_path,
            write_images=False,
            page_chunks=False
        )
        
        md_path = os.path.join(OUTPUT_MD_FOLDER, f"{pdf_name}.md")
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        
        print(f"✅ 完了: {pdf_name}.md")
        return True
        
    except Exception as e:
        print(f"❌ エラー [{pdf_name}]: {e}")
        return False

# ================== メイン ==================
if __name__ == "__main__":
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print("pdfsフォルダにPDFが見つかりません。")
    else:
        print(f"合計 {len(pdf_files)}冊 のPDFを処理します...\n")
        
        success_count = 0
        for pdf_file in tqdm(pdf_files):
            if convert_pdf_to_markdown(str(pdf_file)):
                success_count += 1
        
        print("\n🎉 バッチ処理完了！")
        print(f"成功: {success_count} / {len(pdf_files)} 冊")
        print(f"出力先: {OUTPUT_MD_FOLDER}")