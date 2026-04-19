from docling.document_converter import DocumentConverter
import os
import json
from pathlib import Path

# ================== 設定 ==================
PDF_FOLDER = "./pdfs"                    # ← ここにPDFを入れる
OUTPUT_MD_FOLDER = "./output_markdown"   # Markdown出力先
OUTPUT_JSON_FOLDER = "./output_json"     # 構造情報（JSON）出力先（オプション）

# フォルダ作成
os.makedirs(OUTPUT_MD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)

# DocumentConverterの初期化（規約PDF向けにシンプルに）
converter = DocumentConverter()

# ================== メイン処理 ==================
def convert_pdf_to_markdown(pdf_path: str):
    print(f"処理中: {pdf_path.name if isinstance(pdf_path, Path) else pdf_path}")
    
    # PDF変換
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Markdown出力
    markdown_text = doc.export_to_markdown()
    
    pdf_name = Path(pdf_path).stem
    md_path = os.path.join(OUTPUT_MD_FOLDER, f"{pdf_name}.md")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    
    print(f"✅ Markdown保存完了: {md_path}")
    
    # JSON出力（構造情報が必要な場合）
    json_path = os.path.join(OUTPUT_JSON_FOLDER, f"{pdf_name}.json")
    doc_dict = doc.export_to_dict()          # ← ここを修正
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON保存完了: {json_path}")
    
    return markdown_text

# ================== バッチ実行 ==================
if __name__ == "__main__":
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"エラー: {PDF_FOLDER} フォルダの中にPDFが見つかりません。")
        print("pdfsフォルダを作成して、まずは1冊のPDFを入れて実行してください。")
    else:
        print(f"発見したPDF: {len(pdf_files)} 冊")
        for pdf_file in pdf_files:
            convert_pdf_to_markdown(pdf_file)
        
        print("\n🎉 すべてのPDFの変換が完了しました！")
        print(f"Markdown → {OUTPUT_MD_FOLDER}")
        print(f"JSON（構造情報） → {OUTPUT_JSON_FOLDER}")