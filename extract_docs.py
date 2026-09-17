

import os
import io
import json
import base64
import hashlib
import glob
import unicodedata

import fitz  # PyMuPDF
from openai import OpenAI

client = OpenAI()  # đọc OPENAI_API_KEY từ biến môi trường

SOURCE_DIR = "database/source_pdfs"
OUTPUT_JSON = "database/extracted_docs.json"
IMAGE_CAPTION_CACHE = "database/image_caption_cache.json"

VISION_MODEL = "gpt-4o-mini"


def load_cache() -> dict:
    if os.path.exists(IMAGE_CAPTION_CACHE):
        with open(IMAGE_CAPTION_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    os.makedirs(os.path.dirname(IMAGE_CAPTION_CACHE), exist_ok=True)
    with open(IMAGE_CAPTION_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def caption_image(image_bytes: bytes, cache: dict) -> str:
    h = image_hash(image_bytes)
    if h in cache:
        return cache[h]

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = (
        "Đây là một hình ảnh trích từ tài liệu hướng dẫn kỹ thuật viễn thông "
        "tiếng Việt (VNPT). Hãy xác định loại ảnh và transcribe lại đầy đủ nội "
        "dung dưới dạng văn bản để đưa vào cơ sở tri thức tra cứu:\n"
        "- Nếu là SƠ ĐỒ FLOW/FLOWCHART: liệt kê các bước theo đúng thứ tự "
        "mũi tên, đánh số 1,2,3..., mô tả rõ các điều kiện rẽ nhánh (ví dụ: "
        "'Nếu port chưa được tạo -> ...').\n"
        "- Nếu là ẢNH CHỤP MÀN HÌNH giao diện phần mềm: liệt kê tên các nút "
        "bấm, trường nhập liệu, menu quan trọng và vị trí/ý nghĩa của chúng.\n"
        "- Nếu là BẢNG DỮ LIỆU (chụp Excel/bảng): transcribe lại chính xác "
        "thành bảng Markdown, giữ đúng tên cột và vài dòng dữ liệu mẫu.\n"
        "- Nếu ảnh không mang thông tin kỹ thuật (logo, trang trí...): trả lời "
        "đúng một câu 'Không có nội dung kỹ thuật.'\n"
        "Chỉ trả về nội dung transcribe, không thêm lời dẫn."
    )

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=800,
        temperature=0,
    )
    caption = response.choices[0].message.content.strip()
    cache[h] = caption
    return caption




def extract_pdf(pdf_path: str, cache: dict) -> str:
    doc = fitz.open(pdf_path)
    page_blocks = []

    for page_index, page in enumerate(doc):
        # 1) Text của trang - PyMuPDF giữ dòng, ít lỗi vỡ dấu hơn pdfplumber
        raw_text = page.get_text("text")
        page_text = unicodedata.normalize("NFC", raw_text).strip()

        # 2) Ảnh nhúng trong trang - flowchart / screenshot / bảng
        image_captions = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Bỏ qua ảnh quá nhỏ (thường là icon/logo trang trí, không mang
            # thông tin kỹ thuật) để tiết kiệm chi phí gọi vision API.
            if len(image_bytes) < 3000:
                continue

            try:
                caption = caption_image(image_bytes, cache)
            except Exception as e:
                caption = f"[Không thể xử lý ảnh trang {page_index + 1}: {e}]"

            if caption and "không có nội dung kỹ thuật" not in caption.lower():
                image_captions.append(f"[Nội dung hình ảnh trang {page_index + 1}]\n{caption}")

        block = page_text
        if image_captions:
            block += "\n\n" + "\n\n".join(image_captions)
        if block.strip():
            page_blocks.append(block.strip())

    doc.close()
    return "\n\n".join(page_blocks)



def main():
    cache = load_cache()
    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))

    if not pdf_files:
        print(f"Không tìm thấy PDF nào trong {SOURCE_DIR}")
        return

    docs_data = []
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"[Extracting] {file_name} ...")
        content = extract_pdf(pdf_path, cache)
        docs_data.append({"file_name": file_name, "content": content})
        save_cache(cache)  # lưu cache dần, tránh mất nếu script bị gián đoạn

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)

    print(f"\n[Done] Đã trích xuất {len(docs_data)} file vào {OUTPUT_JSON}")
    print("Nhắc: xoá thư mục database/faiss_index cũ rồi chạy lại app để build index mới từ dữ liệu này.")


if __name__ == "__main__":
    main()