**TelcoOps-Copilot-Van-Hanh-Mang-Vo-Tuyen**

**I. TỔNG QUAN**

Một hệ thống chatbot nội bộ giúp kĩ thuật viên vận hành thực hiện: 
- Tra cứu quy trình/tài liệu kỹ thuật .
- Tra cứu số liệu KPI trạm và cảnh báo thiết bị (Text-to-SQL trên SQLite).
- Tự học từ phản hồi của kỹ sư: khi được đánh giá đúng/sai, hệ thống cập nhật lại và huấn luyện lại bộ định tuyến SQL/RAG.

**II. CẤU TRÚC**

- init_database.py	# Tạo telco_ops.db (SQLite) với dữ liệu KPI trạm + cảnh báo giả lập
- extract_docs.py	  # Trích text + mô tả ảnh (flowchart, screenshot, bảng) từ PDF vào extracted_docs.json
- rag_agent_api.py	# Backend FastAPI: router SQL/RAG tự học, Text-to-SQL, FAISS vector store, feedback loop
- app.py	          # Giao diện Streamlit, có nút 👍/👎 để gửi feedback
- requirements.txt	Danh sách thư viện cần cài

Lưu ý: 
- Sản phẩm được xây dựng để chứng minh ý tưởng, đã chạy được luồng chính (RAG + Text-to-SQL + tự học từ feedback). Vì thế, sản phẩm còn các phần chưa đủ để hoạt động hiệu quả trong các tác vụ thực tế. 
- Danh sách những chi tiết cần hoàn thiện nằm cuối file README.md







C:\Users\PC\AppData\Local\Programs\Python\Python39\python.exe -m pip install pandas numpy pypdf python-docx python-pptx pdfplumber

C:\Users\PC\AppData\Local\Programs\Python\Python39\python.exe -m streamlit run src/app.py
