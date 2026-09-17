**TelcoOps-Copilot-Van-Hanh-Mang-Vo-Tuyen**

**I. TỔNG QUAN**

Một hệ thống chatbot nội bộ giúp kĩ thuật viên vận hành thực hiện: 
- Tra cứu quy trình/tài liệu kỹ thuật .
- Tra cứu số liệu (Text-to-SQL trên SQLite).
- Tự học từ phản hồi của kỹ sư: khi được đánh giá đúng/sai, hệ thống cập nhật lại và huấn luyện lại bộ định tuyến SQL/RAG.

**II. CẤU TRÚC**

- init_database.py	# Tạo telco_ops.db (SQLite) với dữ liệu giả lập
- extract_docs.py	  # Trích text + mô tả ảnh (flowchart, screenshot, bảng) từ PDF vào extracted_docs.json
- rag_agent_api.py	# Backend FastAPI: router SQL/RAG tự học, Text-to-SQL, FAISS vector store, feedback loop
- app.py	          # Giao diện Streamlit
- requirements.txt	# Danh sách thư viện cần cài

**III. HƯỚNG DẪN SỬ DỤNG**

3.1. Ctrl + ~
3.2. Sử dụng lệnh: C:\Users\PC\AppData\Local\Programs\Python\Python39\python.exe -m streamlit run src/app.py

Lưu ý: 
- Sản phẩm được xây dựng để chứng minh ý tưởng, đã chạy được luồng chính (RAG + Text-to-SQL + tự học từ feedback). Vì thế, sản phẩm còn các phần chưa đủ để hoạt động hiệu quả trong các tác vụ thực tế. 

**IV. HẠN CHẾ**

**4.1. An toàn & bảo mật**
- is_sql_safe() chỉ chặn theo từ khoá (SELECT-only, chặn INSERT/DROP). 
Đây là lớp bảo vệ yếu — nên dùng thêm 1 kết nối SQLite chỉ có quyền đọc (read-only) làm lớp chặn thứ hai.
- API (/api/chat, /api/feedback, /api/ingest) không có xác thực (authentication) — bất kỳ ai truy cập được địa chỉ backend đều gọi được, kể cả nạp/sửa tri thức.
- Chưa có giới hạn tốc độ gọi, dễ bị lạm dụng hoặc tốn chi phí LLM/vision API nếu bị spam.
  
**4.2. Vận hành & độ ổn định**
- Chưa xử lý truy cập đồng thời: nhiều request ghi feedback/ingest cùng lúc có thể gây race condition khi ghi FAISS index hoặc SQLite.
- Chưa có test tự động (unit test/integration test) cho bất kỳ thành phần nào.
- Chưa có logging/monitoring cấp production (chỉ print()), không có cảnh báo khi pipeline lỗi.
- app.py dùng st.session_state nên lịch sử chat mất khi refresh trình duyệt — chưa lưu hội thoại persistent theo người dùng.
- Chưa có cơ chế versioning/rollback cho FAISS index và router model.

**4.3. Trích xuất tài liệu**
- Việc mô tả ảnh (flowchart/screenshot/bảng) qua vision LLM tốn chi phí và thời gian, mới cache theo hash ảnh (không cache theo nội dung tương tự); PDF quét (scan) toàn trang chưa được xử lý tối ưu riêng.
- Chưa kiểm định độ chính xác của bước transcribe ảnh — sai sót ở đây sẽ âm thầm lan vào toàn bộ kho tri thức mà không có cảnh báo.

**4.4. Chất lượng tự học**
- Text-to-SQL vẫn có thể sinh sai câu lệnh với câu hỏi phức tạp/mơ hồ; cơ chế tự sửa hiện chỉ thử lại đúng 1 lần.
- FAISS chỉ hỗ trợ thêm tri thức mới, không thể thực sự xoá tri thức cũ sai — bản sửa được ưu tiên trong prompt, nhưng bản sai cũ vẫn tồn tại vĩnh viễn trong index, chiếm dung lượng và có thể gây nhiễu nếu tích luỹ nhiều.
- Router cần một lượng feedback tối thiểu mới bắt đầu học (hiện đặt khá thấp để dễ test) — trong giai đoạn đầu vẫn phụ thuộc hoàn toàn vào LLM để định tuyến, độ chính xác chưa được đánh giá bằng tập kiểm thử độc lập.
- Chưa có cơ chế phát hiện feedback nhiễu/mâu thuẫn).
- 
**V. HƯỚNG PHÁT TRIỂN**
  
5.1.	Thêm xác thực (API key/JWT) cho toàn bộ endpoint.

5.2.	Dùng kết nối SQLite read-only cho nhánh Text-to-SQL.

5.3.	Xây tập câu hỏi kiểm thử (test set) có nhãn đúng để đo độ chính xác router và RAG một cách khách quan, thay vì chỉ dựa cảm tính khi demo.

5.4.	Thêm cơ chế vô hiệu hoá tri thức cũ sai thay vì chỉ chồng thêm bản mới lên trên.

5.5.	Viết test tự động cho các hàm lõi (generate_sql_query, check_verified_cache, route_query).

<img width="1920" height="917" alt="image" src="https://github.com/user-attachments/assets/6f879895-c63d-40b2-a031-56f88ef4c323" />
<img width="778" height="832" alt="image" src="https://github.com/user-attachments/assets/ba80735b-bb7d-41b3-9bd3-13c69bb9ec85" />
<img width="662" height="724" alt="image" src="https://github.com/user-attachments/assets/b2f8c125-adc2-46b9-998a-2fd0a4a2d2d0" />
<img width="653" height="811" alt="image" src="https://github.com/user-attachments/assets/8ddf2c95-a0fd-4a88-8a17-f160f398315f" />
<img width="1142" height="519" alt="image" src="https://github.com/user-attachments/assets/c694debb-c6bd-4bc2-98b9-0f7b75994d6e" />




