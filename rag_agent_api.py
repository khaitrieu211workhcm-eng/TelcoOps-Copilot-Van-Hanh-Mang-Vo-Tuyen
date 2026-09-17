

import os
import json
import sqlite3
import datetime
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain & FAISS Vector Store imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Self-learning router (lightweight, local, retrainable)
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="TelcoOps Copilot Smart Agent", version="3.0")

DB_PATH = "database/telco_ops.db"
JSON_DOCS_PATH = "database/extracted_docs.json"
FAISS_INDEX_DIR = "database/faiss_index"
ROUTER_MODEL_PATH = "database/router_model.joblib"

# Ngưỡng: cần ít nhất bấy nhiêu mẫu feedback đã xác nhận thì mới bắt đầu
# huấn luyện router tự học; và cứ mỗi bấy nhiêu mẫu mới thì tự retrain lại.
# Hạ thấp so với bản trước (20/5) để router bắt đầu học sớm hơn khi đang
# trong giai đoạn test/mới triển khai, ít dữ liệu.
MIN_TRAINING_SAMPLES = 8
RETRAIN_EVERY_N_NEW_SAMPLES = 3

# Ngưỡng khoảng cách FAISS (càng nhỏ càng giống) để coi là "đã có câu trả
# lời được xác nhận từ trước" và dùng lại luôn, khỏi hỏi LLM.
VERIFIED_CACHE_DISTANCE_THRESHOLD = 0.25

# Số ứng viên lấy ra khi tìm trong kho tri thức đã học (verified_answer /
# expert_correction). Bản cũ dùng k=1 nên nếu tài liệu gốc tình cờ gần câu
# hỏi hơn bản đã-sửa, hệ thống bỏ sót hẳn bản đã-sửa. Giờ lấy rộng hơn rồi
# lọc theo loại, để bản đã-sửa không bị che khuất.
LEARNED_SEARCH_K = 5

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def ensure_db():
    """Tạo các bảng cần thiết cho logging + self-learning nếu chưa có."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            decision TEXT,
            answer TEXT,
            source TEXT,
            feedback TEXT,
            correction TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS routing_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            correct_decision TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_vector_store():
    """Khởi tạo hoặc load Vector DB bằng FAISS."""
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    if not os.path.exists(JSON_DOCS_PATH):
        raise FileNotFoundError("Không tìm thấy file extracted_docs.json. Hãy chạy extract_docs.py trước!")

    with open(JSON_DOCS_PATH, "r", encoding="utf-8") as f:
        docs_data = json.load(f)

    texts = []
    metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

    for item in docs_data:
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({"source": item["file_name"], "type": "static_doc"})

    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"[Vector DB] Đã tạo thành công FAISS Index với {len(texts)} chunks tri thức!")
    return vector_store


ensure_db()
vector_store = get_vector_store()

# Router tự học: model + vectorizer được load nếu đã từng huấn luyện trước đó
router_model: Optional[LogisticRegression] = None
router_vectorizer: Optional[TfidfVectorizer] = None
if os.path.exists(ROUTER_MODEL_PATH):
    bundle = joblib.load(ROUTER_MODEL_PATH)
    router_model = bundle["model"]
    router_vectorizer = bundle["vectorizer"]
    print("[Self-Learning Router] Đã load classifier đã huấn luyện từ trước.")


class ChatRequest(BaseModel):
    query: str


class FeedbackRequest(BaseModel):
    log_id: int                    # id của dòng trong conversation_log cần đánh giá
    rating: str                    # "up" hoặc "down"
    correction: Optional[str] = None   # bản trả lời đúng, bắt buộc nếu rating == "down"
    correct_decision: Optional[str] = None  # "SQL" hoặc "RAG" nếu muốn dạy lại router


class IngestRequest(BaseModel):
    file_name: str
    content: str


def query_sql_database(sql_query: str):
    """Thực thi câu lệnh SQL xuống database KPI trạm."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        return {"columns": columns, "data": rows}
    except Exception as e:
        return {"error": str(e)}


SQL_SCHEMA_DESCRIPTION = """
Bảng cell_kpi (chỉ số kỹ thuật trạm, đo theo giờ):
  - timestamp TEXT (định dạng 'YYYY-MM-DD HH:MM:SS')
  - cell_id TEXT
  - site_name TEXT
  - rsrp_dbm REAL        (mức thu tín hiệu, càng gần 0 càng tốt, thường âm)
  - sinr_db REAL         (tỉ số tín hiệu/nhiễu, càng cao càng tốt)
  - prb_utilization_pct REAL   (% sử dụng tài nguyên khối truyền dẫn, cao = nghẽn)
  - throughput_dl_mbps REAL
  - call_drop_rate_pct REAL    (tỉ lệ rớt cuộc gọi, %)

Bảng alarm_logs (lịch sử cảnh báo thiết bị/trạm):
  - timestamp TEXT
  - cell_id TEXT
  - site_name TEXT
  - severity TEXT   (giá trị: 'CRITICAL', 'MAJOR', 'MINOR')
  - alarm_description TEXT
  - status TEXT     (giá trị: 'ACTIVE', 'RESOLVED')
"""

# Các từ khoá tuyệt đối không được xuất hiện trong SQL sinh ra -> chỉ cho
# phép truy vấn đọc (SELECT), chặn mọi thay đổi/xoá dữ liệu.
SQL_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "pragma", "create", "replace", "truncate", "exec", "--", ";--",
)


def is_sql_safe(sql_query: str) -> bool:
    q = sql_query.strip().lower()
    if not q.startswith("select"):
        return False
    if any(bad in q for bad in SQL_FORBIDDEN_KEYWORDS):
        return False
    return True


def generate_sql_query(user_query: str, previous_error: Optional[str] = None) -> str:
    """Dùng LLM sinh câu SELECT phù hợp với đúng câu hỏi của người dùng,
    dựa trên schema thật của database (thay vì 1 câu cứng như bản cũ)."""
    error_hint = (
        f"\nLưu ý: câu SQL trước đó bị lỗi: '{previous_error}'. Hãy sửa lại cho đúng."
        if previous_error else ""
    )
    prompt = f"""
    Bạn là chuyên gia SQLite. Dựa vào schema sau:
    {SQL_SCHEMA_DESCRIPTION}

    Hãy viết ĐÚNG MỘT câu lệnh SELECT (SQLite hợp lệ) để trả lời câu hỏi:
    "{user_query}"

    Quy tắc bắt buộc:
    - Chỉ dùng SELECT, không được INSERT/UPDATE/DELETE/DROP...
    - Nếu câu hỏi liên quan cảnh báo/alarm -> dùng bảng alarm_logs.
    - Nếu câu hỏi liên quan chỉ số kỹ thuật/KPI -> dùng bảng cell_kpi.
    - Nếu không lọc theo thời gian cụ thể, thêm ORDER BY timestamp DESC.
    - Luôn thêm LIMIT (mặc định LIMIT 20 nếu người dùng không nói rõ số lượng).
    - Chỉ trả về DUY NHẤT câu SQL, không giải thích, không markdown, không dấu ```.
    {error_hint}
    """
    sql_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    # phòng trường hợp LLM vẫn bọc trong ```sql ... ```
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    return sql_query


def run_text_to_sql(user_query: str):
    """Sinh SQL từ câu hỏi, kiểm tra an toàn, thực thi; nếu lỗi thì cho LLM
    một cơ hội tự sửa lại (feed lỗi ngược vào prompt) trước khi bỏ cuộc."""
    sql_query = generate_sql_query(user_query)
    if not is_sql_safe(sql_query):
        return {"sql": sql_query, "result": {"error": "Câu SQL sinh ra không an toàn hoặc không phải SELECT."}}

    result = query_sql_database(sql_query)
    if "error" in result:
        # Cho LLM 1 lần sửa lại dựa trên thông báo lỗi thật từ SQLite
        sql_query_retry = generate_sql_query(user_query, previous_error=result["error"])
        if is_sql_safe(sql_query_retry):
            result = query_sql_database(sql_query_retry)
            sql_query = sql_query_retry

    return {"sql": sql_query, "result": result}

def log_conversation(query: str, decision: str, answer: str, source: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO conversation_log (timestamp, query, decision, answer, source, feedback, correction) "
        "VALUES (?, ?, ?, ?, ?, NULL, NULL)",
        (datetime.datetime.utcnow().isoformat(), query, decision, answer, source),
    )
    conn.commit()
    log_id = cur.lastrowid
    conn.close()
    return log_id


def update_feedback(log_id: int, rating: str, correction: Optional[str]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE conversation_log SET feedback = ?, correction = ? WHERE id = ?",
        (rating, correction, log_id),
    )
    conn.commit()
    conn.close()


def add_learned_knowledge(text: str, source_tag: str):
    """Thêm tri thức mới vào FAISS mà KHÔNG build lại toàn bộ index (incremental)."""
    vector_store.add_texts(texts=[text], metadatas=[{"source": source_tag, "type": "learned"}])
    vector_store.save_local(FAISS_INDEX_DIR)


def check_verified_cache(user_query: str):
    """Tìm xem có tri thức đã-được-xác-nhận (verified_answer/expert_correction)
    nào khớp gần đúng câu hỏi này chưa.

    Bản cũ chỉ lấy k=1 trong TOÀN BỘ index (cả tài liệu gốc lẫn tri thức đã
    học) -> nếu 1 tài liệu gốc tình cờ gần câu hỏi hơn, bản đã-sửa bị bỏ sót
    hoàn toàn dù đã tồn tại. Giờ lấy k rộng hơn rồi lọc riêng theo loại
    "learned", và ưu tiên expert_correction (bản sửa của kỹ sư) hơn
    verified_answer (chỉ là câu trả lời được thumbs-up) nếu cả hai cùng khớp.
    """
    results = vector_store.similarity_search_with_score(user_query, k=LEARNED_SEARCH_K)
    learned_hits = [
        (doc, distance) for doc, distance in results
        if doc.metadata.get("type") == "learned" and distance <= VERIFIED_CACHE_DISTANCE_THRESHOLD
    ]
    if not learned_hits:
        return None

    # Ưu tiên expert_correction trước, trong nhóm đó lấy khoảng cách nhỏ nhất
    def sort_key(item):
        doc, distance = item
        is_correction = doc.metadata.get("source") == "expert_correction"
        return (0 if is_correction else 1, distance)

    best_doc, _ = sorted(learned_hits, key=sort_key)[0]
    return best_doc.page_content


def get_training_data_count() -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM routing_training_data")
    count = cur.fetchone()[0]
    conn.close()
    return count


def add_routing_training_sample(query: str, correct_decision: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO routing_training_data (query, correct_decision, timestamp) VALUES (?, ?, ?)",
        (query, correct_decision, datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def train_router_if_ready():
    """Tự động (re)huấn luyện router mỗi khi có đủ dữ liệu feedback mới.
    Đây chính là phần biến hệ thống từ 'gọi LLM mỗi lần' thành 'tự học và
    ngày càng ít phụ thuộc LLM hơn'."""
    global router_model, router_vectorizer

    total = get_training_data_count()
    if total < MIN_TRAINING_SAMPLES:
        return  # chưa đủ dữ liệu để học
    if total % RETRAIN_EVERY_N_NEW_SAMPLES != 0:
        return  # chưa tới mốc retrain

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT query, correct_decision FROM routing_training_data")
    rows = cur.fetchall()
    conn.close()

    queries = [r[0] for r in rows]
    labels = [r[1] for r in rows]

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(queries)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    router_model = model
    router_vectorizer = vectorizer
    joblib.dump({"model": model, "vectorizer": vectorizer}, ROUTER_MODEL_PATH)
    print(f"[Self-Learning Router] Đã tự động huấn luyện lại với {total} mẫu dữ liệu.")


def infer_correct_routing_from_correction(original_query: str, correction: str) -> str:
    """Khi kỹ sư đánh giá 'Sai' và gửi bản sửa nhưng KHÔNG chỉ rõ nguồn dữ
    liệu đúng là SQL hay RAG, dùng LLM suy ra từ chính nội dung bản sửa.

    Đây là mảnh còn thiếu ở bản trước: router chỉ từng học được từ feedback
    'Đúng' (tự củng cố điều nó đã đoán đúng), KHÔNG BAO GIỜ học được từ
    feedback 'Sai' -> router lặp lại đúng những lỗi định tuyến cũ mãi mãi
    dù bạn đã sửa nội dung câu trả lời. Hàm này đóng vòng lặp đó lại.
    """
    prompt = f"""
    Câu hỏi gốc: "{original_query}"
    Câu trả lời ĐÚNG theo kỹ sư đã sửa: "{correction}"

    Để trả lời đúng câu hỏi này, hệ thống LẼ RA nên lấy dữ liệu từ đâu?
    - 'SQL' nếu câu trả lời dựa trên số liệu/KPI/cảnh báo cụ thể từ database.
    - 'RAG' nếu câu trả lời dựa trên quy trình/tài liệu kỹ thuật.
    Chỉ trả về đúng một từ: 'SQL' hoặc 'RAG'.
    """
    try:
        result = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    except Exception:
        return "RAG"
    return "SQL" if "SQL" in result else "RAG"


def route_query(user_query: str) -> str:
    """Định tuyến SQL/RAG. Ưu tiên dùng classifier tự học (nhanh, miễn phí,
    ngày càng chính xác); chỉ rơi về gọi LLM khi classifier chưa sẵn sàng
    hoặc chưa đủ tự tin về dự đoán của mình."""
    if router_model is not None and router_vectorizer is not None:
        X = router_vectorizer.transform([user_query])
        proba = router_model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        prediction = router_model.classes_[int(np.argmax(proba))]
        if confidence >= 0.75:
            return prediction  # classifier tự tin -> dùng luôn, không cần LLM

    # Fallback: dùng LLM làm "giáo viên" khi model tự học chưa đủ tự tin/chưa tồn tại
    router_prompt = f"""
    Bạn là hệ thống điều phối thông minh của trợ lý TelcoOps Copilot nhà mạng VNPT.
    Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và quyết định xem nên chọn nguồn dữ liệu nào:
    - Trả về 'SQL' nếu câu hỏi yêu cầu thống kê số liệu, xem chỉ số KPI trạm, chất lượng mạng, rớt cuộc gọi, công suất, cảnh báo lỗi cụ thể từ cơ sở dữ liệu.
    - Trả về 'RAG' nếu câu hỏi yêu cầu tra cứu quy trình bảo dưỡng, tài liệu kỹ thuật, quy định an toàn, thiết bị (như accu, điều hòa, viba, tủ nguồn,...).

    Câu hỏi người dùng: "{user_query}"
    Chỉ trả về đúng một từ duy nhất: 'SQL' hoặc 'RAG'. Không giải thích thêm.
    """
    try:
        decision = llm.invoke(
            [SystemMessage(content="You are a routing system."), HumanMessage(content=router_prompt)]
        ).content.strip().upper()
    except Exception:
        decision = "RAG"
    return "SQL" if "SQL" in decision else "RAG"



@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    user_query = request.query

    # BƯỚC 0: Kiểm tra cache tri thức đã-được-xác-nhận trước tiên.
    cached_answer = check_verified_cache(user_query)
    if cached_answer:
        log_id = log_conversation(user_query, "CACHE", cached_answer, "Verified Knowledge Cache")
        return {
            "type": "verified_cache",
            "answer": cached_answer,
            "source": "Tri thức đã được xác nhận (học từ feedback trước đó)",
            "log_id": log_id,
        }

    # BƯỚC 1: Định tuyến (ưu tiên classifier tự học, fallback LLM)
    decision = route_query(user_query)

    # BƯỚC 2: Xử lý theo nhánh quyết định
    if decision == "SQL":
        # Text-to-SQL thật: SQL được sinh động theo đúng câu hỏi, có thể
        # truy vấn cả cell_kpi lẫn alarm_logs, thay vì luôn LIMIT 5 cố định.
        sql_info = run_text_to_sql(user_query)
        sql_result = sql_info["result"]

        if "error" in sql_result:
            answer = (
                "Không thể truy vấn được dữ liệu cho câu hỏi này "
                f"(lỗi: {sql_result['error']}). Bạn có thể diễn đạt lại câu hỏi rõ hơn không?"
            )
        else:
            synthesize_prompt = f"""
            Dựa trên dữ liệu giám sát/cảnh báo trạm viễn thông sau đây từ database
            (câu SQL đã dùng: {sql_info['sql']}):
            {sql_result}

            Hãy viết một câu trả lời báo cáo ngắn gọn, chuyên nghiệp bằng tiếng Việt để trả lời câu hỏi: "{user_query}"
            """
            answer = llm.invoke([HumanMessage(content=synthesize_prompt)]).content

        log_id = log_conversation(user_query, "SQL", answer, "SQLite Database (telco_ops.db)")
        return {"type": "sql_agent", "answer": answer, "source": "SQLite Database (telco_ops.db)", "log_id": log_id}

    else:
        docs = vector_store.similarity_search(user_query, k=3)
        if docs:
            # Tách riêng tài liệu đã được kỹ sư xác nhận/sửa ra khỏi tài liệu
            # gốc, và luôn đặt lên đầu context + nhắc LLM ưu tiên tuyệt đối
            # nếu có mâu thuẫn -> tránh việc tài liệu gốc (có thể đã lỗi thời
            # hoặc sai) đè lên bản đã được sửa.
            learned_docs = [d for d in docs if d.metadata.get("type") == "learned"]
            static_docs = [d for d in docs if d.metadata.get("type") != "learned"]

            context_parts = []
            if learned_docs:
                context_parts.append(
                    "=== NỘI DUNG ĐÃ ĐƯỢC KỸ SƯ XÁC NHẬN/SỬA (ƯU TIÊN TUYỆT ĐỐI) ===\n"
                    + "\n---\n".join(d.page_content for d in learned_docs)
                )
            if static_docs:
                context_parts.append(
                    "=== TÀI LIỆU KỸ THUẬT GỐC ===\n"
                    + "\n---\n".join(d.page_content for d in static_docs)
                )
            context = "\n\n".join(context_parts)
            source_file = docs[0].metadata.get("source", "Tài liệu kỹ thuật VNPT")

            rag_prompt = f"""
            Bạn là trợ lý kỹ thuật VNPT. Hãy dựa vào các đoạn tài liệu sau để trả lời câu hỏi của kỹ sư một cách chính xác, rõ ràng.
            QUAN TRỌNG: nếu nội dung ở phần "ĐÃ ĐƯỢC KỸ SƯ XÁC NHẬN/SỬA" mâu thuẫn với "TÀI LIỆU KỸ THUẬT GỐC", LUÔN LUÔN ưu tiên và tin theo phần đã được xác nhận/sửa.

            Tài liệu tham khảo:
            {context}

            Câu hỏi: "{user_query}"
            """
            answer = llm.invoke([HumanMessage(content=rag_prompt)]).content
            log_id = log_conversation(user_query, "RAG", answer, source_file)
            return {"type": "rag", "answer": answer, "source": source_file, "log_id": log_id}
        else:
            answer = "Không tìm thấy thông tin phù hợp trong kho tài liệu VHKT."
            log_id = log_conversation(user_query, "RAG", answer, "None")
            return {"type": "rag", "answer": answer, "source": "None", "log_id": log_id}


@app.post("/api/feedback")
def feedback_endpoint(request: FeedbackRequest):
    """Đây là 'trái tim' của cơ chế tự học: kỹ sư xác nhận câu trả lời
    đúng/sai, hệ thống dùng chính phản hồi đó để tự cải thiện."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT query, answer, decision FROM conversation_log WHERE id = ?", (request.log_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Không tìm thấy log_id này.")

    original_query, original_answer, original_decision = row
    update_feedback(request.log_id, request.rating, request.correction)

    if request.rating == "up":
        # Câu trả lời đúng -> học luôn thành tri thức đã xác nhận
        learned_text = f"Câu hỏi: {original_query}\nCâu trả lời đã xác nhận: {original_answer}"
        add_learned_knowledge(learned_text, "verified_answer")

    elif request.rating == "down":
        if not request.correction:
            raise HTTPException(status_code=400, detail="Cần cung cấp 'correction' khi đánh giá là sai.")
        corrected_text = f"Câu hỏi: {original_query}\nCâu trả lời đúng (đã được kỹ sư sửa): {request.correction}"
        add_learned_knowledge(corrected_text, "expert_correction")

    else:
        raise HTTPException(status_code=400, detail="'rating' phải là 'up' hoặc 'down'.")

    # Xác định nguồn dữ liệu ĐÚNG để dạy lại cho router tự học.
    # - "up": router đã chọn đúng -> dùng luôn original_decision.
    # - "down": nếu người dùng không chỉ rõ correct_decision, suy ra từ nội
    #   dung correction bằng LLM, thay vì bỏ qua hoàn toàn như bản trước
    #   (bản trước: router KHÔNG BAO GIỜ học được từ feedback sai).
    if request.correct_decision in ("SQL", "RAG"):
        correct_decision = request.correct_decision
    elif request.rating == "up":
        correct_decision = original_decision
    elif request.rating == "down" and request.correction:
        correct_decision = infer_correct_routing_from_correction(original_query, request.correction)
    else:
        correct_decision = None

    if correct_decision in ("SQL", "RAG"):
        add_routing_training_sample(original_query, correct_decision)
        train_router_if_ready()

    return {"status": "ok", "message": "Đã ghi nhận feedback và cập nhật tri thức hệ thống."}


@app.post("/api/ingest")
def ingest_endpoint(request: IngestRequest):
    """Nạp thêm tài liệu mới vào kho tri thức mà KHÔNG cần build lại toàn
    bộ FAISS index từ đầu (incremental ingestion)."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    chunks = text_splitter.split_text(request.content)
    vector_store.add_texts(
        texts=chunks,
        metadatas=[{"source": request.file_name, "type": "static_doc"} for _ in chunks],
    )
    vector_store.save_local(FAISS_INDEX_DIR)
    return {"status": "ok", "chunks_added": len(chunks)}


@app.get("/api/stats")
def stats_endpoint():
    """Xem hệ thống đã 'học' được bao nhiêu, để dễ theo dõi tiến độ."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM conversation_log")
    total_conversations = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM conversation_log WHERE feedback = 'up'")
    positive_feedback = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM conversation_log WHERE feedback = 'down'")
    negative_feedback = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM routing_training_data")
    routing_samples = cur.fetchone()[0]
    conn.close()

    return {
        "total_conversations": total_conversations,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "routing_training_samples": routing_samples,
        "router_is_self_learned": router_model is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)