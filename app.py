import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="TelcoOps Copilot - VNPT", layout="centered")

st.title("📡 TelcoOps Copilot - Trợ lý AI Vận hành Mạng Vô tuyến")
st.markdown("Hệ thống AI hỗ trợ tra cứu quy trình VHKT (RAG) và giám sát KPI/Cảnh báo trạm tự động.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def send_feedback(log_id: int, rating: str, correction: str = None, msg_index: int = None):
    try:
        payload = {"log_id": log_id, "rating": rating}
        if correction:
            payload["correction"] = correction
        resp = requests.post(f"{API_BASE}/api/feedback", json=payload)
        if resp.status_code == 200:
            if msg_index is not None:
                st.session_state.messages[msg_index]["feedback_sent"] = rating
            st.toast("Đã ghi nhận phản hồi, hệ thống sẽ học từ việc này." if rating == "up"
                      else "Đã ghi nhận bản sửa, hệ thống sẽ cập nhật tri thức.")
        else:
            st.toast(f"Gửi feedback thất bại (status {resp.status_code})")
    except Exception as e:
        st.toast(f"Không thể gửi feedback: {e}")


# --- Hiển thị lại lịch sử hội thoại ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Chỉ hiển thị khu vực feedback cho câu trả lời của assistant có log_id
        if message["role"] == "assistant" and message.get("log_id") is not None:
            feedback_sent = message.get("feedback_sent")

            if feedback_sent == "up":
                st.caption("✅ Bạn đã đánh giá: Đúng — cảm ơn, hệ thống đã học câu trả lời này.")
            elif feedback_sent == "down":
                st.caption("✏️ Bạn đã gửi bản sửa — hệ thống đã cập nhật tri thức.")
            else:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("👍 Đúng", key=f"up_{i}"):
                        send_feedback(message["log_id"], "up", msg_index=i)
                        st.rerun()
                with col2:
                    if st.button("👎 Sai", key=f"down_{i}"):
                        st.session_state[f"show_correction_{i}"] = True

                # Ô nhập bản sửa, chỉ hiện ra sau khi bấm "👎 Sai"
                if st.session_state.get(f"show_correction_{i}"):
                    correction_text = st.text_area(
                        "Nhập câu trả lời đúng để hệ thống học lại:",
                        key=f"correction_input_{i}",
                    )
                    if st.button("Gửi bản sửa", key=f"submit_correction_{i}"):
                        if correction_text.strip():
                            send_feedback(message["log_id"], "down", correction=correction_text.strip(), msg_index=i)
                            st.session_state[f"show_correction_{i}"] = False
                            st.rerun()
                        else:
                            st.warning("Vui lòng nhập nội dung bản sửa trước khi gửi.")


# --- Ô nhập câu hỏi mới ---
if prompt := st.chat_input("Nhập câu hỏi về quy trình VHKT hoặc tra cứu KPI trạm..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI đang tra cứu tài liệu và cơ sở dữ liệu..."):
            try:
                response = requests.post(f"{API_BASE}/api/chat", json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Không có phản hồi.")
                    source = data.get("source", "N/A")
                    log_id = data.get("log_id")

                    full_response = f"{answer}\n\n*📌 Nguồn tham khảo: `{source}`*"
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "log_id": log_id,
                        "feedback_sent": None,
                    })
                    # rerun để nút feedback xuất hiện ngay dưới câu trả lời vừa gửi
                    st.rerun()
                else:
                    st.error(f"Lỗi kết nối Backend (Status: {response.status_code})")
            except Exception as e:
                st.error(f"Không thể kết nối đến API Server: {e}")