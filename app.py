import streamlit as st
import joblib
import numpy as np

# -----------------------------
# 1️⃣ Tải mô hình và scaler (đã huấn luyện sẵn)
# -----------------------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("student_scaler.pkl")

# -----------------------------
# 2️⃣ Giao diện Streamlit
# -----------------------------
st.title("🎓 Ứng dụng dự đoán học lực học sinh")
st.markdown("### ✏️ Nhập thông tin học sinh:")

# Nhập dữ liệu
hours = st.number_input("Số giờ học trung bình mỗi ngày", min_value=0.0, max_value=24.0, value=2.0)
previous = st.number_input("Điểm trung bình năm trước", min_value=0.0, max_value=100.0, value=70.0)
activity = st.selectbox("Tham gia hoạt động ngoại khóa?", ["Không", "Có"])
sleep = st.number_input("Số giờ ngủ trung bình mỗi ngày", min_value=0.0, max_value=12.0, value=7.0)
papers = st.number_input("Số đề luyện tập đã làm", min_value=0, max_value=50, value=5)

# Chuyển đổi
activity_num = 1 if activity == "Có" else 0
input_data = np.array([[hours, previous, activity_num, sleep, papers]])
scaled_data = scaler.transform(input_data)

# Dự đoán
prediction = model.predict(scaled_data)

# Hiển thị kết quả
if prediction[0] == 2:
    st.success("🎓 Dự đoán: Học sinh có học lực **Giỏi** ⭐")
elif prediction[0] == 1:
    st.info("📘 Dự đoán: Học sinh có học lực **Khá**")
else:
    st.warning("📙 Dự đoán: Học sinh có học lực **Trung bình** hoặc yếu")

