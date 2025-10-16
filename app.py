import streamlit as st
import joblib
import numpy as np
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ===================================
# 1️⃣ Tạo dataset mẫu (hoặc dùng file của bạn)
# ===================================
data = {
    'Hours_Study': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Previous Scores': [40, 50, 55, 60, 65, 70, 80, 85, 90, 95],
    'Extracurricular Activities': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    'Sleep Hours': [4, 5, 6, 7, 6, 8, 7, 8, 9, 8],
    'Sample Question Papers Practiced': [2, 3, 4, 5, 6, 7, 8, 9, 10, 10],
    'Grade': [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # 0=TB, 1=Khá, 2=Giỏi
}
df = pd.DataFrame(data)

# ===================================
# 2️⃣ Tách dữ liệu
# ===================================
X = df[['Hours_Study', 'Previous Scores', 'Extracurricular Activities',
        'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Grade']

# Chia tập huấn luyện / kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===================================
# 3️⃣ Chuẩn hóa dữ liệu
# ===================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================
# 4️⃣ Huấn luyện mô hình Logistic Regression
# ===================================
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ===================================
# 5️⃣ Lưu lại model và scaler
# ===================================
joblib.dump(model, "student_model.pkl")
joblib.dump(scaler, "student_scaler.pkl")

print("✅ Mô hình và scaler đã được lưu thành công!")

# -----------------------------
# 1️⃣ Tải mô hình và scaler
# -----------------------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("student_scaler.pkl")

st.title("🎓 Ứng dụng dự đoán học lực học sinh")

st.markdown("### ✏️ Nhập thông tin học sinh:")

# -----------------------------
# 2️⃣ Nhập 5 đặc trưng
# -----------------------------
hours = st.number_input("Số giờ học trung bình mỗi ngày", min_value=0.0, max_value=24.0, value=2.0)
previous = st.number_input("Điểm trung bình năm trước", min_value=0.0, max_value=100.0, value=70.0)
activity = st.selectbox("Tham gia hoạt động ngoại khóa?", ["Không", "Có"])
sleep = st.number_input("Số giờ ngủ trung bình mỗi ngày", min_value=0.0, max_value=12.0, value=7.0)
papers = st.number_input("Số đề luyện tập đã làm", min_value=0, max_value=50, value=5)

# Chuyển dữ liệu sang dạng số
activity_num = 1 if activity == "Có" else 0

# -----------------------------
# 3️⃣ Tạo input cho mô hình
# -----------------------------
input_data = np.array([[hours, previous, activity_num, sleep, papers]])

# Chuẩn hóa dữ liệu
scaled_data = scaler.transform(input_data)

# -----------------------------
# 4️⃣ Dự đoán
# -----------------------------
prediction = model.predict(scaled_data)

# -----------------------------
# 5️⃣ Hiển thị kết quả
# -----------------------------
if prediction[0] == 2:
    st.success("🎓 Dự đoán: Học sinh có học lực **Giỏi** ⭐")
elif prediction[0] == 1:
    st.info("📘 Dự đoán: Học sinh có học lực **Khá**")
else:
    st.warning("📙 Dự đoán: Học sinh có học lực **Trung bình** hoặc yếu")
