# Khai Báo Sử Dụng Công Cụ AI (AI Usage Declaration)

Tài liệu này được lập theo quy định tại **Mục 2.3 (Hình thức báo cáo và nộp bài)** và **Mục 2.5 (Rubric đánh giá bài tập lớn A3 - Tiêu chí Liêm chính học thuật và sử dụng AI)** của học phần **Nhập môn Học máy (FIT4013)**.

## 1. Mục đích và phạm vi sử dụng công cụ AI
Trong quá trình thực hiện **Đề tài 13: Dự đoán nhân viên có khả năng nghỉ việc**, nhóm có sử dụng công cụ trợ lý AI (Google Antigravity / LLM) với vai trò hỗ trợ kỹ thuật trong các công việc sau:
- **Gợi ý cấu trúc mã nguồn (Code Scaffolding):** Hỗ trợ viết khung `ColumnTransformer`, `Pipeline` và lớp biến đổi đặc trưng tùy chỉnh (`HRFeatureEngineer` kế thừa từ `BaseEstimator, TransformerMixin`) trong thư viện `scikit-learn` để đảm bảo không xảy ra rò rỉ dữ liệu (*data leakage*).
- **Trực quan hóa dữ liệu (Visualization):** Hỗ trợ tinh chỉnh định dạng biểu đồ `matplotlib` và `seaborn` (bố cục `subplots`, nhãn số liệu, độ phân giải `300 DPI`) để phục vụ chèn vào báo cáo khoa học.
- **Soạn thảo và định dạng LaTeX:** Hỗ trợ định dạng bảng biểu (`booktabs`), sơ đồ quy trình (`tikz`) và kiểm tra lỗi cú pháp biên dịch LaTeX.

## 2. Cam kết kiểm chứng và làm chủ sản phẩm của nhóm
- **Kiểm tra tính đúng đắn:** Toàn bộ mã nguồn trong `notebooks/DeTai13_Employee_Attrition.ipynb` đã được chạy thực nghiệm trực tiếp trên bộ dữ liệu gốc `WA_Fn-UseC_-HR-Employee-Attrition.csv` (`1,470` bản ghi, `35` cột) với `random_state = 42`.
- **Hiểu và giải trình kết quả:** Mọi con số thống kê, bảng kết quả (`Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC-AUC`, `Confusion Matrix`) và các ca phân tích lỗi (`False Negatives`, `False Positives`) trong báo cáo LaTeX đều được trích xuất trung thực từ kết quả chạy thực nghiệm trong Notebook, không bịa đặt số liệu.
- **Trách nhiệm học thuật:** Tất cả thành viên trong nhóm đã đọc hiểu toàn bộ quy trình tiền xử lý (Luồng A và Luồng B), cơ sở toán học của các thuật toán (`K-NN`, `Gaussian Naive Bayes`, `Decision Tree`), ý nghĩa các siêu tham số và chịu hoàn toàn trách nhiệm giải trình khi bảo vệ bài tập lớn.
