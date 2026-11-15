# Tài Liệu Ôn Tập Bảo Vệ Vấn Đáp & Phân Công Nhiệm Vụ Nhóm (Đề Tài 13)

Theo quy định tại **Mục 2** và **Mục 2.5 (Rubric đánh giá bài tập lớn A3)** trong file hướng dẫn của ThS. Lê Thị Thùy Trang:
> *"Tất cả thành viên phải hiểu và giải thích được pipeline, mã nguồn và kết quả của nhóm. Phần nội dung mà thành viên không giải thích được khi bảo vệ có thể không được công nhận trong đánh giá."*

Tài liệu này giúp các thành viên trong nhóm phân chia phần trình bày mạch lạc và nắm vững câu trả lời cho các câu hỏi chuyên môn khi vấn đáp.

---

## 1. Phân Công Nhiệm Vụ Giữa 4 Thành Viên Nhóm 7

| Thành viên (Nhóm 7) | MSSV | Phụ trách chính | Nội dung trình bày & Vấn đáp |
| :--- | :---: | :--- | :--- |
| **1. Nguyễn Văn An** | `1871020003` | **Khám phá dữ liệu (EDA) & Đặt vấn đề** | Chương 1 Báo cáo, Phần 1 Notebook (`fig1`, `fig2`, `fig3`, `fig4`, `fig10`), giải thích ý nghĩa 35 cột, phân tích mất cân bằng lớp (`83.88%` vs `16.12%`), ngoại lệ IQR và đa cộng tuyến. |
| **2. Nguyễn Việt Anh** | `1871020047` | **Tiền xử lý & Kỹ thuật đặc trưng (Luồng A vs Luồng B)** | Chương 2 & Chương 3 Báo cáo, Phần 2 Notebook (`src/preprocessing.py`, lớp `HRFeatureEngineer`, `fig11`), cơ chế chống Data Leakage bằng `Pipeline`/`ColumnTransformer`. |
| **3. Đào Thị Tươi** | `1871020639` | **Huấn luyện mô hình & Tinh chỉnh tham số (`K-NN`, `Naive Bayes`, `Decision Tree`)** | Chương 4 Báo cáo, Phần 3 & 4 Notebook (`src/models.py`, `GridSearchCV`, `fig5`, `fig6`, `fig7`, `fig8`), giải thích công thức toán học và ý nghĩa siêu tham số. |
| **4. Lê Đình Huy** | `1871020307` | **Phân tích lỗi (Error Analysis), Mô hình mở rộng (`Random Forest`, `PCA`) & Demo (`app.py`)** | Chương 5 & Chương 6 Báo cáo, Phần 5 Notebook (`fig9`, `fig12`, `fig13`), giải thích các ca `False Negatives`/`False Positives`, chạy demo ứng dụng Streamlit `app.py`. |

---

## 2. Bộ 10 Câu Hỏi Vấn Đáp Trọng Tâm Giảng Viên Thường Hỏi & Câu Trả Lời Mẫu

### Câu 1: Tại sao trong bài toán này nhóm không đánh giá mô hình dựa trên độ chính xác (Accuracy) mà lại chọn F1-score, Recall và ROC-AUC?
- **Trả lời:** Bộ dữ liệu bị mất cân bằng lớp nghiêm trọng: lớp `No` (Ở lại) chiếm tới **83.88%** (1,233 mẫu) trong khi lớp `Yes` (Nghỉ việc) chỉ chiếm **16.12%** (237 mẫu). Minh chứng thực nghiệm ở mô hình `DummyClassifier(strategy='most_frequent')`: mô hình này đoán tất cả nhân viên đều không nghỉ việc, đạt `Accuracy = 84.01%` nhưng `Recall = 0%` và `F1 = 0.0000` (bỏ lỡ 100% nhân viên nghỉ việc). Do đó, nhóm phải dùng **Recall (lớp Yes)** để đo tỷ lệ phát hiện đúng nhân viên sắp nghỉ, **Precision** để kiểm soát tỷ lệ cảnh báo nhầm, **F1-score** (trung bình điều hòa của Precision và Recall) làm hàm mục tiêu `scoring='f1'` trong `GridSearchCV`, và **ROC-AUC** để đánh giá khả năng phân hạng rủi ro tổng thể.

### Câu 2: Nhóm đã làm thế nào để ngăn chặn hiện tượng rò rỉ dữ liệu (Data Leakage) trong quá trình tiền xử lý và chọn đặc trưng?
- **Trả lời:** Nhóm thực hiện 3 lớp bảo vệ:
  1. Chia tập `Train / Test` (`80% / 20%`, `stratify=y`, `random_state=42`) **ngay từ đầu** trước mọi bước biến đổi.
  2. Loại bỏ cột định danh `EmployeeNumber` và 3 cột hằng số (`EmployeeCount`, `Over18`, `StandardHours`).
  3. Đóng gói toàn bộ quy trình (`HRFeatureEngineer` $\rightarrow$ `SimpleImputer` $\rightarrow$ `StandardScaler`/`RobustScaler` $\rightarrow$ `OneHotEncoder` $\rightarrow$ `SelectKBest(f_classif)` $\rightarrow$ `Classifier`) vào trong một đối tượng `sklearn.pipeline.Pipeline`. Khi chạy `GridSearchCV` 5-fold trên tập `Train`, ở mỗi vòng lặp, các tham số thống kê (như trung bình $\mu$, độ lệch chuẩn $\sigma$, trung vị, IQR, điểm ANOVA F-score) chỉ được `fit` trên 4 fold huấn luyện và `transform` trên 1 fold kiểm định, tuyệt đối không nhìn thấy dữ liệu của fold kiểm định hay tập `Test`.

### Câu 3: Điểm khác biệt cốt lõi giữa Luồng A (Pipeline cơ bản) và Luồng B (Pipeline cải tiến) là gì? Tại sao Luồng B lại cho kết quả tốt hơn?
- **Trả lời:**
  - **Luồng A** chỉ loại bỏ 4 cột ID/hằng số, dùng `OneHotEncoder` cho biến phân loại và `StandardScaler` cho biến số.
  - **Luồng B** khắc phục 4 nhược điểm lớn của dữ liệu thô phát hiện từ EDA:
    1. **Tạo 7 đặc trưng nghiệp vụ HR mới** có sức phân tách cao (ví dụ `OverTime_FrequentTravel` có tỷ lệ nghỉ việc tới **56.1%**, `YoungSingleLowIncome` có tỷ lệ nghỉ việc **43.9%**, `OverallSatisfaction` tổng hợp 4 khía cạnh hài lòng).
    2. **Biến đổi $\log(1+x)$ (`np.log1p`) và dùng `RobustScaler`** cho 5 biến bị lệch phải mạnh và có nhiều ngoại lệ (`MonthlyIncome` có 114 ngoại lệ, `YearsSinceLastPromotion` có 107 ngoại lệ). Điều này giúp kéo phân phối về gần phân phối chuẩn (rất có lợi cho `GaussianNB`) và không bị các giá trị ngoại lệ làm méo thang đo khoảng cách (có lợi cho `K-NN`).
    3. **Mã hóa thứ bậc tự nhiên** cho `BusinessTravel` (`0 -> 1 -> 2`) thay vì tách rời bằng One-Hot.
    4. **Loại bỏ biến đa cộng tuyến cao** (`JobLevel` tương quan $r=0.95$ với `MonthlyIncome`) và 3 biến nhiễu ngẫu nhiên (`DailyRate`, `HourlyRate`, `MonthlyRate`), giúp giảm vi phạm giả định độc lập có điều kiện của Naive Bayes.

### Câu 4: Tại sao thuật toán K-NN lại có Recall và F1-score thấp hơn Gaussian Naive Bayes và Decision Tree trên bộ dữ liệu này?
- **Trả lời:** Có 2 nguyên nhân toán học chính:
  1. **Lời nguyền số chiều (*Curse of Dimensionality*) và dữ liệu hỗn hợp:** Sau khi One-Hot Encoding các biến phân loại, không gian đặc trưng có tới 44–46 chiều (kết hợp cả biến liên tục lẫn biến nhị phân 0/1). Khoảng cách Euclidean/Manhattan trong không gian hơn 40 chiều bị loãng, làm giảm ý nghĩa của các láng giềng gần nhất.
  2. **Mất cân bằng lớp cục bộ:** Do lớp `No` chiếm tới 83.88%, xung quanh một điểm dữ liệu lớp `Yes` trong không gian đặc trưng vẫn có rất nhiều điểm lớp `No`. Khi biểu quyết đa số với $k$ lớn, lớp `No` luôn thắng thế khiến `Recall` của K-NN chỉ đạt `10.64%` (Default Luồng A) và tăng lên `25.53%` (`F1 = 0.3288`) khi dùng `SelectKBest(k=15)` và $k=3$ ở Luồng B.

### Câu 5: Tham số `var_smoothing` trong Gaussian Naive Bayes có ý nghĩa gì và tại sao tinh chỉnh nó lại giúp tăng mạnh F1-score?
- **Trả lời:** Trong `GaussianNB`, xác suất có điều kiện của mỗi đặc trưng được tính theo hàm mật độ phân phối chuẩn với phương sai $\sigma_{jc}^2$. Các đặc trưng nhị phân sinh ra từ `OneHotEncoder` (như các nhánh của `JobRole`, `EducationField`) có phương sai rất nhỏ trên lớp thiểu số (`Yes`), khiến mẫu số $2\sigma_{jc}^2 \approx 0$ và làm hàm hợp lý bị suy biến (sinh ra tới 99 ca `False Positives` ở cấu hình mặc định `var_smoothing = 1e-9`). Khi tăng `var_smoothing` lên `0.0316` (Luồng B), một lượng phương sai ổn định được cộng thêm vào mọi đặc trưng, làm mượt biên quyết định và đẩy `F1-score` lên **0.4821** (`ROC-AUC = 0.7507`).

### Câu 6: Tại sao Cây quyết định (Decision Tree) ở cấu hình mặc định (`Default`) bị quá khớp (Overfitting), và nhóm đã khắc phục như thế nào?
- **Trả lời:** Ở cấu hình mặc định (`max_depth=None`, `min_samples_leaf=1`, `class_weight=None`), Cây quyết định tiếp tục chia nhánh cho đến khi các nút lá hoàn toàn thuần nhất, dẫn đến cây quá sâu, học thuộc lòng nhiễu của tập Train và thiên lệch về lớp đa số (`F1 = 0.3238`, `Recall = 36.17%`). Khi nhóm dùng `GridSearchCV` bật `class_weight='balanced'` (đánh trọng số phạt cao gấp $\approx 5.2$ lần khi phân loại sai mẫu lớp `Yes`), giới hạn độ sâu `max_depth = 4` và ép mỗi nút lá phải có tối thiểu `min_samples_leaf = 15` mẫu (Luồng B), cây tổng quát hóa tốt hơn hẳn: `Recall` tăng vọt từ `38.30%` lên **68.09%** và `F1-score` đạt **0.4444**.

### Câu 7: Hãy giải thích các trường hợp mô hình dự đoán sai (False Negatives và False Positives) trong phần Phân tích lỗi!
- **Trả lời:**
  - **False Negatives (Thực tế nghỉ việc nhưng dự đoán ở lại - 15 ca):** Ví dụ nhân viên `#58` (41 tuổi, lương $\$19,545$, không làm thêm giờ) và `#165` (58 tuổi, lương $\$10,312$, 40 năm làm việc). Họ có thu nhập rất cao và không làm thêm giờ nên mô hình dự đoán xác suất nghỉ việc thấp ($\hat{p} = 0.163$). Tuy nhiên, cả hai đều có `YearsSinceLastPromotion = 15 năm` (15 năm không thăng chức) hoặc nghỉ hưu sớm / chuyển công ty ở cấp quản lý cấp cao.
  - **False Positives (Thực tế ở lại nhưng dự đoán nghỉ việc - 65 ca):** Ví dụ nhân viên `#1495` (24 tuổi, lương thấp $\$2,033$, mới làm 1 năm, `JobSatisfaction = 2`, $\hat{p} = 0.958$). Hồ sơ này mang mọi dấu hiệu điển hình của người sắp nghỉ việc. Trong thực tế quản trị nhân sự, các ca FP này không phải là lỗi vô ích mà chính là **danh sách cảnh báo sớm (*Early Warning Watchlist*)** để phòng HR can thiệp trước khi họ thực sự nộp đơn.
