# Bài Tập Lớn Học Phần Nhập Môn Học Máy (FIT4013)
## Đề tài 13: Dự đoán nhân viên có khả năng nghỉ việc (Employee Attrition Prediction)

- **Nhóm thực hiện:** **Nhóm 7**
- **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
- **Danh sách thành viên & Phân công nhiệm vụ (Nhóm 7):**

| STT | Họ và tên | Mã sinh viên (MSSV) | Nhiệm vụ phụ trách chính | Tỷ lệ đóng góp |
| :-: | :--- | :---: | :--- | :-: |
| 1 | **Nguyễn Văn An** | `1871020003` | Đặt vấn đề, khám phá dữ liệu (EDA), thống kê mô tả, kiểm tra ngoại lệ IQR, phân tích tương quan và Chương 1 báo cáo LaTeX | 25% |
| 2 | **Nguyễn Việt Anh** | `1871020047` | Thiết kế Pipeline chống rò rỉ dữ liệu, xây dựng Luồng A & Luồng B (`HRFeatureEngineer`, `src/preprocessing.py`), Chương 2 & 3 báo cáo | 25% |
| 3 | **Đào Thị Tươi** | `1871020639` | Huấn luyện Baseline, tinh chỉnh siêu tham số (`GridSearchCV` 5-Fold) cho K-NN, Gaussian NB, Decision Tree (`src/models.py`), Chương 4 báo cáo | 25% |
| 4 | **Lê Đình Huy** | `1871020307` | Đánh giá mô hình, phân tích lỗi (*Error Analysis*), thực nghiệm mở rộng (Random Forest, PCA, Threshold Tuning), xây dựng Web Demo `app.py`, Chương 5 & 6 | 25% |

Dự án xây dựng quy trình Học máy hoàn chỉnh (từ Khám phá dữ liệu - EDA, Tiền xử lý đa luồng, Kỹ thuật đặc trưng, Huấn luyện & Tinh chỉnh siêu tham số đến Phân tích lỗi) cho bài toán dự đoán nhân viên có khả năng nghỉ việc trên bộ dữ liệu **IBM HR Analytics Employee Attrition & Performance**.

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
BTLLL/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Bộ dữ liệu gốc (1,470 bản ghi x 35 cột)
├── src/                                        # Mã nguồn đóng gói dạng module chuẩn hóa
│   ├── __init__.py
│   ├── preprocessing.py                        # Luồng A, Luồng B và lớp HRFeatureEngineer
│   ├── models.py                               # Không gian tham số & cấu hình GridSearchCV
│   └── evaluation.py                           # Các hàm tính toán metric & đánh giá mô hình
├── notebooks/
│   └── DeTai13_Employee_Attrition.ipynb        # Bản code Jupyter Notebook hoàn chỉnh (đã nhúng sẵn bảng & 12 biểu đồ)
├── report/
│   ├── main.tex                                # Bản báo cáo khoa học LaTeX hoàn chỉnh (6 chương chuẩn quy định)
│   ├── slides.tex                              # Bản Slide thuyết trình bằng LaTeX Beamer phục vụ bảo vệ
│   ├── references.bib                          # Danh mục tài liệu tham khảo BibTeX
│   ├── BaoCao_DeTai13_Overleaf.zip             # Gói nén sẵn sàng tải lên Overleaf để biên dịch 1-click
│   ├── TaiLieu_OnTap_BaoVe_VanDap.md           # Bộ câu hỏi & đáp án ôn tập bảo vệ vấn đáp cho các thành viên
│   └── figures/                                # 12 biểu đồ chất lượng cao (300 DPI) phục vụ báo cáo LaTeX
├── main.py                                     # Script chạy tái lập toàn bộ pipeline bằng 1 lệnh duy nhất
├── app.py                                      # Ứng dụng Web Demo (Streamlit) dự đoán nguy cơ nghỉ việc (Điểm thưởng)
├── requirements.txt                            # Khai báo các thư viện Python cần thiết
├── AI_USAGE.md                                 # Khai báo minh bạch việc sử dụng công cụ AI
└── README.md                                   # Hướng dẫn cài đặt và tái lập kết quả
```

---

## 2. Nguồn Dữ Liệu

- **Tên bộ dữ liệu:** IBM HR Analytics Employee Attrition & Performance (`WA_Fn-UseC_-HR-Employee-Attrition.csv`).
- **Quy mô:** `1,470` mẫu nhân viên $\times$ `35` cột thuộc tính (`26` cột số, `9` cột phân loại).
- **Biến mục tiêu (`Attrition`):** `No` (`1,233` mẫu - `83.88%`) và `Yes` (`237` mẫu - `16.12%`).
- **Đường dẫn tệp dữ liệu:** Được giải nén từ `archive.zip` vào `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

---

## 3. Hướng Dẫn Cài Đặt và Chạy Lại Chương Trình (Khả Năng Tái Lập)

### Bước 1: Cài đặt thư viện
Yêu cầu Python `>= 3.10`. Chạy lệnh sau tại thư mục gốc của dự án:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy Jupyter Notebook
Mở và chạy toàn bộ các ô lệnh trong `notebooks/DeTai13_Employee_Attrition.ipynb`:
```bash
jupyter notebook notebooks/DeTai13_Employee_Attrition.ipynb
```
> **Lưu ý về khả năng tái tạo:** Toàn bộ quá trình chia dữ liệu (`80% Train / 20% Test`, `stratify=y`), kiểm định chéo `StratifiedKFold(n_splits=5)` và khởi tạo mô hình đều được cố định với `RANDOM_STATE = 42`.

### Bước 3: Chạy Ứng dụng Demo Giao diện (Điểm thưởng Mục 2.4.1)
```bash
streamlit run app.py
```

### Bước 4: Biên dịch Báo cáo LaTeX (`report/main.tex`)
Có thể tải toàn bộ thư mục `report/` (gồm `main.tex`, `references.bib` và thư mục `figures/`) lên **Overleaf** hoặc biên dịch trực tiếp trên máy cá nhân bằng `pdflatex` + `bibtex`:
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 4. Tóm Tắt Kết Quả Thực Nghiệm Chính (Tập Test = 294 mẫu)

| Mô hình | Luồng dữ liệu | Giai đoạn | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | Luồng A | Baseline | 0.8401 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Dummy (Stratified)** | Luồng A | Baseline | 0.7143 | 0.1224 | 0.1277 | 0.1250 | 0.4768 |
| **K-NN** | Luồng A | Default | 0.8367 | 0.4545 | 0.1064 | 0.1724 | 0.6181 |
| **Gaussian Naive Bayes** | Luồng A | Default | 0.6156 | 0.2500 | 0.7021 | 0.3687 | 0.6956 |
| **Decision Tree** | Luồng A | Default | 0.7585 | 0.2931 | 0.3617 | 0.3238 | 0.5979 |
| **K-NN** | Luồng A | Tuned | 0.8469 | 0.5625 | 0.1915 | 0.2857 | 0.6677 |
| **Gaussian Naive Bayes** | Luồng A | Tuned | 0.7891 | 0.3729 | 0.4681 | 0.4151 | 0.7258 |
| **Decision Tree** | Luồng A | Tuned | 0.7449 | 0.3409 | 0.6383 | 0.4444 | 0.7026 |
| **K-NN** | **Luồng B** | **Tuned** | 0.8333 | 0.4615 | 0.2553 | **0.3288** | 0.6562 |
| **Gaussian Naive Bayes** | **Luồng B** | **Tuned** | 0.8027 | 0.4154 | 0.5745 | **0.4821** | **0.7507** |
| **Decision Tree** | **Luồng B** | **Tuned** | 0.7279 | 0.3299 | **0.6809** | **0.4444** | 0.6972 |
| **Random Forest (Mở rộng)** | Luồng A | Tuned | 0.7993 | 0.4091 | 0.5745 | 0.4779 | **0.7819** |
| **Random Forest (Mở rộng)** | **Luồng B** | **Tuned** | **0.8129** | **0.4394** | **0.6170** | **0.5133** | 0.7785 |
