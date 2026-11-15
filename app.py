import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import joblib
import streamlit as st
import pandas as pd
import numpy as np

from src.preprocessing import (
    RANDOM_STATE, HRFeatureEngineer, load_and_split_data,
    build_stream_a_preprocessor, build_stream_b_preprocessor
)
from src.evaluation import compute_hr_business_cost, find_optimal_threshold

st.set_page_config(page_title="Hệ Thống Dự Đoán Nhân Viên Nghỉ Việc - Đề tài 13 (Nhóm 7)", layout="wide")


@st.cache_resource
def load_resources():
    df, X_train, X_test, y_train, y_test = load_and_split_data()
    model_path = 'models/best_pipelines.joblib'
    if os.path.exists(model_path):
        models = joblib.load(model_path)
    else:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        prep_b, _ = build_stream_b_preprocessor(X_train)
        models = {
            'KNN_Luồng B': Pipeline([('prep', prep_b), ('select', SelectKBest(f_classif, k=15)), ('clf', KNeighborsClassifier(n_neighbors=3))]).fit(X_train, y_train),
            'NB_Luồng B': Pipeline([('prep', prep_b), ('clf', GaussianNB(var_smoothing=0.0316227766))]).fit(X_train, y_train),
            'DT_Luồng B': Pipeline([('prep', prep_b), ('clf', DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=15, class_weight='balanced', random_state=42))]).fit(X_train, y_train),
            'RF_Luồng B': Pipeline([('prep', prep_b), ('clf', RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=10, class_weight='balanced', random_state=42))]).fit(X_train, y_train),
        }
    return df, X_test, y_test, models


df_raw, X_test_df, y_test_s, models_dict = load_resources()

st.title("Hệ Thống Dự Đoán Nhân Viên Có Khả Năng Nghỉ Việc (Đề Tài 13 - Nhóm 7)")
st.markdown(
    "**Học phần:** Nhập môn Học máy (FIT4013) | **GVHD:** ThS. Lê Thị Thùy Trang  \n"
    "**Nhóm 7:** Nguyễn Văn An (`1871020003`) · Nguyễn Việt Anh (`1871020047`) · Đào Thị Tươi (`1871020639`) · Lê Đình Huy (`1871020307`)"
)

tab1, tab2, tab3 = st.tabs([
    "1. Dự đoán Hồ sơ Nhân viên Cá nhân",
    "2. Dự đoán Hàng loạt & Xuất Danh sách Cảnh báo (Batch CSV)",
    "3. Phân tích Hiệu quả Kinh tế & Tối ưu Ngưỡng Quyết định HR"
])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Thông tin Cá nhân & Công việc")
        age = st.slider("Tuổi (Age)", 18, 60, 28)
        marital = st.selectbox("Tình trạng hôn nhân (MaritalStatus)", ["Single", "Married", "Divorced"])
        distance = st.slider("Khoảng cách từ nhà đến công ty (km)", 1, 29, 12)
        dept = st.selectbox("Phòng ban (Department)", df_raw['Department'].unique())
        job_role = st.selectbox("Chức danh công việc (JobRole)", df_raw['JobRole'].unique())

    with col2:
        st.subheader("Lương thưởng & Điều kiện làm việc")
        income = st.number_input("Thu nhập hàng tháng - MonthlyIncome ($)", 1000, 20000, 2600, step=100)
        overtime = st.selectbox("Làm thêm giờ (OverTime)", ["Yes", "No"])
        travel = st.selectbox("Tần suất đi công tác (BusinessTravel)", ["Travel_Frequently", "Travel_Rarely", "Non-Travel"])
        stock = st.slider("Mức quyền chọn cổ phiếu (StockOptionLevel)", 0, 3, 0)
        num_comp = st.slider("Số công ty từng làm việc (NumCompaniesWorked)", 0, 9, 2)

    with col3:
        st.subheader("Thâm niên & Chỉ số Hài lòng")
        total_years = st.slider("Tổng số năm kinh nghiệm (TotalWorkingYears)", 0, 40, 3)
        years_comp = st.slider("Số năm tại công ty (YearsAtCompany)", 0, 40, 2)
        years_role = st.slider("Số năm ở vị trí hiện tại (YearsInCurrentRole)", 0, 18, 1)
        years_promo = st.slider("Số năm kể từ lần thăng chức cuối", 0, 15, 2)
        job_sat = st.slider("Hài lòng công việc (JobSatisfaction: 1-4)", 1, 4, 2)
        env_sat = st.slider("Hài lòng môi trường (EnvironmentSatisfaction: 1-4)", 1, 4, 2)
        wlb = st.slider("Cân bằng công việc - cuộc sống (WorkLifeBalance: 1-4)", 1, 4, 2)

    sample = df_raw.drop(columns=['Attrition']).iloc[[0]].copy()
    sample['Age'] = age
    sample['MaritalStatus'] = marital
    sample['DistanceFromHome'] = distance
    sample['Department'] = dept
    sample['JobRole'] = job_role
    sample['MonthlyIncome'] = income
    sample['OverTime'] = overtime
    sample['BusinessTravel'] = travel
    sample['StockOptionLevel'] = stock
    sample['NumCompaniesWorked'] = num_comp
    sample['TotalWorkingYears'] = total_years
    sample['YearsAtCompany'] = years_comp
    sample['YearsInCurrentRole'] = years_role
    sample['YearsSinceLastPromotion'] = years_promo
    sample['JobSatisfaction'] = job_sat
    sample['EnvironmentSatisfaction'] = env_sat
    sample['WorkLifeBalance'] = wlb

    if st.button("Chạy Dự Đoán Trên Cả 4 Mô Hình (Luồng B)", type="primary"):
        st.divider()
        st.subheader("Kết quả Dự đoán từ 4 Mô hình Đã Tinh chỉnh (Luồng B)")
        m_cols = st.columns(4)
        model_labels = [
            ('KNN_Luồng B', 'K-Nearest Neighbors'),
            ('NB_Luồng B', 'Gaussian Naive Bayes'),
            ('DT_Luồng B', 'Decision Tree'),
            ('RF_Luồng B', 'Random Forest (Mở rộng)')
        ]
        for idx, (m_key, m_title) in enumerate(model_labels):
            pipe = models_dict[m_key]
            prob = pipe.predict_proba(sample)[0, 1]
            pred = pipe.predict(sample)[0]
            with m_cols[idx]:
                st.metric(
                    label=m_title,
                    value=f"{prob*100:.1f}%",
                    delta="NGUY CƠ NGHỈ VIỆC" if pred == 1 else "Ổn định (Ở lại)",
                    delta_color="inverse" if pred == 1 else "normal"
                )

        fe_out = HRFeatureEngineer(drop_collinear=False).fit_transform(sample)
        st.subheader("Các Đặc trưng Nghiệp vụ HR (Luồng B) Được Trích xuất Tự động:")
        fe_disp = fe_out[[
            'OverallSatisfaction', 'PromotionStagnationRatio', 'CompanyTenureRatio',
            'IncomePerWorkingYear', 'RoleStabilityRatio', 'OverTime_FrequentTravel', 'YoungSingleLowIncome'
        ]].round(2)
        st.dataframe(fe_disp, use_container_width=True)

        risk_reasons = []
        if overtime == "Yes":
            risk_reasons.append("Làm thêm giờ (`OverTime = Yes`) làm tăng nguy cơ kiệt sức (Burnout).")
        if travel == "Travel_Frequently":
            risk_reasons.append("Đi công tác thường xuyên (`Travel_Frequently`).")
        if income < 3500:
            risk_reasons.append(f"Thu nhập hàng tháng (`${income:,}`) thấp hơn mức trung vị nhóm ở lại (`$5,204`).")
        if stock == 0:
            risk_reasons.append("Không có quyền chọn cổ phiếu gắn kết dài hạn (`StockOptionLevel = 0`).")
        if float(fe_out['OverallSatisfaction'].iloc[0]) <= 2.25:
            risk_reasons.append(f"Điểm hài lòng tổng hợp thấp (`OverallSatisfaction = {float(fe_out['OverallSatisfaction'].iloc[0]):.2f}/4.0`).")

        if risk_reasons:
            st.warning("**Các yếu tố rủi ro phát hiện trên hồ sơ này:**\n- " + "\n- ".join(risk_reasons))
        else:
            st.success("Hồ sơ nhân viên có các chỉ số đãi ngộ và cân bằng công việc tốt, nguy cơ nghỉ việc thấp.")

with tab2:
    st.subheader("Dự đoán Hàng loạt & Lọc Danh sách Nhân viên Nguy cơ Cao")
    uploaded_file = st.file_uploader("Tải lên file CSV hồ sơ nhân viên (hoặc để trống để dùng Tập Kiểm thử chuẩn N = 294):", type=["csv"])
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        model_choice = st.selectbox(
            "Chọn mô hình dự đoán:",
            ["RF_Luồng B", "DT_Luồng B", "NB_Luồng B", "KNN_Luồng B"],
            format_func=lambda x: {
                "RF_Luồng B": "Random Forest Mở rộng (Luồng B - F1 cao nhất 0.519)",
                "DT_Luồng B": "Decision Tree (Luồng B - Recall cao nhất 68.1%)",
                "NB_Luồng B": "Gaussian Naive Bayes (Luồng B - F1 cao nhất nhóm 3 thuật toán chính 0.482)",
                "KNN_Luồng B": "K-Nearest Neighbors (Luồng B)"
            }[x]
        )
    with col_cfg2:
        custom_thresh = st.slider("Ngưỡng xác suất cảnh báo nghỉ việc (Decision Threshold):", 0.20, 0.80, 0.50, 0.05)

    chosen_pipe = models_dict[model_choice]
    if uploaded_file is not None:
        raw_batch = pd.read_csv(uploaded_file)
        has_actual = 'Attrition' in raw_batch.columns
        X_batch = raw_batch.drop(columns=['Attrition']) if has_actual else raw_batch.copy()
        batch_df = raw_batch.copy()
    else:
        X_batch = X_test_df.copy()
        batch_df = X_test_df.copy()
        batch_df['Actual_Attrition'] = y_test_s.map({1: 'Yes', 0: 'No'}).values

    probs = chosen_pipe.predict_proba(X_batch)[:, 1]
    preds = (probs >= custom_thresh).astype(int)
    batch_df['Predicted_Prob'] = np.round(probs, 3)
    batch_df['Predicted_Label'] = pd.Series(preds, index=batch_df.index).map({1: 'Yes (Nguy cơ)', 0: 'No (Ổn định)'})

    high_risk = batch_df[batch_df['Predicted_Prob'] >= custom_thresh].sort_values(by='Predicted_Prob', ascending=False)
    st.write(f"Phát hiện **{len(high_risk)} / {len(batch_df)}** nhân viên có xác suất nghỉ việc $\ge {custom_thresh*100:.0f}\\%$.")
    cols_show = [c for c in ['EmployeeNumber', 'Age', 'Department', 'JobRole', 'MonthlyIncome', 'OverTime', 'BusinessTravel', 'JobSatisfaction', 'Actual_Attrition', 'Predicted_Prob', 'Predicted_Label'] if c in high_risk.columns]
    st.dataframe(high_risk[cols_show], use_container_width=True)

    csv_bytes = high_risk[cols_show].to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="Tải xuống Danh sách Nhân viên Nguy cơ Cao (.CSV)",
        data=csv_bytes,
        file_name="HR_High_Risk_Watchlist.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("Phân tích Hiệu quả Kinh tế (Cost-Sensitive HR ROI) trên Tập Kiểm thử (N = 294)")
    c1, c2, c3 = st.columns(3)
    with c1:
        cost_fn = st.number_input("Chi phí mất 1 nhân viên - False Negative ($)", 5000, 50000, 15000, step=1000)
    with c2:
        cost_fp = st.number_input("Chi phí can thiệp giữ chân 1 nhân viên ($)", 500, 10000, 1500, step=250)
    with c3:
        ret_rate = st.slider("Tỷ lệ giữ chân thành công khi HR can thiệp sớm (%)", 20, 90, 50, step=5) / 100.0

    roi_rows = []
    for m_key, m_name in [
        ('KNN_Luồng B', 'K-NN (Luồng B)'),
        ('NB_Luồng B', 'Gaussian NB (Luồng B)'),
        ('DT_Luồng B', 'Decision Tree (Luồng B)'),
        ('RF_Luồng B', 'Random Forest (Luồng B)')
    ]:
        p_pred = models_dict[m_key].predict(X_test_df)
        p_prob = models_dict[m_key].predict_proba(X_test_df)[:, 1]
        c_def = compute_hr_business_cost(y_test_s, p_pred, cost_fn, cost_fp, ret_rate)
        _, _, best_c = find_optimal_threshold(y_test_s, p_prob, cost_fn, cost_fp, ret_rate)
        roi_rows.append({
            'Mô hình': m_name,
            'TP (Phát hiện đúng)': c_def['tp'],
            'FN (Bỏ sót)': c_def['fn'],
            'FP (Báo nhầm)': c_def['fp'],
            'Chi phí Không Mô hình ($)': f"${c_def['baseline_cost_usd']:,.0f}",
            'Chi phí Dùng Mô hình ($)': f"${c_def['model_cost_usd']:,.0f}",
            'Tiết kiệm Ròng (Ngưỡng 0.50)': f"${c_def['net_savings_usd']:,.0f}",
            'Ngưỡng Tối ưu Chi phí': best_c['threshold'],
            'Tiết kiệm Tối đa ($)': f"${best_c['net_savings_usd']:,.0f}"
        })
    st.dataframe(pd.DataFrame(roi_rows), use_container_width=True)
