import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="HR Employee Attrition Predictor - Đề tài 13", layout="wide")

DROP_COLS = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_collinear=True):
        self.drop_collinear = drop_collinear
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_out = X.copy()
        for c in DROP_COLS:
            if c in X_out.columns:
                X_out = X_out.drop(columns=[c])
        X_out['PromotionStagnationRatio'] = X_out['YearsSinceLastPromotion'] / (X_out['YearsAtCompany'] + 1.0)
        X_out['CompanyTenureRatio'] = X_out['YearsAtCompany'] / (X_out['TotalWorkingYears'] + 1.0)
        X_out['IncomePerWorkingYear'] = X_out['MonthlyIncome'] / (X_out['TotalWorkingYears'] + 1.0)
        X_out['OverallSatisfaction'] = (
            X_out['EnvironmentSatisfaction'] + X_out['JobSatisfaction'] +
            X_out['RelationshipSatisfaction'] + X_out['WorkLifeBalance']
        ) / 4.0
        X_out['RoleStabilityRatio'] = X_out['YearsInCurrentRole'] / (X_out['YearsAtCompany'] + 1.0)
        X_out['OverTime_FrequentTravel'] = (
            (X_out['OverTime'] == 'Yes') & (X_out['BusinessTravel'] == 'Travel_Frequently')
        ).astype(int)
        X_out['YoungSingleLowIncome'] = (
            (X_out['Age'] < 30) & (X_out['MaritalStatus'] == 'Single') & (X_out['MonthlyIncome'] < 3500)
        ).astype(int)
        travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        X_out['BusinessTravel_Ord'] = X_out['BusinessTravel'].map(travel_map).fillna(1)
        X_out = X_out.drop(columns=['BusinessTravel'])
        for col in ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'DistanceFromHome']:
            if col in X_out.columns:
                X_out[col] = np.log1p(np.maximum(0, X_out[col]))
        if self.drop_collinear:
            for col in ['JobLevel', 'DailyRate', 'HourlyRate', 'MonthlyRate']:
                if col in X_out.columns:
                    X_out = X_out.drop(columns=[col])
        return X_out

@st.cache_resource
def load_and_train_models():
    df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    X = df.drop(columns=['Attrition'])
    y = (df['Attrition'] == 'Yes').astype(int)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    fe_sample = HRFeatureEngineer(drop_collinear=True).fit_transform(X_train)
    cat_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    num_cols = [c for c in fe_sample.columns if c not in cat_cols]

    prep_b = Pipeline([
        ('fe', HRFeatureEngineer(drop_collinear=True)),
        ('ct', ColumnTransformer([
            ('num', RobustScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
        ], remainder='drop'))
    ])

    dt_model = Pipeline([('prep', prep_b), ('clf', DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=15, class_weight='balanced', random_state=42))])
    nb_model = Pipeline([('prep', prep_b), ('clf', GaussianNB(var_smoothing=0.0316227766))])

    dt_model.fit(X_train, y_train)
    nb_model.fit(X_train, y_train)
    return df, dt_model, nb_model

df_raw, dt_pipe, nb_pipe = load_and_train_models()

st.title("Hệ Thống Dự Đoán Nhân Viên Có Khả Năng Nghỉ Việc (Đề Tài 13 - Nhóm 7)")
st.markdown("**Học phần:** Nhập môn Học máy (FIT4013) | **GVHD:** ThS. Lê Thị Thùy Trang | **Thực hiện:** Nhóm 7 (Nguyễn Văn An, Nguyễn Việt Anh, Đào Thị Tươi, Lê Đình Huy)")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("1. Thông tin cá nhân & Công việc")
    age = st.slider("Tuổi (Age)", 18, 60, 29)
    marital = st.selectbox("Tình trạng hôn nhân (MaritalStatus)", ["Single", "Married", "Divorced"])
    distance = st.slider("Khoảng cách từ nhà đến công ty (km)", 1, 30, 10)
    dept = st.selectbox("Phòng ban (Department)", df_raw['Department'].unique())
    job_role = st.selectbox("Vị trí công việc (JobRole)", df_raw['JobRole'].unique())

with col2:
    st.subheader("2. Lương thưởng & Điều kiện làm việc")
    income = st.number_input("Thu nhập hàng tháng - MonthlyIncome ($)", 1000, 20000, 2800, step=100)
    overtime = st.selectbox("Làm thêm giờ (OverTime)", ["Yes", "No"])
    travel = st.selectbox("Tần suất đi công tác (BusinessTravel)", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    stock = st.slider("Mức quyền chọn cổ phiếu (StockOptionLevel)", 0, 3, 0)

with col3:
    st.subheader("3. Thâm niên & Mức độ hài lòng")
    total_years = st.slider("Tổng số năm đi làm (TotalWorkingYears)", 0, 40, 4)
    years_comp = st.slider("Số năm tại công ty (YearsAtCompany)", 0, 40, 2)
    years_promo = st.slider("Số năm kể từ lần thăng chức cuối", 0, 15, 2)
    job_sat = st.slider("Mức độ hài lòng công việc (JobSatisfaction: 1-4)", 1, 4, 2)
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
sample['TotalWorkingYears'] = total_years
sample['YearsAtCompany'] = years_comp
sample['YearsSinceLastPromotion'] = years_promo
sample['JobSatisfaction'] = job_sat
sample['EnvironmentSatisfaction'] = env_sat
sample['WorkLifeBalance'] = wlb

if st.button("Dự đoán nguy cơ nghỉ việc", type="primary"):
    prob_dt = dt_pipe.predict_proba(sample)[0, 1]
    prob_nb = nb_pipe.predict_proba(sample)[0, 1]
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Xác suất nghỉ việc (Decision Tree - Luồng B)", f"{prob_dt*100:.1f}%",
                  "Nguy cơ CAO (Cần can thiệp)" if prob_dt >= 0.5 else "Nguy cơ THẤP (Ổn định)")
    with c2:
        st.metric("Xác suất nghỉ việc (Gaussian NB - Luồng B)", f"{prob_nb*100:.1f}%",
                  "Nguy cơ CAO (Cần can thiệp)" if prob_nb >= 0.5 else "Nguy cơ THẤP (Ổn định)")
