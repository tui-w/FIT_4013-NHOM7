import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42
DROP_COLS = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
CAT_COLS_A = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
CAT_COLS_B = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']


def load_and_split_data(csv_path='data/WA_Fn-UseC_-HR-Employee-Attrition.csv', test_size=0.20, random_state=RANDOM_STATE):
    """Đọc dữ liệu gốc và chia tập Train/Test theo phương pháp phân tầng (Stratified Split)."""
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Attrition'])
    y = (df['Attrition'] == 'Yes').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return df, X_train, X_test, y_train, y_test


class HRFeatureEngineer(BaseEstimator, TransformerMixin):
    """Bộ biến đổi đặc trưng tùy chỉnh cho Luồng B (Pipeline cải tiến):
    1. Loại bỏ 4 cột định danh và hằng số.
    2. Tạo 7 đặc trưng nghiệp vụ Nhân sự (HR Domain Features).
    3. Mã hóa thứ bậc tự nhiên cho BusinessTravel (0, 1, 2).
    4. Biến đổi log1p cho 5 đặc trưng liên tục bị lệch phải mạnh.
    5. Loại bỏ biến đa cộng tuyến cao (JobLevel) và biến nhiễu ngẫu nhiên (DailyRate, HourlyRate, MonthlyRate).
    """
    def __init__(self, drop_collinear=True):
        self.drop_collinear = drop_collinear

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        for c in DROP_COLS:
            if c in X_out.columns:
                X_out = X_out.drop(columns=[c])

        # 1. Tạo 7 đặc trưng nghiệp vụ nhân sự
        X_out['PromotionStagnationRatio'] = X_out['YearsSinceLastPromotion'] / (X_out['YearsAtCompany'] + 1.0)
        X_out['CompanyTenureRatio'] = X_out['YearsAtCompany'] / (X_out['TotalWorkingYears'] + 1.0)
        X_out['IncomePerWorkingYear'] = X_out['MonthlyIncome'] / (X_out['TotalWorkingYears'] + 1.0)
        X_out['OverallSatisfaction'] = (
            X_out['EnvironmentSatisfaction'] +
            X_out['JobSatisfaction'] +
            X_out['RelationshipSatisfaction'] +
            X_out['WorkLifeBalance']
        ) / 4.0
        X_out['RoleStabilityRatio'] = X_out['YearsInCurrentRole'] / (X_out['YearsAtCompany'] + 1.0)
        X_out['OverTime_FrequentTravel'] = (
            (X_out['OverTime'] == 'Yes') & (X_out['BusinessTravel'] == 'Travel_Frequently')
        ).astype(int)
        X_out['YoungSingleLowIncome'] = (
            (X_out['Age'] < 30) & (X_out['MaritalStatus'] == 'Single') & (X_out['MonthlyIncome'] < 3500)
        ).astype(int)

        # 2. Mã hóa thứ bậc tự nhiên cho BusinessTravel
        travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}
        X_out['BusinessTravel_Ord'] = X_out['BusinessTravel'].map(travel_map).fillna(1)
        X_out = X_out.drop(columns=['BusinessTravel'])

        # 3. Biến đổi log1p cho các đặc trưng số lệch phải mạnh
        skewed_cols = ['MonthlyIncome', 'TotalWorkingYears', 'YearsAtCompany', 'YearsSinceLastPromotion', 'DistanceFromHome']
        for col in skewed_cols:
            if col in X_out.columns:
                X_out[col] = np.log1p(np.maximum(0, X_out[col]))

        # 4. Loại bỏ cột đa cộng tuyến cao và cột nhiễu
        if self.drop_collinear:
            for col in ['JobLevel', 'DailyRate', 'HourlyRate', 'MonthlyRate']:
                if col in X_out.columns:
                    X_out = X_out.drop(columns=[col])

        return X_out


def build_stream_a_preprocessor(X_sample):
    """Xây dựng bộ tiền xử lý Luồng A (Pipeline cơ bản)."""
    num_cols_a = [c for c in X_sample.columns if c not in DROP_COLS + CAT_COLS_A]
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols_a),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
            ]), CAT_COLS_A)
        ],
        remainder='drop'
    )


def build_stream_b_preprocessor(X_sample):
    """Xây dựng bộ tiền xử lý Luồng B (Pipeline cải tiến)."""
    fe_sample = HRFeatureEngineer(drop_collinear=True).fit_transform(X_sample)
    num_cols_b = [c for c in fe_sample.columns if c not in CAT_COLS_B]
    return Pipeline([
        ('fe', HRFeatureEngineer(drop_collinear=True)),
        ('ct', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), num_cols_b),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
            ]), CAT_COLS_B)
        ], remainder='drop'))
    ]), num_cols_b
