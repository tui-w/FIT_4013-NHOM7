"""
main.py - Script chạy tái lập toàn bộ Pipeline Học máy cho Đề tài 13:
Dự đoán nhân viên có khả năng nghỉ việc (IBM HR Analytics Employee Attrition).
Hỗ trợ:
  - python main.py           : Chạy đầy đủ kiểm định thống kê, huấn luyện 2 Luồng A/B, đánh giá & phân tích chi phí HR.
  - python main.py --fast    : Sử dụng mô hình đã lưu trong models/best_pipelines.joblib để đánh giá nhanh.
"""
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    RANDOM_STATE, DROP_COLS, CAT_COLS_A, CAT_COLS_B,
    load_and_split_data, build_stream_a_preprocessor, build_stream_b_preprocessor
)
from src.models import get_param_grids, extract_top_cv_configs
from src.evaluation import evaluate_estimator, compute_hr_business_cost, find_optimal_threshold, run_statistical_tests


def main():
    fast_mode = '--fast' in sys.argv
    print("=" * 85)
    print("ĐỀ TÀI 13: DỰ ĐOÁN NHÂN VIÊN CÓ KHẢ NĂNG NGHỈ VIỆC (FIT4013 - NHÓM 7)")
    print("=" * 85)

    df, X_train, X_test, y_train, y_test = load_and_split_data()
    df['Attrition_Bin'] = (df['Attrition'] == 'Yes').astype(int)
    print(f"[1] Dữ liệu gốc: {df.shape[0]} bản ghi x {df.shape[1]-1} cột (Missing: {df.isna().sum().sum()})")
    print(f"    - Tập Train (80%): {X_train.shape[0]} mẫu (Attrition=Yes: {y_train.sum()} mẫu - {y_train.mean()*100:.2f}%)")
    print(f"    - Tập Test  (20%): {X_test.shape[0]} mẫu (Attrition=Yes: {y_test.sum()} mẫu - {y_test.mean()*100:.2f}%)")

    num_cols_a = [c for c in X_train.columns if c not in DROP_COLS + CAT_COLS_A]
    df_mw, df_chi = run_statistical_tests(df, num_cols_a, CAT_COLS_A)
    print("\n[2] TOP 5 ĐẶC TRƯNG SỐ CÓ Ý NGHĨA THỐNG KÊ MẠNH NHẤT (MANN-WHITNEY U TEST):")
    print(df_mw.head(5).to_string(index=False))

    print("\n[3] TOP 5 ĐẶC TRƯNG PHÂN LOẠI CÓ Ý NGHĨA THỐNG KÊ MẠNH NHẤT (CHI-SQUARE TEST):")
    print(df_chi.head(5).to_string(index=False))

    prep_a = build_stream_a_preprocessor(X_train)
    prep_b, num_cols_b = build_stream_b_preprocessor(X_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grids = get_param_grids()
    results = []

    # 1. Baseline
    results.append(evaluate_estimator(
        'Dummy (Most Frequent)', 'Luồng A', 'Baseline',
        Pipeline([('prep', prep_a), ('clf', DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE))]),
        X_train, y_train, X_test, y_test, cv, cv_score=0.0
    ))
    results.append(evaluate_estimator(
        'Dummy (Stratified)', 'Luồng A', 'Baseline',
        Pipeline([('prep', prep_a), ('clf', DummyClassifier(strategy='stratified', random_state=RANDOM_STATE))]),
        X_train, y_train, X_test, y_test, cv
    ))

    model_path = 'models/best_pipelines.joblib'
    saved_models = joblib.load(model_path) if (fast_mode and os.path.exists(model_path)) else {}

    for stream_name, prep in [('Luồng A', prep_a), ('Luồng B', prep_b)]:
        results.append(evaluate_estimator('K-NN', stream_name, 'Default', Pipeline([('prep', prep), ('clf', KNeighborsClassifier())]), X_train, y_train, X_test, y_test, cv))
        results.append(evaluate_estimator('Gaussian NB', stream_name, 'Default', Pipeline([('prep', prep), ('clf', GaussianNB())]), X_train, y_train, X_test, y_test, cv))
        results.append(evaluate_estimator('Decision Tree', stream_name, 'Default', Pipeline([('prep', prep), ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))]), X_train, y_train, X_test, y_test, cv))

        if fast_mode and f'KNN_{stream_name}' in saved_models:
            results.append(evaluate_estimator('K-NN', stream_name, 'Tuned', saved_models[f'KNN_{stream_name}'], X_train, y_train, X_test, y_test, cv))
            results.append(evaluate_estimator('Gaussian NB', stream_name, 'Tuned', saved_models[f'NB_{stream_name}'], X_train, y_train, X_test, y_test, cv))
            results.append(evaluate_estimator('Decision Tree', stream_name, 'Tuned', saved_models[f'DT_{stream_name}'], X_train, y_train, X_test, y_test, cv))
            results.append(evaluate_estimator('Random Forest (Mở rộng)', stream_name, 'Tuned', saved_models[f'RF_{stream_name}'], X_train, y_train, X_test, y_test, cv))
        else:
            gs_knn = GridSearchCV(Pipeline([('prep', prep), ('select', SelectKBest(f_classif)), ('clf', KNeighborsClassifier())]), grids['K-NN'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
            saved_models[f'KNN_{stream_name}'] = gs_knn.best_estimator_
            results.append(evaluate_estimator('K-NN', stream_name, 'Tuned', gs_knn.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_knn.best_score_))

            gs_nb = GridSearchCV(Pipeline([('prep', prep), ('select', SelectKBest(f_classif)), ('clf', GaussianNB())]), grids['Gaussian NB'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
            saved_models[f'NB_{stream_name}'] = gs_nb.best_estimator_
            results.append(evaluate_estimator('Gaussian NB', stream_name, 'Tuned', gs_nb.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_nb.best_score_))

            gs_dt = GridSearchCV(Pipeline([('prep', prep), ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))]), grids['Decision Tree'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
            saved_models[f'DT_{stream_name}'] = gs_dt.best_estimator_
            results.append(evaluate_estimator('Decision Tree', stream_name, 'Tuned', gs_dt.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_dt.best_score_))

            gs_rf = GridSearchCV(Pipeline([('prep', prep), ('clf', RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE))]), grids['Random Forest'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
            saved_models[f'RF_{stream_name}'] = gs_rf.best_estimator_
            results.append(evaluate_estimator('Random Forest (Mở rộng)', stream_name, 'Tuned', gs_rf.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_rf.best_score_))

    if not fast_mode:
        os.makedirs('models', exist_ok=True)
        joblib.dump(saved_models, model_path)

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_prob', 'estimator', 'CM']} for r in results])
    print("\n[4] BẢNG TỔNG HỢP 16 PHƯƠNG ÁN THỰC NGHIỆM & HIỆU QUẢ KINH TẾ HR (NET SAVINGS USD):")
    print(res_df.to_string(index=False))


if __name__ == '__main__':
    main()
