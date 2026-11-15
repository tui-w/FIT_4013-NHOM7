"""
main.py - Script chạy tái lập toàn bộ Pipeline Học máy cho Đề tài 13:
Dự đoán nhân viên có khả năng nghỉ việc (IBM HR Analytics Employee Attrition).
Chạy lệnh: python main.py
"""
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    RANDOM_STATE, load_and_split_data,
    build_stream_a_preprocessor, build_stream_b_preprocessor
)
from src.models import get_param_grids
from src.evaluation import evaluate_estimator


def main():
    print("=" * 75)
    print("ĐỀ TÀI 13: DỰ ĐOÁN NHÂN VIÊN CÓ KHẢ NĂNG NGHỈ VIỆC (FIT4013)")
    print("=" * 75)

    df, X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"[1] Đã tải dữ liệu: {df.shape[0]} bản ghi x {df.shape[1]} cột")
    print(f"    - Tập Train: {X_train.shape[0]} mẫu (Attrition=Yes: {y_train.sum()} mẫu - {y_train.mean()*100:.2f}%)")
    print(f"    - Tập Test : {X_test.shape[0]} mẫu (Attrition=Yes: {y_test.sum()} mẫu - {y_test.mean()*100:.2f}%)")

    prep_a = build_stream_a_preprocessor(X_train)
    prep_b, _ = build_stream_b_preprocessor(X_train)

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

    # 2. Default & Tuned Models across Stream A and Stream B
    for stream_name, prep in [('Luồng A', prep_a), ('Luồng B', prep_b)]:
        results.append(evaluate_estimator('K-NN', stream_name, 'Default', Pipeline([('prep', prep), ('clf', KNeighborsClassifier())]), X_train, y_train, X_test, y_test, cv))
        results.append(evaluate_estimator('Gaussian NB', stream_name, 'Default', Pipeline([('prep', prep), ('clf', GaussianNB())]), X_train, y_train, X_test, y_test, cv))
        results.append(evaluate_estimator('Decision Tree', stream_name, 'Default', Pipeline([('prep', prep), ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))]), X_train, y_train, X_test, y_test, cv))

        gs_knn = GridSearchCV(Pipeline([('prep', prep), ('select', SelectKBest(f_classif)), ('clf', KNeighborsClassifier())]), grids['K-NN'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
        results.append(evaluate_estimator('K-NN', stream_name, 'Tuned', gs_knn.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_knn.best_score_))

        gs_nb = GridSearchCV(Pipeline([('prep', prep), ('select', SelectKBest(f_classif)), ('clf', GaussianNB())]), grids['Gaussian NB'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
        results.append(evaluate_estimator('Gaussian NB', stream_name, 'Tuned', gs_nb.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_nb.best_score_))

        gs_dt = GridSearchCV(Pipeline([('prep', prep), ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))]), grids['Decision Tree'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
        results.append(evaluate_estimator('Decision Tree', stream_name, 'Tuned', gs_dt.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_dt.best_score_))

        gs_rf = GridSearchCV(Pipeline([('prep', prep), ('clf', RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE))]), grids['Random Forest'], cv=cv, scoring='f1', n_jobs=1).fit(X_train, y_train)
        results.append(evaluate_estimator('Random Forest (Mở rộng)', stream_name, 'Tuned', gs_rf.best_estimator_, X_train, y_train, X_test, y_test, cv, cv_score=gs_rf.best_score_))

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_pred', 'y_prob', 'estimator']} for r in results])
    print("\n[2] BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM:")
    print(res_df.to_string(index=False))


if __name__ == '__main__':
    main()
