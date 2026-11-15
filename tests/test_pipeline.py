import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import sys
import unittest
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import (
    load_and_split_data, HRFeatureEngineer,
    build_stream_a_preprocessor, build_stream_b_preprocessor,
    DROP_COLS, CAT_COLS_A, CAT_COLS_B, RANDOM_STATE
)
from src.models import get_param_grids, extract_top_cv_configs
from src.evaluation import compute_hr_business_cost, find_optimal_threshold, run_statistical_tests


class TestAttritionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
        cls.df, cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_and_split_data(data_path)
        cls.df['Attrition_Bin'] = (cls.df['Attrition'] == 'Yes').astype(int)

    def test_dataset_integrity_and_stratified_split(self):
        """Kiểm tra kích thước tập dữ liệu gốc (1470x35), 0 missing values, và tỷ lệ Stratified 80/20."""
        self.assertEqual(self.df.shape[0], 1470)
        self.assertEqual(self.df.isna().sum().sum(), 0)
        self.assertEqual(len(self.X_train), 1176)
        self.assertEqual(len(self.X_test), 294)
        self.assertAlmostEqual(self.y_train.mean(), self.y_test.mean(), places=2)

    def test_hr_feature_engineer_no_leakage(self):
        """Kiểm tra HRFeatureEngineer tạo đúng 7 đặc trưng nghiệp vụ HR, mã hóa thứ bậc và loại bỏ đa cộng tuyến."""
        fe = HRFeatureEngineer(drop_collinear=True)
        X_tr_fe = fe.fit_transform(self.X_train)
        X_te_fe = fe.transform(self.X_test)

        expected_new_cols = [
            'PromotionStagnationRatio', 'CompanyTenureRatio', 'IncomePerWorkingYear',
            'OverallSatisfaction', 'RoleStabilityRatio', 'OverTime_FrequentTravel',
            'YoungSingleLowIncome', 'BusinessTravel_Ord'
        ]
        for col in expected_new_cols:
            self.assertIn(col, X_tr_fe.columns)
            self.assertIn(col, X_te_fe.columns)

        for dropped in DROP_COLS + ['JobLevel', 'DailyRate', 'HourlyRate', 'MonthlyRate', 'BusinessTravel']:
            self.assertNotIn(dropped, X_tr_fe.columns)

        self.assertEqual(X_tr_fe.isna().sum().sum(), 0)
        self.assertEqual(X_te_fe.isna().sum().sum(), 0)

    def test_stream_a_and_b_preprocessors(self):
        """Kiểm tra kích thước ma trận đặc trưng sau biến đổi của Luồng A (44 chiều) và Luồng B (46 chiều)."""
        prep_a = build_stream_a_preprocessor(self.X_train)
        Xa_tr = prep_a.fit_transform(self.X_train)
        Xa_te = prep_a.transform(self.X_test)
        self.assertEqual(Xa_tr.shape, (1176, 44))
        self.assertEqual(Xa_te.shape, (294, 44))

        prep_b, num_cols_b = build_stream_b_preprocessor(self.X_train)
        Xb_tr = prep_b.fit_transform(self.X_train)
        Xb_te = prep_b.transform(self.X_test)
        self.assertEqual(Xb_tr.shape, (1176, 46))
        self.assertEqual(Xb_te.shape, (294, 46))
        self.assertEqual(len(num_cols_b), 27)

    def test_saved_models_joblib_and_metrics(self):
        """Kiểm tra file models/best_pipelines.joblib chứa đủ 8 pipeline đã huấn luyện và dự đoán chính xác."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_pipelines.joblib')
        self.assertTrue(os.path.exists(model_path))
        saved_models = joblib.load(model_path)
        expected_keys = [
            'KNN_Luồng A', 'NB_Luồng A', 'DT_Luồng A', 'RF_Luồng A',
            'KNN_Luồng B', 'NB_Luồng B', 'DT_Luồng B', 'RF_Luồng B'
        ]
        for k in expected_keys:
            self.assertIn(k, saved_models)
            preds = saved_models[k].predict(self.X_test)
            probs = saved_models[k].predict_proba(self.X_test)[:, 1]
            self.assertEqual(len(preds), 294)
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_business_cost_and_statistical_tests(self):
        """Kiểm tra hàm tính chi phí kinh tế HR, tối ưu ngưỡng quyết định và kiểm định thống kê."""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_pipelines.joblib')
        saved_models = joblib.load(model_path)
        rf_b = saved_models['RF_Luồng B']
        y_prob = rf_b.predict_proba(self.X_test)[:, 1]
        df_thresh, best_f1, best_cost = find_optimal_threshold(self.y_test, y_prob)
        self.assertEqual(len(df_thresh), 71)
        self.assertGreater(best_f1['f1'], 0.50)
        self.assertGreater(best_cost['net_savings_usd'], 100000.0)

        num_cols = [c for c in self.X_train.columns if c not in DROP_COLS + CAT_COLS_A]
        df_mw, df_chi = run_statistical_tests(self.df, num_cols, CAT_COLS_A)
        self.assertEqual(len(df_mw), len(num_cols))
        self.assertEqual(len(df_chi), len(CAT_COLS_A))


if __name__ == '__main__':
    unittest.main(verbosity=2)
