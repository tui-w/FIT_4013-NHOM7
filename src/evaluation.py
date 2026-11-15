import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def evaluate_estimator(name, stream, stage, model, X_tr, y_tr, X_te, y_te, cv, cv_score=None):
    """Huấn luyện và tính toán đầy đủ các độ đo đánh giá cho một mô hình."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred
    if cv_score is None:
        cv_score = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='f1', n_jobs=1).mean()
    cm = confusion_matrix(y_te, y_pred)
    cost_info = compute_hr_business_cost(y_te, y_pred)
    return {
        'Model': name,
        'Stream': stream,
        'Stage': stage,
        'CV_F1': round(float(cv_score), 4),
        'Accuracy': round(accuracy_score(y_te, y_pred), 4),
        'Precision': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_te, y_pred, zero_division=0), 4),
        'F1': round(f1_score(y_te, y_pred, zero_division=0), 4),
        'ROC_AUC': round(roc_auc_score(y_te, y_prob), 4),
        'PR_AUC': round(average_precision_score(y_te, y_prob), 4),
        'Net_Savings_USD': round(cost_info['net_savings_usd'], 2),
        'CM': cm.tolist(),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'estimator': model
    }


def compute_hr_business_cost(y_true, y_pred, cost_fn=15000.0, cost_fp=1500.0, retention_success_rate=0.50):
    """
    Tính toán hiệu quả kinh tế (Cost-Sensitive HR Analytics) trên tập kiểm tra:
    - cost_fn ($15,000): Chi phí tuyển dụng, đào tạo lại và mất năng suất khi mất 1 nhân viên (False Negative).
    - cost_fp ($1,500): Chi phí chương trình giữ chân / phúc lợi / phỏng vấn HR cho 1 nhân viên được dự báo nghỉ việc.
    - retention_success_rate (50%): Tỷ lệ giữ chân thành công khi HR can thiệp đúng nhân viên sắp nghỉ việc (True Positive).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    baseline_cost = (tp + fn) * cost_fn  # Không có mô hình: mất toàn bộ nhân viên nghỉ việc thực tế
    # Khi dùng mô hình:
    # - FN: không phát hiện được -> mất 100% cost_fn
    # - TP: phát hiện đúng -> tốn chi phí can thiệp cost_fp, giữ lại được retention_success_rate * cost_fn
    # - FP: báo động nhầm -> tốn chi phí can thiệp cost_fp
    model_cost = (
        fn * cost_fn
        + tp * (cost_fp + (1.0 - retention_success_rate) * cost_fn)
        + fp * cost_fp
    )
    net_savings = baseline_cost - model_cost
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'baseline_cost_usd': float(baseline_cost),
        'model_cost_usd': float(model_cost),
        'net_savings_usd': float(net_savings)
    }


def find_optimal_threshold(y_true, y_prob, cost_fn=15000.0, cost_fp=1500.0, retention_success_rate=0.50):
    """Dò tìm ngưỡng quyết định (Decision Threshold) tối ưu theo F1-score và theo Chi phí HR."""
    thresholds = np.linspace(0.15, 0.85, 71)
    records = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cost_res = compute_hr_business_cost(y_true, preds, cost_fn, cost_fp, retention_success_rate)
        records.append({
            'threshold': round(float(t), 4),
            'precision': round(precision_score(y_true, preds, zero_division=0), 4),
            'recall': round(recall_score(y_true, preds, zero_division=0), 4),
            'f1': round(f1_score(y_true, preds, zero_division=0), 4),
            'net_savings_usd': round(cost_res['net_savings_usd'], 2),
            'model_cost_usd': round(cost_res['model_cost_usd'], 2)
        })
    df_thresh = pd.DataFrame(records)
    best_f1_row = df_thresh.loc[df_thresh['f1'].idxmax()].to_dict()
    best_cost_row = df_thresh.loc[df_thresh['net_savings_usd'].idxmax()].to_dict()
    return df_thresh, best_f1_row, best_cost_row


def run_statistical_tests(df, num_cols, cat_cols, target_col='Attrition_Bin'):
    """Kiểm định giả thuyết thống kê Mann-Whitney U (biến số) và Chi-square (biến phân loại)."""
    mw_rows = []
    for c in num_cols:
        g0 = df[df[target_col] == 0][c]
        g1 = df[df[target_col] == 1][c]
        u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        mw_rows.append({
            'Feature': c,
            'Mean_No': round(g0.mean(), 2),
            'Mean_Yes': round(g1.mean(), 2),
            'Median_No': round(g0.median(), 2),
            'Median_Yes': round(g1.median(), 2),
            'Skewness': round(df[c].skew(), 2),
            'p_value': float(p_val),
            'Significant': bool(p_val < 0.05)
        })
    df_mw = pd.DataFrame(mw_rows).sort_values('p_value').reset_index(drop=True)

    chi_rows = []
    for c in cat_cols:
        ct = pd.crosstab(df[c], df[target_col])
        chi2, p_val, dof, _ = stats.chi2_contingency(ct)
        rates = df.groupby(c)[target_col].mean() * 100
        chi_rows.append({
            'Feature': c,
            'Num_Categories': df[c].nunique(),
            'Chi2_Stat': round(float(chi2), 2),
            'DoF': int(dof),
            'p_value': float(p_val),
            'Max_Attrition_Group': f"{rates.idxmax()} ({rates.max():.1f}%)",
            'Min_Attrition_Group': f"{rates.idxmin()} ({rates.min():.1f}%)",
            'Significant': bool(p_val < 0.05)
        })
    df_chi = pd.DataFrame(chi_rows).sort_values('p_value').reset_index(drop=True)
    return df_mw, df_chi
