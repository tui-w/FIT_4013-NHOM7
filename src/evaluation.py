import numpy as np
import pandas as pd
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
        'CM': confusion_matrix(y_te, y_pred).tolist(),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'estimator': model
    }
