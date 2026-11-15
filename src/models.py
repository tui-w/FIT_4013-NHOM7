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

RANDOM_STATE = 42


def get_param_grids():
    """Trả về không gian tìm kiếm siêu tham số cho các thuật toán."""
    return {
        'K-NN': {
            'select__k': [15, 20, 25, 'all'],
            'clf__n_neighbors': [3, 5, 7, 9, 11, 15, 19],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan']
        },
        'Gaussian NB': {
            'select__k': [15, 20, 25, 30, 'all'],
            'clf__var_smoothing': np.logspace(-9, 0, 19)
        },
        'Decision Tree': {
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [3, 4, 5, 6, 8, 10, None],
            'clf__min_samples_leaf': [5, 10, 15, 20],
            'clf__class_weight': [None, 'balanced']
        },
        'Random Forest': {
            'clf__max_depth': [4, 6, 8],
            'clf__min_samples_leaf': [5, 10],
            'clf__class_weight': ['balanced', 'balanced_subsample']
        }
    }


def extract_top_cv_configs(grid_search, top_n=5):
    """Trích xuất Top N cấu hình siêu tham số tốt nhất từ GridSearchCV."""
    cv_df = pd.DataFrame(grid_search.cv_results_)
    cv_df = cv_df.sort_values(by='rank_test_score').head(top_n)
    cols = ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
    out = cv_df[cols].copy()
    out['mean_test_score'] = out['mean_test_score'].round(4)
    out['std_test_score'] = out['std_test_score'].round(4)
    out['params'] = out['params'].apply(lambda d: ', '.join(f"{k.replace('clf__', '').replace('select__', 'k_feat=')}: {v}" for k, v in d.items()))
    out.columns = ['Hạng', 'CV F1 (Mean)', 'CV F1 (Std)', 'Cấu hình tham số']
    return out
