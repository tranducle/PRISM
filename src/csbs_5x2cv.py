import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import copy

def paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring, random_seed=42):
    """Manual implementation of 5x2 CV paired t-test"""
    np.random.seed(random_seed)
    
    p1 = np.zeros(5)
    p2 = np.zeros(5)
    variance_sum = 0.0
    
    for i in range(5):
        kf = KFold(n_splits=2, shuffle=True, random_state=random_seed + i)
        
        diff_folds = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Clone and fit
            e1 = copy.deepcopy(estimator1)
            e2 = copy.deepcopy(estimator2)
            
            e1.fit(X_train, y_train)
            e2.fit(X_train, y_train)
            
            score1 = average_precision_score(y_test, e1.predict_proba(X_test)[:, 1])
            score2 = average_precision_score(y_test, e2.predict_proba(X_test)[:, 1])
            
            diff_folds.append(score1 - score2)
            
        p1[i] = diff_folds[0]
        p2[i] = diff_folds[1]
        
        variance_sum += (diff_folds[0] - np.mean(diff_folds))**2 + (diff_folds[1] - np.mean(diff_folds))**2
        
    numerator = p1[0]
    denominator = np.sqrt(variance_sum / 5.0)
    
    t_stat = numerator / denominator
    p_value = stats.t.sf(np.abs(t_stat), 5) * 2
    
    return t_stat, p_value

def run_5x2cv_test(X, y):
    """
    Runs a 5x2 cross-validation paired t-test comparing XGBoost against baseline classifiers.
    """
    models = {
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'ANN': MLPClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000)
    }
    
    results = {}
    base_model_name = 'XGBoost'
    base_model = models[base_model_name]
    
    print(f"Running 5x2 CV paired t-test against {base_model_name}")
    print("-" * 50)
    
    for name, model in models.items():
        if name == base_model_name:
            continue
            
        print(f"Comparing {base_model_name} vs {name}...")
        
        # We use mlxtend for robust 5x2cv evaluation
        t_stat, p_value = paired_ttest_5x2cv(
            estimator1=base_model,
            estimator2=model,
            X=X,
            y=y,
            scoring='average_precision',
            random_seed=42
        )
        
        results[name] = {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f} (Significant diff? {'Yes' if p_value < 0.05 else 'No'})")
        
    return results

if __name__ == "__main__":
    import sys
    import os
    # Add scripts dir to path to import 05_stage1_ml functions
    paper_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(paper_dir, '2_Data', 'scripts')
    sys.path.append(scripts_dir)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage1_ml", os.path.join(scripts_dir, "05_stage1_ml.py"))
    stage1_ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1_ml)
    
    PROC = os.path.join(os.path.dirname(os.path.dirname(scripts_dir)), "2_Data", "processed")
    
    print("Loading CSBS dataset...")
    df = pd.read_csv(os.path.join(PROC, "csbs_pooled_clean.csv"), low_memory=False)
    X, y, groups, feature_cols = stage1_ml.prepare_data(df)
    
    print("Starting 5x2 CV tests...")
    
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    # Simple imputation as done in the pipeline
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    
    # Scale for models that need it (SVM, Logistic, ANN) is handled in run_5x2cv_test
    # Actually wait, in my run_5x2cv_test I didn't add scaling. I should just run it on X_imp for tree models.
    # Let me just use the base pipelines from stage1_ml for a completely fair test.
    
    models = stage1_ml.build_models(42)
    base_model_name = 'XGBoost'
    base_model = models[base_model_name]
    
    print(f"Running 5x2 CV paired t-test against {base_model_name}")
    print("-" * 50)
    
    results = {}
    for name, model in models.items():
        if name == base_model_name:
            continue
            
        print(f"Comparing {base_model_name} vs {name}...")
        
        t_stat, p_value = paired_ttest_5x2cv(
            estimator1=base_model,
            estimator2=model,
            X=X,
            y=y,
            scoring='average_precision',
            random_seed=42
        )
        
        results[name] = {
            't_stat': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f} (Significant diff? {'Yes' if p_value < 0.05 else 'No'})")

