import numpy as np
import pandas as pd
import os
import sys
import shap
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def calculate_jaccard(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def get_rf_shap(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return sv[1]
    elif len(sv.shape) == 3:
        return sv[:, :, 1]
    return sv

def run_robustness_test():
    paper_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scripts_dir = os.path.join(paper_dir, '2_Data', 'scripts')
    sys.path.append(scripts_dir)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage1_ml", os.path.join(scripts_dir, "05_stage1_ml.py"))
    stage1_ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage1_ml)
    
    PROC = os.path.join(os.path.dirname(os.path.dirname(scripts_dir)), "2_Data", "processed")
    
    print("1. Loading CSBS dataset...")
    df = pd.read_csv(os.path.join(PROC, "csbs_pooled_clean.csv"), low_memory=False)
    X, y, groups, feature_cols = stage1_ml.prepare_data(df)
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print("2. Training Baseline RF Model to get ground truth SHAP rankings...")
    base_model = RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    base_model.fit(X_imp, y)
    
    sample_idx = np.random.choice(len(X_imp), min(2000, len(X_imp)), replace=False)
    X_sample = X_imp.iloc[sample_idx]
    
    shap_values_base = get_rf_shap(base_model, X_sample)
    mean_abs_shap_base = np.abs(shap_values_base).mean(axis=0)
    base_ranking = np.argsort(mean_abs_shap_base)[::-1]
    base_top10 = X_imp.columns[base_ranking[:10]].tolist()
    base_top20 = X_imp.columns[base_ranking[:20]].tolist()
    
    print(f"Baseline Top 10: {base_top10}")
    
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    
    print("3. Running Falsification Test (Asymmetric Noise - Social Desirability)...")
    
    binary_cols = [c for c in X_imp.columns if set(X_imp[c].unique()).issubset({0, 1})]
    
    for sigma in noise_levels:
        print(f"--- Asymmetric Noise Level: {sigma*100}% ---")
        jaccards = []
        spearmans = []
        
        for seed in seeds:
            np.random.seed(seed)
            
            X_noisy = X_imp.copy()
            if sigma > 0:
                # Asymmetric Noise: Over-reporting. Firms lying that they have a control (0 -> 1)
                for col in binary_cols:
                    # Find instances where they don't have the control
                    zeros_mask = (X_noisy[col] == 0)
                    # Flip a coin for those zeros to become 1 with probability sigma
                    flip_mask = np.random.rand(zeros_mask.sum()) < sigma
                    # Apply flips
                    X_noisy.loc[zeros_mask, col] = np.where(flip_mask, 1, 0)
            
            noisy_model = RandomForestClassifier(n_estimators=50, max_depth=6, class_weight='balanced', random_state=seed, n_jobs=-1)
            noisy_model.fit(X_noisy, y)
            
            shap_values_noisy = get_rf_shap(noisy_model, X_noisy.iloc[sample_idx])
            mean_abs_shap_noisy = np.abs(shap_values_noisy).mean(axis=0)
            
            noisy_ranking = np.argsort(mean_abs_shap_noisy)[::-1]
            noisy_top10 = X_noisy.columns[noisy_ranking[:10]].tolist()
            
            j_score = calculate_jaccard(base_top10, noisy_top10)
            jaccards.append(j_score)
            
            base_ranks_for_top20 = [base_top20.index(f) for f in base_top20]
            noisy_ranks_for_top20 = []
            noisy_full_list = X_noisy.columns[noisy_ranking].tolist()
            for f in base_top20:
                noisy_ranks_for_top20.append(noisy_full_list.index(f))
                
            rho, _ = spearmanr(base_ranks_for_top20, noisy_ranks_for_top20)
            spearmans.append(rho)
            
        results.append({
            'Noise_Level': sigma,
            'Jaccard_Mean': np.mean(jaccards),
            'Jaccard_Std': np.std(jaccards),
            'Spearman_Mean': np.mean(spearmans),
            'Spearman_Std': np.std(spearmans)
        })
        
    df_results = pd.DataFrame(results)
    
    # Generate Professional Plot
    print("4. Generating fig_robustness.pdf...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = df_results['Noise_Level'] * 100
    j_mean = df_results['Jaccard_Mean']
    j_std = df_results['Jaccard_Std']
    s_mean = df_results['Spearman_Mean']
    s_std = df_results['Spearman_Std']
    
    ax.plot(x, j_mean, label='Jaccard (Top 10)', color='#ef4444', marker='o', linewidth=2)
    ax.fill_between(x, j_mean - j_std, j_mean + j_std, color='#ef4444', alpha=0.2)
    
    ax.plot(x, s_mean, label='Spearman Rho (Top 20)', color='#3b82f6', marker='s', linewidth=2)
    ax.fill_between(x, s_mean - s_std, s_mean + s_std, color='#3b82f6', alpha=0.2)
    
    # 70% threshold line
    ax.axhline(y=0.7, color='#6b7280', linestyle='--', linewidth=1.5, label='0.70 Stability Threshold')
    
    ax.set_xlabel('Asymmetric Falsification Noise (%, 0 \u2192 1)', fontsize=11, weight='bold')
    ax.set_ylabel('Ranking Stability / Correlation', fontsize=11, weight='bold')
    ax.set_title('Robustness against Reporting Bias (Social Desirability)', fontsize=12, weight='bold', pad=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#d1d5db')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    figs_dir = os.path.join(paper_dir, '7_Manuscript_Draft', 'MANUSCRIPT', 'figs')
    os.makedirs(figs_dir, exist_ok=True)
    out_pdf = os.path.join(figs_dir, 'fig_robustness.pdf')
    plt.tight_layout()
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved plot to {out_pdf}")

if __name__ == "__main__":
    run_robustness_test()
