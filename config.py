from dataclasses import dataclass

@dataclass
class Config:
    # Data
    data_path: str = "gender_cleaned.csv"   
    output_dir: str = "outputs"

    # Split 
    test_size: float = 0.20
    random_state: int = 42

    # Cross Validation 
    k_folds: int = 5
    selection_metric: str = "f1"  

    # TF-IDF 
    tfidf_ngram_ranges = [(1, 1), (1, 2)]
    tfidf_min_dfs = [2, 5]
    tfidf_max_df: float = 0.95
    tfidf_max_features = None 

    # Logistic Regression
    lr_Cs =  [0.5, 1.0, 2.0]
    lr_max_iter: int = 2000
    lr_class_weight = None  

    # Saving
    metrics_filename: str = "final_metrics.json"
    cv_filename: str = "cv_results.csv"
