import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# In this file the cleaned data is loaded
# We perform a split of train and test data by authors (as there are multiple popsts per author)
# The cross validation is applied under the training data only to pick the parameters
# We fit the model on the train data 
# We evaluate the performance with the test data and save the outputs

def run_training_pipeline(cfg, data_path=None, run_name="run"):
    if data_path is None:
        data_path = cfg.data_path
    
    run_output_dir = os.path.join(cfg.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df["post_clean"] = df["post_clean"].astype(str)
    df["female"] = pd.to_numeric(df["female"], errors="coerce")
    df = df.dropna(subset=["female"])
    df["female"] = df["female"].astype(int)
    df = df[df["female"].isin([0, 1])]
    df = df[df["post_clean"].str.strip().astype(bool)].reset_index(drop=True)
    
    print("[Data] Posts:", len(df))
    print("[Data] Authors:", df["auhtor_ID"].nunique())
    print("[Data] Class counts:\n", df["female"].value_counts())
    
    # Train/Test split
    # Each author should appear only on train or test, never on both
    author_labels = df.groupby("auhtor_ID")["female"].first()
    authors = author_labels.index.to_numpy()
    y_auth = author_labels.to_numpy()
    
    train_auth, test_auth = train_test_split(
        authors,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_auth,
    )
    
    train_df = df[df["auhtor_ID"].isin(train_auth)].reset_index(drop=True)
    test_df = df[df["auhtor_ID"].isin(test_auth)].reset_index(drop=True)
    
    print("[Split] Train posts:", len(train_df), "Test posts:", len(test_df))
    print("[Split] Train authors:", train_df["auhtor_ID"].nunique(), "Test authors:", test_df["auhtor_ID"].nunique())

    X_train = train_df["post_clean"].to_numpy()
    y_train = train_df["female"].to_numpy()
    g_train = train_df["auhtor_ID"].to_numpy()

    X_test = test_df["post_clean"].to_numpy()
    y_test = test_df["female"].to_numpy()
    
    def make_pipeline(ngram_range, min_df, C):
        tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=cfg.tfidf_max_df,
            max_features=cfg.tfidf_max_features,
            lowercase=False,  
        )

        lr = LogisticRegression(
            C=C,
            solver="saga",
            max_iter=cfg.lr_max_iter,
            class_weight=cfg.lr_class_weight,
        )

        return Pipeline([("tfidf", tfidf), ("lr", lr)])
    
    # Cross validation on train data
    # We are using StratifiedGroupKFold to keep class balance and 
    # to not let authors leak across folds
    
    cv = StratifiedGroupKFold(
        n_splits=cfg.k_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )
    
    results = []
    best_score = -1
    best_params = None
    print("[CV] Running cross validation")
    
    for ngram_range in cfg.tfidf_ngram_ranges:
        for min_df in cfg.tfidf_min_dfs:
            for C in cfg.lr_Cs:
                fold_scores = []
                
                for tr_idx, va_idx in cv.split(X_train, y_train, groups=g_train):
                    pipe = make_pipeline(ngram_range, min_df, C)
                    pipe.fit(X_train[tr_idx], y_train[tr_idx])

                    probability = pipe.predict_proba(X_train[va_idx])[:, 1]

                    if cfg.selection_metric == "roc_auc":
                        score = roc_auc_score(y_train[va_idx], probability)
                    else:
                        pred = (probability >= 0.5).astype(int)
                        score = f1_score(y_train[va_idx], pred, zero_division=0)

                    fold_scores.append(score)
                
                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))
                results.append({
                    "ngram_range": str(ngram_range),
                    "min_df": min_df,
                    "C": C,
                    f"{cfg.selection_metric}_mean": mean_score,
                    f"{cfg.selection_metric}_std": std_score,
                })
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = (ngram_range, min_df, C)
                    
    cv_df = pd.DataFrame(results)
    cv_path = os.path.join(run_output_dir, cfg.cv_filename)
    cv_df.to_csv(cv_path, index=False)
    print("[CV] Saved:", cv_path)
    print("[CV] Best parameters:", best_params, "Best cross validation score:", best_score)
    
    # Training the final model and doing model evaluation
    best_ngram_range, best_min_df, best_C = best_params
    final_model = make_pipeline(best_ngram_range, best_min_df, best_C)
    final_model.fit(X_train, y_train)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    
    metrics = {
        "run_name": run_name,
        "data_path": data_path,
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "best_params": {
            "ngram_range": str(best_ngram_range),
            "min_df": int(best_min_df),
            "C": float(best_C),
        }
    }

    metrics_path = os.path.join(run_output_dir, cfg.metrics_filename)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("[Final] Saved metrics:", metrics_path)
    
    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    roc_path = os.path.join(run_output_dir, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print("[Final] Saved ROC plot:", roc_path)
    
    # Top tokens for interpretation
    tfidf = final_model.named_steps["tfidf"]
    lr = final_model.named_steps["lr"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = lr.coef_.ravel()
    top_n = 20
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]  # This for female = 1
    top_neg_idx = np.argsort(coefs)[:top_n]         # This for female = 0
    top_tokens_df = pd.DataFrame({
        "top_positive_token": feature_names[top_pos_idx],
        "top_positive_coef": coefs[top_pos_idx],
        "top_negative_token": feature_names[top_neg_idx],
        "top_negative_coef": coefs[top_neg_idx],
    })
    tokens_path = os.path.join(run_output_dir, "top_tokens.csv")
    top_tokens_df.to_csv(tokens_path, index=False)
    print("[Final] Saved top tokens:", tokens_path)
    
    return metrics