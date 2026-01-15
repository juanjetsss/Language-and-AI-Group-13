import os
from config import Config
from preprocessing import load_clean_and_filter, preprocess_dataset
from train_eval import run_training_pipeline


if __name__ == "__main__":
    cfg = Config()
    raw_path = "gender.csv"
    cleaned_path = "gender_cleaned.csv"
    cleaned_punct_path = "gender_cleaned_punct.csv"
    
    if os.path.exists(cleaned_path) and os.path.exists(cleaned_punct_path):
        print("[Main] Cleaned datasets already exist, skipping preprocessing.")
    else:
        print("[Main] Cleaned datasets not found, starting preprocessing")
        base_df = load_clean_and_filter(raw_path, allowed_languages={"en"})
        df_clean = preprocess_dataset(
            csv_path=raw_path,
            allowed_languages={"en"},
            spacy_model="en_core_web_sm",
            min_token_len=2,
            n_process=1,
            keep_punct_tokens=False,
            df_preloaded=base_df,
        )
        df_clean[["auhtor_ID", "post_clean", "female"]].to_csv("gender_cleaned.csv", index=False)
        df_clean_punct = preprocess_dataset(
            csv_path=raw_path,
            allowed_languages={"en"},
            spacy_model="en_core_web_sm",
            min_token_len=2,
            n_process=1,
            keep_punct_tokens=True,
            df_preloaded=base_df,
        )
        df_clean_punct[["auhtor_ID", "post_clean", "female"]].to_csv("gender_cleaned_punct.csv", index=False)

        print("\n[Main] Saved cleaned datasets:")
        print("  - gender_cleaned.csv")
        print("  - gender_cleaned_punct.csv")
    
    runs = [
        ("baseline", "gender_cleaned.csv"),
        ("punct", "gender_cleaned_punct.csv"),
    ]
    for run_name, data_path in runs:
        run_training_pipeline(cfg, data_path=data_path, run_name=run_name)

        run_output_dir = os.path.join(cfg.output_dir, run_name)
        print(f"\nFinished run: {run_name}")
        print(f"Outputs saved in: {run_output_dir}")
        print("Files:")
        print(f"  - {cfg.cv_filename}")
        print(f"  - {cfg.metrics_filename}")
        print(f"  - roc_curve.png")
        print(f"  - top_tokens.csv")
          
