import json
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

def main():
    print("\033[1;32m[INFO] Starting End-to-End Hotel No-Show Prediction Pipeline...\033[0m")

    # Simple YAML reader fallback
    config = {
        'raw_data_path': './data/mini_project_2_data.csv',
        'target_column': 'no_show',
        'id_column': 'booking_id',
        'sgd_to_usd_rate': 1.28,
        'split_ratios': {'test_size': 0.20, 'val_size': 0.50, 'random_state': 42},
        'numerical_features': ['price_usd', 'stay_duration', 'num_adults', 'num_children', 'arrival_day', 'checkout_day'],
        'categorical_features': ['branch', 'booking_month', 'arrival_month', 'checkout_month', 'country', 'first_time', 'room', 'platform'],
        'model_save_path': './models/best_no_show_model.joblib'
    }

    # Step 1: Data Preparation
    data_prep = DataPreparation(config)
    df_clean = data_prep.load_and_clean_data(config['raw_data_path'])
    print(f"\033[1;32m[INFO] Cleaned dataset shape: {df_clean.shape}\033[0m")

    preprocessor = data_prep.build_preprocessor()

    # Step 2: Model Training & Evaluation
    trainer = ModelTraining(config, preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(df_clean)

    print("\033[1;32m[INFO] Training Baseline Models...\033[0m")
    baseline_results = trainer.train_baselines(X_train, y_train, X_val, y_val)
    print("Baseline Metrics (Validation Set):")
    print(json.dumps(baseline_results, indent=2))

    print("\033[1;32m[INFO] Tuning Decision Tree & Persisting Artifacts...\033[0m")
    best_model, best_params, test_metrics = trainer.tune_and_save(X_train, y_train, X_test, y_test)
    print(f"Best Hyperparameters: {best_params}")
    print("Final Holdout Test Evaluation Metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("\033[1;32m[INFO] Pipeline Execution Successfully Completed!\033[0m")

if __name__ == '__main__':
    main()
