import yaml
import sys
import logging
from src.utils import setup_logger, set_seed
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

logger = setup_logger()

def main(config_path: str = "config.yaml"):
    logger.info("==================================================")
    logger.info("Starting Refactored Hotel No-Show Prediction Pipeline")
    logger.info("==================================================")

    # 1. Load Configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded runtime configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration file at {config_path}: {e}")
        sys.exit(1)

    # 2. Set Global Reproducibility Seed
    seed = config['split']['random_state']
    set_seed(seed)
    logger.info(f"Global random seed initialized to {seed}")

    # 3. Data Preparation Phase
    data_prep = DataPreparation(config)
    data_path = config['data']['raw_path']
    df = data_prep.load_and_clean_data(data_path)

    # 4. Build Preprocessor
    preprocessor = data_prep.build_preprocessor()

    # 5. Model Training Phase
    trainer = ModelTraining(config, preprocessor)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(df)

    # 6. Train Baseline Candidate Models
    best_name, best_pipe, baseline_results = trainer.train_and_evaluate_baselines(X_train, y_train, X_val, y_val)

    # 7. Tune Hyperparameters, Optimize Decision Threshold, Evaluate & Persist Artifacts
    final_pipeline = trainer.tune_and_save(best_pipe, X_train, y_train, X_val, y_val, X_test, y_test)

    logger.info("Pipeline Execution Completed Successfully.")

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
