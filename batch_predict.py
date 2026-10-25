import sys
import json
import joblib
import pandas as pd
import yaml
from src.utils import setup_logger
from src.data_preparation import DataPreparation, _coerce_numeric, _coerce_binary

logger = setup_logger()

def run_batch_prediction(input_csv: str, output_csv: str, config_path: str = "config.yaml"):
    logger.info(f"Starting batch prediction pipeline for input file: {input_csv}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_path = f"{config['models']['save_dir']}/{config['models']['model_filename']}"
    thresh_path = f"{config['models']['save_dir']}/{config['models']['threshold_filename']}"

    # Load Model Pipeline
    pipeline = joblib.load(model_path)
    
    # Load Optimal Threshold
    with open(thresh_path, 'r') as f:
        thresh_data = json.load(f)
    threshold = thresh_data.get('optimal_threshold', 0.5)

    # Data Preparation & Feature Engineering
    data_prep = DataPreparation(config)
    df_raw = pd.read_csv(input_csv)
    df_raw = data_prep.normalize_column_names(df_raw)
    
    # Preprocess features without requiring target column for batch inference
    df = df_raw.copy()
    if 'price_usd' not in df.columns and 'price' in df.columns:
        df['price_usd'] = df['price'].apply(data_prep.parse_price_to_usd)
    
    df['room'] = df['room'].fillna('Unknown')
    df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())
    df['stay_duration'] = data_prep.compute_stay_duration_vectorized(df)
    
    num_adults = _coerce_numeric(df['num_adults'], fill_value=1)
    df['num_adults'] = num_adults
    num_children = _coerce_numeric(df['num_children'], fill_value=0)
    df['num_children'] = num_children
    df['total_guests'] = num_adults + num_children
    df['price_per_night'] = df['price_usd'] / df['stay_duration'].replace(0, 1)
    df['price_per_guest'] = df['price_usd'] / df['total_guests'].replace(0, 1)
    
    first_time_col = 'first_time_customer' if 'first_time_customer' in df.columns else 'first_time'
    first_time = _coerce_binary(df[first_time_col], fill_value=0)
    df[first_time_col] = first_time
    df['first_time_x_price'] = first_time * df['price_usd']

    feature_cols = config['features']['numerical_features'] + config['features']['categorical_features'] + config['features']['binary_features']
    X_infer = df[feature_cols]

    # Predict
    probs = pipeline.predict_proba(X_infer)[:, 1]
    preds = (probs >= threshold).astype(int)

    df_out = df_raw.copy()
    df_out['no_show_probability'] = probs
    df_out['predicted_no_show'] = preds

    df_out.to_csv(output_csv, index=False)
    logger.info(f"Batch prediction completed successfully. Saved predictions to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python batch_predict.py <input_csv> <output_csv> [config_path]")
        sys.exit(1)
    
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    cfg_file = sys.argv[3] if len(sys.argv) > 3 else "config.yaml"
    run_batch_prediction(in_file, out_file, cfg_file)
