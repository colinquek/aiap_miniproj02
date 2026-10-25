import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'project_name': 'Test Pipeline',
            'data': {
                'target_column': 'no_show',
                'required_columns': ['no_show', 'branch', 'booking_platform', 'first_time_customer', 'room', 'price', 'num_adults', 'num_children', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day']
            },
            'split': {'test_size': 0.2, 'val_size': 0.25, 'random_state': 42, 'stratify': True},
            'features': {
                'numerical_features': ['price_usd', 'stay_duration', 'num_adults', 'num_children', 'total_guests', 'price_per_night', 'price_per_guest', 'first_time_x_price'],
                'categorical_features': ['branch', 'booking_platform', 'room', 'arrival_month'],
                'binary_features': ['first_time_customer']
            },
            'models': {
                'save_dir': self.test_dir,
                'model_filename': 'test_model.joblib',
                'metrics_filename': 'metrics.json',
                'threshold_filename': 'thresh.json'
            },
            'hyperparameters': {'cv_folds': 2, 'scoring': 'roc_auc'}
        }

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_full_pipeline_end_to_end(self):
        N = 100
        df_raw = pd.DataFrame({
            'no_show': np.random.choice([0, 1], size=N, p=[0.7, 0.3]),
            'branch': np.random.choice(['East', 'West', 'Central'], size=N),
            'booking_platform': np.random.choice(['Direct', 'OTA1', 'OTA2'], size=N),
            'first_time_customer': np.random.choice([0, 1], size=N),
            'room': np.random.choice(['Standard', 'Deluxe', 'Suite'], size=N),
            'price': np.random.uniform(50, 300, size=N),
            'num_adults': np.random.randint(1, 4, size=N),
            'num_children': np.random.randint(0, 3, size=N),
            'arrival_month': np.random.choice(['January', 'February', 'March'], size=N),
            'arrival_day': np.random.randint(1, 20, size=N),
            'checkout_month': np.random.choice(['January', 'February', 'March'], size=N),
            'checkout_day': np.random.randint(21, 28, size=N)
        })

        raw_csv = os.path.join(self.test_dir, "raw_data.csv")
        df_raw.to_csv(raw_csv, index=False)

        # 1. Data Prep
        data_prep = DataPreparation(self.config)
        df = data_prep.load_and_clean_data(raw_csv)
        preprocessor = data_prep.build_preprocessor()

        # 2. Model Training
        trainer = ModelTraining(self.config, preprocessor)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(df)
        
        best_name, best_pipe, baseline_results = trainer.train_and_evaluate_baselines(X_train, y_train, X_val, y_val)
        final_pipe = trainer.tune_and_save(best_pipe, X_train, y_train, X_val, y_val, X_test, y_test)

        self.assertIsNotNone(final_pipe)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'test_model.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'metrics.json')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'thresh.json')))

if __name__ == '__main__':
    unittest.main()
