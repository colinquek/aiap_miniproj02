import unittest
import numpy as np
import pandas as pd
from src.data_preparation import DataPreparation, _coerce_numeric, _coerce_binary

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.config = {
            'data': {
                'target_column': 'no_show',
                'required_columns': ['no_show', 'branch', 'booking_platform', 'first_time_customer', 'room', 'price', 'num_adults', 'num_children', 'arrival_month', 'arrival_day', 'checkout_month', 'checkout_day']
            },
            'features': {
                'numerical_features': ['price_usd', 'stay_duration', 'num_adults', 'num_children', 'total_guests', 'price_per_night', 'price_per_guest', 'first_time_x_price'],
                'categorical_features': ['branch', 'booking_platform', 'room', 'arrival_month'],
                'binary_features': ['first_time_customer']
            }
        }

    def test_parse_price_to_usd(self):
        self.assertEqual(DataPreparation.parse_price_to_usd("$100.50"), 100.50)
        self.assertEqual(DataPreparation.parse_price_to_usd(150.0), 150.0)
        self.assertEqual(DataPreparation.parse_price_to_usd("SGD$ 100"), 74.0)
        self.assertTrue(np.isnan(DataPreparation.parse_price_to_usd(None)))
        self.assertTrue(np.isnan(DataPreparation.parse_price_to_usd("invalid_price")))

    def test_compute_stay_duration_vectorized(self):
        data_prep = DataPreparation(self.config)
        df = pd.DataFrame({
            'arrival_month': ['January', 'January', 'December'],
            'checkout_month': ['January', 'February', 'January'],
            'arrival_day': [10, 28, 30],
            'checkout_day': [15, 2, 5]
        })
        durations = data_prep.compute_stay_duration_vectorized(df)
        self.assertEqual(durations.iloc[0], 5)
        self.assertEqual(durations.iloc[1], 5) # 31-28 + 2
        self.assertEqual(durations.iloc[2], 6) # December -> January

    def test_coerce_numeric_handles_word_values(self):
        # Regression: real dataset contains 'one'/'two' in num_adults
        s = pd.Series(['1', 'two', 'one', '3', None, '  TWO '])
        out = _coerce_numeric(s, fill_value=1)
        self.assertEqual(list(out), [1.0, 2.0, 1.0, 3.0, 1.0, 2.0])

    def test_coerce_binary_maps_yes_no(self):
        # Regression: first_time column contains 'Yes'/'No' strings
        s = pd.Series(['Yes', 'No', 'yes', 'no', None, '1'])
        out = _coerce_binary(s, fill_value=0)
        self.assertEqual(list(out), [1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    def test_load_and_clean_writes_back_numeric_guest_columns(self):
        # Regression: raw CSV has 'one'/'two' in num_adults and 'Yes'/'No' in
        # first_time; cleaned values must be written back into df because the
        # raw column names are the model features used by split_data.
        import os
        import tempfile

        df_raw = pd.DataFrame({
            'booking_id': ['B1', 'B2'],
            'no_show': [1, 0],
            'branch': ['A', 'B'],
            'booking_month': ['January', 'February'],
            'arrival_month': ['January', 'January'],
            'arrival_day': [10, 20],
            'checkout_month': ['January', 'January'],
            'checkout_day': [15, 25],
            'country': ['SG', 'MY'],
            'first_time': ['Yes', 'No'],
            'room': ['Deluxe', 'Standard'],
            'price': ['SGD$ 100', 150],
            'platform': ['Web', 'App'],
            'num_adults': ['one', 'two'],
            'num_children': ['1', '0'],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'raw.csv')
            df_raw.to_csv(path, index=False)
            df = DataPreparation(self.config).load_and_clean_data(path)

        self.assertEqual(list(df['num_adults']), [1.0, 2.0])
        self.assertEqual(list(df['num_children']), [1.0, 0.0])
        self.assertEqual(list(df['first_time_customer']), [1.0, 0.0])
        self.assertTrue(pd.api.types.is_numeric_dtype(df['num_adults']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['num_children']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['first_time_customer']))


if __name__ == '__main__':
    unittest.main()
