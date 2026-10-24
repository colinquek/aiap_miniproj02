import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataPreparation:
    """
    DataPreparation handles dataset loading, currency conversion, missing value imputation,
    stay duration feature engineering, and ColumnTransformer pipeline construction.
    """
    def __init__(self, config: dict):
        self.config = config
        self.sgd_to_usd_rate = config.get('sgd_to_usd_rate', 1.28)
        self.days_in_month = {
            'January': 31, 'February': 28, 'March': 31, 'April': 30, 'May': 31, 'June': 30,
            'July': 31, 'August': 31, 'September': 30, 'October': 31, 'November': 30, 'December': 31
        }
        self.num_str_map = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}

    def parse_price_to_usd(self, raw_price):
        if pd.isna(raw_price):
            return np.nan
        p_str = str(raw_price).strip()
        if 'SGD$' in p_str:
            return float(p_str.replace('SGD$', '').strip()) / self.sgd_to_usd_rate
        elif 'USD$' in p_str:
            return float(p_str.replace('USD$', '').strip())
        else:
            try:
                return float(p_str)
            except:
                return np.nan

    def compute_stay_duration(self, row):
        a_m, c_m = row['arrival_month'], row['checkout_month']
        a_d, c_d = row['arrival_day'], row['checkout_day_clean']
        if pd.isna(a_m) or pd.isna(c_m) or pd.isna(a_d) or pd.isna(c_d):
            return np.nan
        if a_m == c_m:
            return c_d - a_d
        else:
            return (self.days_in_month.get(a_m, 30) - a_d) + c_d

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        df_raw = pd.read_csv(file_path)
        # Drop rows with missing target label
        target_col = self.config.get('target_column', 'no_show')
        df = df_raw.dropna(subset=[target_col]).copy().reset_index(drop=True)
        df[target_col] = df[target_col].astype(int)

        # Standardise price in USD
        df['price_usd'] = df['price'].apply(self.parse_price_to_usd)

        # Standardise text numbers
        df['num_adults'] = df['num_adults'].replace(self.num_str_map).astype(float)
        df['num_children'] = df['num_children'].replace(self.num_str_map).astype(float)

        # Room and price imputation
        df['room'] = df['room'].fillna('Unknown')
        price_medians = df.groupby(['branch', 'room'])['price_usd'].transform('median')
        df['price_usd'] = df['price_usd'].fillna(price_medians)

        # Correct negative checkout days and calculate stay duration
        df['checkout_day_clean'] = df['checkout_day'].abs()
        df['stay_duration'] = df.apply(self.compute_stay_duration, axis=1)

        # Clean auxiliary columns
        df.drop(columns=['price', 'checkout_day_clean'], inplace=True, errors='ignore')
        return df

    def build_preprocessor(self) -> ColumnTransformer:
        num_cols = self.config['numerical_features']
        cat_cols = self.config['categorical_features']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        return preprocessor
