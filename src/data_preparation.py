import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import setup_logger

logger = setup_logger()

MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

ALIAS_MAP = {
    'platform': 'booking_platform',
    'booking_channel': 'booking_platform',
    'first_time': 'first_time_customer',
    'is_first_time': 'first_time_customer'
}

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}


def _coerce_numeric(series: pd.Series, fill_value: float) -> pd.Series:
    """Coerces a mixed-format column (numeric strings, word numbers, NaN) to float."""
    lowered = series.astype(str).str.strip().str.lower()
    word_replaced = lowered.replace(WORD_TO_NUM, regex=False)
    return pd.to_numeric(word_replaced, errors='coerce').fillna(fill_value)


def _coerce_binary(series: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Coerces a Yes/No or 0/1 column to 0.0/1.0 float (NaN -> fill_value)."""
    lowered = series.astype(str).str.strip().str.lower()
    binary_map = {'yes': 1.0, 'no': 0.0, 'true': 1.0, 'false': 0.0, 'y': 1.0, 'n': 0.0}
    mapped = lowered.replace(binary_map, regex=False)
    return pd.to_numeric(mapped, errors='coerce').fillna(fill_value)


class DataPreparation:
    """Encapsulates data loading, schema validation, vectorized cleaning, 
    and feature engineering for the Hotel No-Show dataset.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_col = config['data']['target_column']
        self.req_cols = config['data']['required_columns']

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes raw header aliases (e.g. platform -> booking_platform, first_time -> first_time_customer)."""
        renames = {}
        for col in df.columns:
            if col in ALIAS_MAP:
                canonical = ALIAS_MAP[col]
                renames[col] = canonical
            else:
                # Reverse mapping if config expects raw alias name
                for alias, canonical in ALIAS_MAP.items():
                    if col == canonical and alias in self.req_cols:
                        renames[col] = alias

        if renames:
            logger.info(f"Normalizing raw column headers: {renames}")
            df = df.rename(columns=renames)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        """Validates that all required columns are present in the input dataset."""
        missing = [col for col in self.req_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Schema Validation Error: Missing required columns: {missing}")
        logger.info("Schema validation passed successfully.")

    @staticmethod
    def parse_price_to_usd(price_val: Any) -> float:
        """Parses mixed-format price strings (SGD$, USD$, numbers) safely into float."""
        if pd.isna(price_val):
            return np.nan
        if isinstance(price_val, (int, float)):
            return float(price_val)
        try:
            val_str = str(price_val).replace('$', '').replace(',', '').strip()
            match = re.search(r"[-+]?\d*\.\d+|\d+", val_str)
            if match:
                res = float(match.group())
                if 'SGD' in str(price_val).upper():
                    res = res * 0.74 # Currency conversion rate
                return res
            return np.nan
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse price value '{price_val}': {e}")
            return np.nan

    def compute_stay_duration_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized computation of stay duration handling same-month and cross-month stays."""
        arr_m = df['arrival_month'].map(MONTH_MAP).fillna(1).astype(int)
        chk_m = df['checkout_month'].map(MONTH_MAP).fillna(1).astype(int)
        arr_d = pd.to_numeric(df['arrival_day'], errors='coerce').fillna(1).astype(int)
        chk_d = pd.to_numeric(df['checkout_day'], errors='coerce').fillna(1).astype(int)

        # Same month
        same_month_mask = (arr_m == chk_m)
        duration_same = chk_d - arr_d

        # Cross month (checkout month > arrival month)
        month_days = arr_m.map(DAYS_IN_MONTH)
        duration_cross = (month_days - arr_d) + chk_d

        # Year rollover (December -> January)
        year_rollover_mask = (chk_m < arr_m)
        duration_year = (31 - arr_d) + chk_d

        duration = np.where(same_month_mask, duration_same,
                   np.where(year_rollover_mask, duration_year, duration_cross))
        
        # Enforce minimum stay of 1 night
        return pd.Series(np.maximum(duration, 1), index=df.index)

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """Loads dataset, performs schema validation, vectorized cleaning, 
        imputation, and feature engineering.
        """
        logger.info(f"Loading raw dataset from {file_path}")
        try:
            df_raw = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to read input CSV file at {file_path}: {e}")
            raise

        # Normalize column aliases BEFORE validating schema
        df_raw = self.normalize_column_names(df_raw)
        self.validate_schema(df_raw)

        # 1. Clean Target
        df = df_raw.dropna(subset=[self.target_col]).copy().reset_index(drop=True)
        df[self.target_col] = df[self.target_col].astype(int)

        # 2. Vectorized Price Parsing
        if 'price_usd' not in df.columns:
            df['price_usd'] = df['price'].apply(self.parse_price_to_usd)

        # 3. Categorical Null Imputation
        df['room'] = df['room'].fillna('Unknown')

        # 4. Grouped Median Imputation for Price
        price_medians = df.groupby(['branch', 'room'])['price_usd'].transform('median')
        df['price_usd'] = df['price_usd'].fillna(price_medians).fillna(df['price_usd'].median())

        # 5. Vectorized Stay Duration Computation
        df['stay_duration'] = self.compute_stay_duration_vectorized(df)

        # 6. Interaction Feature Engineering
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

        logger.info(f"Data cleaning and feature engineering complete. Total rows: {len(df)}")
        return df

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs an sklearn ColumnTransformer for numerical scaling and categorical encoding."""
        num_cols = self.config['features']['numerical_features']
        cat_cols = self.config['features']['categorical_features']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        return preprocessor
