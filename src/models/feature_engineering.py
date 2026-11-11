import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

class FeatureEngineeringPipeline:
    """Prepares and scales feature matrices for customer churn modeling."""
    
    NUMERIC_FEATURES = [
        "recency_days", "frequency_orders", "monetary_value", 
        "avg_order_value", "total_items_bought", "cancellation_count",
        "tenure_months", "age", "support_tickets"
    ]
    
    CATEGORICAL_FEATURES = [
        "gender", "city", "subscription_type", "rfm_segment"
    ]

    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.preprocessor = None

    def build_preprocessor(self) -> ColumnTransformer:
        """Constructs Scikit-Learn ColumnTransformer for scaling and encoding."""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.NUMERIC_FEATURES),
                ("cat", categorical_transformer, self.CATEGORICAL_FEATURES)
            ]
        )
        return preprocessor

    def prepare_data(self, df_retention: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Cleans retention dataframe, fills missing values, fits feature preprocessor,
        and splits dataset into train and test sets.
        """
        logger.info(f"Preparing dataset for machine learning model training ({len(df_retention)} records)...")
        df = df_retention.copy()

        # Fill missing values
        for col in self.NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median() if not df[col].dropna().empty else 0)
            else:
                df[col] = 0.0

        for col in self.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str)
            else:
                df[col] = "Unknown"

        target_col = "churn_label"
        if target_col not in df.columns:
            if "churn" in df.columns:
                df[target_col] = df["churn"]
            else:
                df[target_col] = (df["recency_days"] > 90).astype(int)

        X = df[self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES]
        y = df[target_col].astype(int).values

        # Build and fit preprocessor
        self.preprocessor = self.build_preprocessor()
        X_transformed = self.preprocessor.fit_transform(X)

        # Retrieve feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.CATEGORICAL_FEATURES))
        feature_names = self.NUMERIC_FEATURES + cat_feature_names

        # Save preprocessor artifact
        preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.joblib")
        joblib.dump(self.preprocessor, preprocessor_path)
        logger.info(f"Saved feature preprocessor artifact to {preprocessor_path}")

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Dataset split complete: Train shape {X_train.shape}, Test shape {X_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names
