import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from src.utils.logger import setup_logger

logger = setup_logger("model_predictor")

class ChurnPredictor:
    """Inference engine for Customer Churn predictions using trained pipeline artifacts."""
    
    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "best_churn_model.joblib")
        self.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        self.meta_path = os.path.join(artifacts_dir, "model_metadata.json")
        
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.load_artifacts()

    def load_artifacts(self):
        """Loads trained model, preprocessor, and metadata from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path):
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            if os.path.exists(self.meta_path):
                with open(self.meta_path, "r") as f:
                    self.metadata = json.load(f)
            logger.info(f"Loaded churn prediction model: {self.metadata.get('best_model_name', 'Unknown')}")
        else:
            logger.warning(f"Model artifacts missing in '{self.artifacts_dir}'. Predictor uninitialized.")

    def predict_dataframe(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Runs predictions on input DataFrame, returning probabilities and binary churn labels."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model artifacts are not loaded. Train model first.")

        df = df_input.copy()
        
        # Transform features
        X_transformed = self.preprocessor.transform(df)
        
        # Probabilities & Predictions
        churn_probs = self.model.predict_proba(X_transformed)[:, 1] if hasattr(self.model, "predict_proba") else [0.5]*len(df)
        churn_preds = (churn_probs >= 0.5).astype(int)
        
        results = df_input.copy()
        results["churn_probability"] = np.round(churn_probs, 4)
        results["predicted_churn"] = churn_preds
        results["risk_level"] = pd.cut(
            results["churn_probability"],
            bins=[-0.01, 0.35, 0.70, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )
        return results

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs churn inference on a single customer dictionary."""
        df_single = pd.DataFrame([customer_data])
        df_res = self.predict_dataframe(df_single)
        
        row = df_res.iloc[0]
        return {
            "customer_id": customer_data.get("customer_id", "Unknown"),
            "churn_probability": float(row["churn_probability"]),
            "predicted_churn": int(row["predicted_churn"]),
            "risk_level": str(row["risk_level"]),
            "model_version": self.metadata.get("best_model_name", "RandomForest")
        }
