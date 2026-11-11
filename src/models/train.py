import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.utils.logger import setup_logger

logger = setup_logger("model_trainer")

class ChurnModelTrainer:
    """Trains, tracks, and selects best ML models for Customer Churn prediction."""
    
    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.best_model = None
        self.best_model_name = ""
        self.best_f1_score = -1.0
        self.results_summary = {}

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculates evaluation metrics for binary classification."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
        }
        return metrics

    def train_and_evaluate_all(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_names: list) -> Dict[str, Any]:
        """Trains Logistic Regression, Random Forest, and XGBoost/GBDT models."""
        logger.info("Initializing model training pipeline...")
        
        models_to_train = {
            "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        if HAS_XGBOOST:
            models_to_train["XGBoost"] = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        else:
            models_to_train["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)

        if HAS_MLFLOW:
            try:
                mlflow.set_experiment("Customer_Churn_Prediction")
            except Exception as e:
                logger.warning(f"MLflow set_experiment warning: {e}")

        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test)
            logger.info(f"{model_name} Results: F1={metrics['f1_score']}, ROC-AUC={metrics['roc_auc']}, Precision={metrics['precision']}, Recall={metrics['recall']}")

            self.results_summary[model_name] = metrics

            # Log to MLflow if enabled
            if HAS_MLFLOW:
                try:
                    with mlflow.start_run(run_name=model_name):
                        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(model, f"model_{model_name.lower()}")
                except Exception as e:
                    logger.warning(f"Failed logging {model_name} to MLflow: {e}")

            # Check if this model is the best performer
            if metrics["f1_score"] > self.best_f1_score:
                self.best_f1_score = metrics["f1_score"]
                self.best_model = model
                self.best_model_name = model_name

        logger.info(f"Best Model Selected: '{self.best_model_name}' with F1-Score: {self.best_f1_score}")

        # Save Best Model Artifacts locally
        best_model_path = os.path.join(self.artifacts_dir, "best_churn_model.joblib")
        joblib.dump(self.best_model, best_model_path)
        
        meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")
        metadata = {
            "best_model_name": self.best_model_name,
            "best_f1_score": self.best_f1_score,
            "all_metrics": self.results_summary,
            "feature_names": feature_names,
            "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved best model artifact to '{best_model_path}' and metadata to '{meta_path}'.")
        return metadata
