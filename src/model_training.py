import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from src.utils import setup_logger

logger = setup_logger()

class ModelTraining:
    """Handles stratified dataset splitting, baseline candidate training, 
    threshold optimization, evaluation, and pipeline persistence.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        self.config = config
        self.preprocessor = preprocessor
        self.save_dir = config['models']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Performs a 3-way stratified train/validation/test split."""
        target_col = self.config['data']['target_column']
        num_cols = self.config['features']['numerical_features']
        cat_cols = self.config['features']['categorical_features']
        bin_cols = self.config['features']['binary_features']
        
        feature_cols = num_cols + cat_cols + bin_cols
        X = df[feature_cols]
        y = df[target_col]

        test_sz = self.config['split']['test_size']
        val_sz = self.config['split']['val_size']
        seed = self.config['split']['random_state']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_sz, random_state=seed, stratify=y if self.config['split']['stratify'] else None
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_sz, random_state=seed, stratify=y_temp if self.config['split']['stratify'] else None
        )

        logger.info(f"Dataset Split Ratios -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_candidate_models(self) -> Dict[str, Any]:
        """Defines a diverse set of baseline and ensemble classification models."""
        seed = self.config['split']['random_state']
        return {
            'Logistic Regression (Balanced)': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed),
            'KNN Classifier (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'Decision Tree (Balanced)': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=seed),
            'Random Forest (Balanced)': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=seed, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=seed)
        }

    def evaluate_model(self, model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
        """Computes comprehensive evaluation metrics including PR-AUC, Confusion Matrix, and Specificity."""
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_eval)[:, 1]
            y_preds = (y_probs >= threshold).astype(int)
            roc_auc = roc_auc_score(y_eval, y_probs)
            precision_arr, recall_arr, _ = precision_recall_curve(y_eval, y_probs)
            pr_auc = auc(recall_arr, precision_arr)
        else:
            y_preds = model.predict(X_eval)
            y_probs = y_preds
            roc_auc = roc_auc_score(y_eval, y_preds)
            pr_auc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_eval, y_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            'accuracy': float(accuracy_score(y_eval, y_preds)),
            'precision': float(precision_score(y_eval, y_preds, zero_division=0)),
            'recall': float(recall_score(y_eval, y_preds, zero_division=0)),
            'f1_score': float(f1_score(y_eval, y_preds, zero_division=0)),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'specificity': float(specificity),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'applied_threshold': float(threshold)
        }
        return metrics

    def optimize_threshold(self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Finds the optimal classification probability threshold on validation set maximizing F1-score."""
        if not hasattr(model, "predict_proba"):
            return 0.5
        y_probs = model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = 0.0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh

        logger.info(f"Optimal Validation Threshold: {best_thresh:.2f} (Val F1 Score: {best_f1:.4f})")
        return float(best_thresh)

    def train_and_evaluate_baselines(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[str, Pipeline, Dict[str, Any]]:
        """Trains candidate classifiers, evaluates on validation set, and selects the best performer."""
        candidates = self.get_candidate_models()
        best_score = -1.0
        best_model_name = ""
        best_pipeline = None
        results = {}

        for name, clf in candidates.items():
            logger.info(f"Training Candidate Model: {name}")
            pipe = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', clf)
            ])
            pipe.fit(X_train, y_train)
            metrics = self.evaluate_model(pipe, X_val, y_val)
            results[name] = metrics
            logger.info(f"{name} -> Val ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1_score']:.4f}")

            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = name
                best_pipeline = pipe

        logger.info(f"Selected Best Candidate: {best_model_name} with Val ROC-AUC: {best_score:.4f}")
        return best_model_name, best_pipeline, results

    def tune_and_save(self, best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Pipeline:
        """Performs hyperparameter tuning via GridSearchCV, optimizes probability threshold, and persists artifacts."""
        clf = best_pipeline.named_steps['classifier']
        
        if isinstance(clf, KNeighborsClassifier):
            param_grid = {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}
        elif isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier)):
            param_grid = {'classifier__n_estimators': [100, 150], 'classifier__max_depth': [5, 8, 12]}
        elif isinstance(clf, DecisionTreeClassifier):
            param_grid = {'classifier__max_depth': [5, 8, 12], 'classifier__min_samples_split': [2, 5, 10]}
        elif isinstance(clf, LogisticRegression):
            param_grid = {'classifier__C': [0.1, 1.0, 10.0]}
        else:
            param_grid = {}

        logger.info("Executing GridSearchCV Hyperparameter Tuning...")
        if param_grid:
            grid = GridSearchCV(best_pipeline, param_grid, cv=self.config['hyperparameters']['cv_folds'], scoring=self.config['hyperparameters']['scoring'], n_jobs=-1)
            grid.fit(X_train, y_train)
            tuned_pipeline = grid.best_estimator_
        else:
            tuned_pipeline = best_pipeline
            tuned_pipeline.fit(X_train, y_train)

        # Threshold Optimization
        optimal_thresh = self.optimize_threshold(tuned_pipeline, X_val, y_val)

        # Final Evaluation on Test Set
        test_metrics = self.evaluate_model(tuned_pipeline, X_test, y_test, threshold=optimal_thresh)
        logger.info(f"Final Test Set Metrics (Threshold {optimal_thresh:.2f}): {json.dumps(test_metrics, indent=2)}")

        # Artifact Persistence
        model_path = os.path.join(self.save_dir, self.config['models']['model_filename'])
        metrics_path = os.path.join(self.save_dir, self.config['models']['metrics_filename'])
        thresh_path = os.path.join(self.save_dir, self.config['models']['threshold_filename'])

        joblib.dump(tuned_pipeline, model_path)
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=4)
        with open(thresh_path, 'w') as f:
            json.dump({'optimal_threshold': optimal_thresh}, f, indent=4)

        logger.info(f"Successfully saved model pipeline to {model_path}")
        return tuned_pipeline
