import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTraining:
    """
    ModelTraining handles stratified data splitting, candidate model training,
    hyperparameter optimization, metric evaluation, and artifact persistence.
    """
    def __init__(self, config: dict, preprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def split_data(self, df: pd.DataFrame):
        target_col = self.config['target_column']
        id_col = self.config.get('id_column', 'booking_id')
        
        X = df.drop(columns=[id_col, target_col], errors='ignore')
        y = df[target_col]

        test_sz = self.config['split_ratios']['test_size']
        val_sz = self.config['split_ratios']['val_size']
        seed = self.config['split_ratios']['random_state']

        # Stratified train / val / test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_sz, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_sz, random_state=seed, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def evaluate_model(self, model, X_data, y_true):
        y_pred = model.predict(X_data)
        y_prob = model.predict_proba(X_data)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        return {
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'F1-Score': float(f1_score(y_true, y_pred, zero_division=0)),
            'ROC-AUC': float(roc_auc_score(y_true, y_prob))
        }

    def train_baselines(self, X_train, y_train, X_val, y_val):
        candidates = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNN Classifier (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
        }

        results = {}
        for name, clf in candidates.items():
            pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', clf)])
            pipe.fit(X_train, y_train)
            metrics = self.evaluate_model(pipe, X_val, y_val)
            results[name] = metrics
        
        return results

    def tune_and_save(self, X_train, y_train, X_test, y_test):
        dt_clf = DecisionTreeClassifier(random_state=42)
        pipe = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', dt_clf)])

        param_grid = {
            'classifier__max_depth': [5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        test_metrics = self.evaluate_model(best_model, X_test, y_test)

        # Persist model artifact
        save_path = self.config.get('model_save_path', 'models/best_no_show_model.joblib')
        joblib.dump(best_model, save_path)

        return best_model, grid_search.best_params_, test_metrics
