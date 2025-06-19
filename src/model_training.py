import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Training Random Forest Model"""
        print("=== Training Random Forest ===")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return rf_model, accuracy
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Training XGBoost Model"""
        print("=== Training XGBoost ===")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=6,
            learning_rate=0.1
        )
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"XGBoost Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return xgb_model, accuracy
    
    def train_catboost(self, X_train, y_train, X_test, y_test):
        """Training CatBoost Model"""
        print("=== Training CatBoost ===")
        
        cb_model = CatBoostClassifier(
            iterations=100,
            random_seed=42,
            depth=6,
            learning_rate=0.1,
            verbose=False
        )
        
        cb_model.fit(X_train, y_train)
        y_pred = cb_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['catboost'] = cb_model
        self.results['catboost'] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"CatBoost Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return cb_model, accuracy
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Training semua model dan bandingkan performa"""
        print("=== TRAINING ALL MODELS ===")
        
        # Train semua model
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_catboost(X_train, y_train, X_test, y_test)
        
        # Bandingkan hasil
        print("\n=== MODEL COMPARISON ===")
        for model_name, result in self.results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
            
        # Pilih model terbaik
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        best_model = self.models[best_model_name]
        
        print(f"\nBest Model: {best_model_name} with accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return best_model, best_model_name
    
    def save_models(self, save_dir="../models/"):
        """Simpan semua model yang sudah ditraining"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"Model {model_name} saved to {model_path}")
    
    def get_feature_importance(self, model_name, feature_names):
        """Dapatkan feature importance dari model"""
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n=== {model_name.upper()} FEATURE IMPORTANCE ===")
                print(importance_df)
                
                return importance_df
        return None

if __name__ == "__main__":
    # Test model training
    from data_preprocessing import DataPreprocessor
    
    # Load dan preprocess data
    preprocessor = DataPreprocessor("../data/raw/Training.csv")
    df = preprocessor.load_data()
    df = preprocessor.clean_data()
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Train models
    trainer = ModelTrainer()
    best_model, best_model_name = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    # Feature importance
    trainer.get_feature_importance(best_model_name, X.columns)