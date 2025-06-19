import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import joblib
import os

class ExplainableAI:
    def __init__(self, model_path=None):
        self.model = None
        self.explainer_shap = None
        self.explainer_lime = None
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
    
    def load_model(self, model_path):
        """Load model yang sudah disimpan"""
        try:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def setup_shap_explainer(self, X_train):
        """Setup SHAP explainer"""
        try:
            # Untuk tree-based models
            if hasattr(self.model, 'tree_'):
                self.explainer_shap = shap.TreeExplainer(self.model)
            else:
                # Untuk model lainnya
                self.explainer_shap = shap.Explainer(self.model, X_train)
            print("SHAP explainer setup complete")
        except Exception as e:
            print(f"Error setting up SHAP: {e}")
    
    def setup_lime_explainer(self, X_train, feature_names, class_names):
        """Setup LIME explainer"""
        try:
            self.explainer_lime = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification'
            )
            print("LIME explainer setup complete")
        except Exception as e:
            print(f"Error setting up LIME: {e}")
    
    def explain_with_shap(self, X_sample, save_plot=True, plot_dir="../results/plots/"):
        """Buat penjelasan menggunakan SHAP"""
        if self.explainer_shap is None:
            print("SHAP explainer not setup. Please run setup_shap_explainer first.")
            return None
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer_shap.shap_values(X_sample)
            
            # Jika multiclass, ambil class pertama untuk visualisasi
            if isinstance(shap_values, list):
                shap_values_plot = shap_values[0]
            else:
                shap_values_plot = shap_values
            
            # Create summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", show=False)
            
            if save_plot:
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(os.path.join(plot_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
                print(f"SHAP plot saved to {plot_dir}")
            
            plt.show()
            
            return shap_values
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
            return None
    
    def explain_with_lime(self, X_sample, instance_idx=0):
        """Buat penjelasan menggunakan LIME untuk satu instance"""
        if self.explainer_lime is None:
            print("LIME explainer not setup. Please run setup_lime_explainer first.")
            return None
        
        try:
            # Explain instance
            exp = self.explainer_lime.explain_instance(
                X_sample.iloc[instance_idx].values,
                self.model.predict_proba,
                num_features=len(X_sample.columns)
            )
            
            print(f"=== LIME Explanation for Instance {instance_idx} ===")
            for feature, weight in exp.as_list():
                print(f"{feature}: {weight:.4f}")
            
            # Show plot
            exp.show_in_notebook(show_table=True)
            
            return exp
        except Exception as e:
            print(f"Error in LIME explanation: {e}")
            return None
    
    def create_risk_profile(self, X_sample, feature_names, instance_idx=0):
        """Buat profil risiko untuk satu siswa"""
        if self.model is None:
            print("Model not loaded")
            return None
        
        try:
            # Prediksi
            prediction = self.model.predict(X_sample.iloc[[instance_idx]])
            prediction_proba = self.model.predict_proba(X_sample.iloc[[instance_idx]])
            
            # Grade mapping
            grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            risk_mapping = {0: 'Rendah', 1: 'Rendah', 2: 'Sedang', 3: 'Tinggi', 4: 'Sangat Tinggi'}
            
            predicted_grade = grade_mapping[prediction[0]]
            risk_level = risk_mapping[prediction[0]]
            confidence = np.max(prediction_proba)
            
            print(f"=== PROFIL RISIKO SISWA ===")
            print(f"Prediksi Grade: {predicted_grade}")
            print(f"Tingkat Risiko: {risk_level}")
            print(f"Confidence: {confidence:.2%}")
            
            # Feature values untuk siswa ini
            print(f"\n=== PROFIL SISWA ===")
            student_data = X_sample.iloc[instance_idx]
            for feature, value in zip(feature_names, student_data.values):
                print(f"{feature}: {value:.2f}")
            
            return {
                'predicted_grade': predicted_grade,
                'risk_level': risk_level,
                'confidence': confidence,
                'student_profile': student_data.to_dict()
            }
            
        except Exception as e:
            print(f"Error creating risk profile: {e}")
            return None
    
    def batch_explain(self, X_sample, top_n=5):
        """Buat penjelasan untuk beberapa siswa dengan risiko tertinggi"""
        if self.model is None:
            print("Model not loaded")
            return None
        
        try:
            # Prediksi untuk semua siswa
            predictions = self.model.predict(X_sample)
            predictions_proba = self.model.predict_proba(X_sample)
            
            # Cari siswa dengan risiko tertinggi (grade D dan F)
            high_risk_indices = np.where((predictions == 3) | (predictions == 4))[0]
            
            if len(high_risk_indices) == 0:
                print("Tidak ada siswa dengan risiko tinggi ditemukan")
                return None
            
            # Ambil top N siswa dengan confidence tertinggi untuk risiko tinggi
            high_risk_confidence = predictions_proba[high_risk_indices].max(axis=1)
            top_indices = high_risk_indices[np.argsort(high_risk_confidence)[-top_n:]]
            
            print(f"=== TOP {len(top_indices)} SISWA BERISIKO TINGGI ===")
            
            risk_profiles = []
            for i, idx in enumerate(top_indices):
                print(f"\n--- Siswa #{idx} ---")
                profile = self.create_risk_profile(X_sample, X_sample.columns, idx)
                risk_profiles.append(profile)
            
            return risk_profiles
            
        except Exception as e:
            print(f"Error in batch explanation: {e}")
            return None

if __name__ == "__main__":
    # Test explainable AI
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor("../data/raw/Training.csv")
    df = preprocessor.load_data()
    df = preprocessor.clean_data()
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Load model (asumsi best model adalah random forest)
    explainer = ExplainableAI("../models/random_forest_model.pkl")
    
    # Setup explainers
    explainer.setup_shap_explainer(X_train)
    
    class_names = ['A', 'B', 'C', 'D', 'F']
    explainer.setup_lime_explainer(X_train, X_train.columns.tolist(), class_names)
    
    # Create explanations
    explainer.explain_with_shap(X_test.head(10))
    explainer.batch_explain(X_test.head(50))