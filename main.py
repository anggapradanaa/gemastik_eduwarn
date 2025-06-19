#!/usr/bin/env python3
"""
EDUWARN Project - Main Execution Script
=======================================

Sistem peringatan dini pendidikan yang mengintegrasikan:
- Preprocessing data
- Pelatihan model
- Explainable AI
- Sistem rekomendasi

Author: EDUWARN Team
Date: 2025
"""

import os
import numpy as np
import sys
import argparse
import logging
from pathlib import Path

# Menambahkan src ke Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modul-modul dari src
try:
    from src.data_preprocessing import DataPreprocessor
    from src.model_training import ModelTrainer
    from src.explainable_ai import ExplainableAI
    from src.recommendation_system import RecommendationSystem
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Pastikan semua file di folder 'src' tersedia dan dapat diimport")
    sys.exit(1)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eduwarn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EDUWARNPipeline:
    """
    Pipeline utama untuk sistem EDUWARN
    """
    
    def __init__(self, data_path="data/raw/Training.csv"):
        """
        Inisialisasi pipeline EDUWARN
        
        Args:
            data_path (str): Path ke file data training
        """
        self.data_path = data_path
        self.preprocessor = None
        self.trainer = None
        self.explainer = None
        self.recommender = None
        
        # Membuat direktori output jika belum ada
        self._create_directories()
        
    def _create_directories(self):
        """Membuat direktori yang diperlukan"""
        directories = ['models', 'results', 'results/plots', 'results/reports']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def run_preprocessing(self):
        """Menjalankan tahap preprocessing data"""
        logger.info("Memulai preprocessing data...")
        try:
            self.preprocessor = DataPreprocessor(self.data_path)  # ✅ Berikan data_path
            
            # Jalankan seluruh pipeline preprocessing
            df = self.preprocessor.load_data()
            if df is None:
                raise ValueError("Gagal memuat data")
                
            self.preprocessor.explore_data()
            self.preprocessor.clean_data()
            X, y = self.preprocessor.prepare_features()
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
            
            # Return data yang sudah diproses
            processed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X': X,
                'y': y
            }
            
            logger.info("Preprocessing data selesai")
            return processed_data
        except Exception as e:
            logger.error(f"Error dalam preprocessing: {e}")
            raise
            
    def run_training(self, processed_data):
        """Menjalankan tahap pelatihan model"""
        logger.info("Memulai pelatihan model...")
        try:
            self.trainer = ModelTrainer()
            
            # Extract data dari processed_data
            X_train = processed_data['X_train']
            y_train = processed_data['y_train']
            X_test = processed_data['X_test']
            y_test = processed_data['y_test']
            
            # Train semua model dan pilih yang terbaik
            best_model, best_model_name = self.trainer.train_all_models(X_train, y_train, X_test, y_test)
            
            # Simpan model
            self.trainer.save_models("models/")
            logger.info(f"Model terbaik ({best_model_name}) berhasil disimpan")
            
            # Tampilkan feature importance
            if hasattr(processed_data['X_train'], 'columns'):
                feature_names = processed_data['X_train'].columns
                self.trainer.get_feature_importance(best_model_name, feature_names)
            
            return {
                'model': best_model,
                'model_name': best_model_name,
                'trainer': self.trainer
            }
            
        except Exception as e:
            logger.error(f"Error dalam training: {e}")
            raise
            
    def run_explainable_ai(self, model_data, processed_data):
        """Menjalankan analisis Explainable AI"""
        logger.info("Memulai analisis Explainable AI...")
        try:
            self.explainer = ExplainableAI()
            
            # Set model ke explainer
            self.explainer.model = model_data['model']
            
            # Extract data
            X_train = processed_data['X_train']
            X_test = processed_data['X_test']
            
            # Setup explainers
            logger.info("Setting up SHAP explainer...")
            self.explainer.setup_shap_explainer(X_train)
            
            logger.info("Setting up LIME explainer...")
            class_names = ['A', 'B', 'C', 'D', 'F']
            self.explainer.setup_lime_explainer(X_train, X_train.columns.tolist(), class_names)
            
            # Create explanations
            explanations = {}
            
            # SHAP explanations dengan sample kecil
            logger.info("Generating SHAP explanations...")
            try:
                sample_data = X_test.head(5)  # Kurangi dari 10 ke 5
                logger.info(f"Processing SHAP for {len(sample_data)} samples...")
                
                # Calculate SHAP values tanpa plotting dulu
                if self.explainer.explainer_shap is not None:
                    shap_values = self.explainer.explainer_shap.shap_values(sample_data)
                    explanations['shap_values'] = "Generated successfully"
                    logger.info("SHAP values calculated successfully")
                else:
                    explanations['shap_values'] = "Failed - no explainer"
                    logger.warning("SHAP explainer not available")
                    
            except Exception as e:
                logger.error(f"SHAP analysis failed: {e}")
                explanations['shap_values'] = f"Failed: {str(e)}"
            
            # Batch risk analysis
            logger.info("Analyzing high-risk students...")
            try:
                risk_profiles = self.explainer.batch_explain(X_test.head(50), top_n=3)  # Kurangi sample
                explanations['risk_profiles'] = risk_profiles
                logger.info(f"Risk analysis completed for {len(risk_profiles) if risk_profiles else 0} students")
            except Exception as e:
                logger.error(f"Risk analysis failed: {e}")
                explanations['risk_profiles'] = []
            
            # LIME explanation - skip jika terlalu lama
            logger.info("Generating LIME explanation...")
            try:
                # Hanya untuk 1 instance
                lime_exp = self.explainer.explain_with_lime(X_test, instance_idx=0)
                explanations['lime_explanation'] = "Generated" if lime_exp else "Failed"
                logger.info("LIME explanation completed")
            except Exception as e:
                logger.error(f"LIME analysis failed: {e}")
                explanations['lime_explanation'] = f"Failed: {str(e)}"
            
            # Save summary report
            logger.info("Saving explanation report...")
            self._save_explanation_report(explanations, "results/reports/")
            
            logger.info("Analisis Explainable AI selesai")
            return explanations
            
        except Exception as e:
            logger.error(f"Error dalam Explainable AI: {e}")
            # Return partial results instead of raising
            return {'error': str(e), 'status': 'failed'}

    def _save_explanation_report(self, explanations, report_dir):
        """Simpan laporan penjelasan ke file"""
        import os
        import json
        import numpy as np
        from datetime import datetime
        
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        try:
            # Create summary report
            report = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'Explainable AI Analysis',
                'summary': {
                    'shap_analysis': explanations.get('shap_values', 'Not generated'),
                    'risk_profiles': len(explanations.get('risk_profiles', [])),
                    'lime_analysis': explanations.get('lime_explanation', 'Not generated')
                }
            }
            
            # Save risk profiles in readable format
            if explanations.get('risk_profiles'):
                report['high_risk_students'] = []
                for i, profile in enumerate(explanations['risk_profiles']):
                    if profile:
                        # Convert numpy types to native Python types
                        confidence_value = profile.get('confidence', 0)
                        if isinstance(confidence_value, (np.integer, np.floating)):
                            confidence_value = float(confidence_value)
                        
                        student_summary = {
                            'student_id': i + 1,
                            'predicted_grade': str(profile.get('predicted_grade', 'Unknown')),
                            'risk_level': str(profile.get('risk_level', 'Unknown')),
                            'confidence': confidence_value
                        }
                        report['high_risk_students'].append(student_summary)
            
            # Save to JSON file with proper handling of numpy types
            report_path = os.path.join(report_dir, 'explainable_ai_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            logger.info(f"Explanation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving explanation report: {e}")

    def _json_serializer(self, obj):
        """Custom JSON serializer untuk handle numpy types"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj)
            
    def run_recommendation_system(self, model, data):
        """Menjalankan sistem rekomendasi"""
        logger.info("Memulai sistem rekomendasi...")
        try:
            self.recommender = RecommendationSystem()
            recommendations = self.recommender.generate_recommendations(model, data)
            
            # Simpan rekomendasi
            self.recommender.save_recommendations(recommendations, "results/reports/")
            logger.info("Sistem rekomendasi selesai")
            return recommendations
        except Exception as e:
            logger.error(f"Error dalam recommendation system: {e}")
            raise
            
    def run_full_pipeline(self):
        """Menjalankan seluruh pipeline EDUWARN"""
        logger.info("=== MEMULAI EDUWARN PIPELINE ===")
        
        try:
            # 1. Preprocessing
            processed_data = self.run_preprocessing()
            
            # 2. Model Training
            trained_model_data = self.run_training(processed_data)
            
            # 3. Explainable AI
            explanations = self.run_explainable_ai(trained_model_data, processed_data)
            
            # 4. Recommendation System
            recommendations = self.run_recommendation_system(trained_model_data, processed_data)
            
            logger.info("=== EDUWARN PIPELINE SELESAI ===")
            
            return {
                'model': trained_model_data['model'],
                'model_name': trained_model_data['model_name'],
                'data': processed_data,
                'explanations': explanations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Pipeline gagal: {e}")
            raise

def main():
    """Fungsi utama"""
    parser = argparse.ArgumentParser(description='EDUWARN - Sistem Peringatan Dini Pendidikan')
    parser.add_argument('--data', default='data/raw/Training.csv', 
                       help='Path ke file data training')
    parser.add_argument('--mode', choices=['full', 'preprocess', 'train', 'explain', 'recommend'], 
                       default='full', help='Mode eksekusi')
    parser.add_argument('--model-path', help='Path ke model yang sudah ditraining (untuk mode selain full/train)')
    
    args = parser.parse_args()
    
    # Validasi file data
    if not os.path.exists(args.data):
        logger.error(f"File data tidak ditemukan: {args.data}")
        sys.exit(1)
        
    # Inisialisasi pipeline
    pipeline = EDUWARNPipeline(args.data)
    
    try:
        if args.mode == 'full':
            # Jalankan seluruh pipeline
            results = pipeline.run_full_pipeline()
            print("\n=== HASIL PIPELINE ===")
            print(f"Model: {type(results['model']).__name__}")
            print(f"Data shape: {results['data'].shape if hasattr(results['data'], 'shape') else 'N/A'}")
            print("Explanations dan Recommendations tersimpan di folder results/")
            
        elif args.mode == 'preprocess':
            processed_data = pipeline.run_preprocessing()
            print(f"Data berhasil dipreprocess. Shape: {processed_data.shape if hasattr(processed_data, 'shape') else 'N/A'}")
            
        elif args.mode == 'train':
            processed_data = pipeline.run_preprocessing()
            model = pipeline.run_training(processed_data)
            print(f"Model berhasil dilatih: {type(model).__name__}")
            
        elif args.mode == 'explain':
            if not args.model_path or not os.path.exists(args.model_path):
                logger.error("Model path diperlukan dan harus valid untuk mode explain")
                sys.exit(1)
            # Load model dan jalankan explanation
            # (implementasi tergantung pada struktur ModelTrainer)
            print("Mode explain - memerlukan implementasi load model")
            
        elif args.mode == 'recommend':
            if not args.model_path or not os.path.exists(args.model_path):
                logger.error("Model path diperlukan dan harus valid untuk mode recommend")
                sys.exit(1)
            # Load model dan jalankan recommendation
            print("Mode recommend - memerlukan implementasi load model")
            
    except KeyboardInterrupt:
        logger.info("Proses dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error dalam eksekusi: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()