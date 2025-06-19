import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import logging
import os

logger = logging.getLogger(__name__)

class RecommendationSystem:
    def __init__(self):
        self.intervention_rules = self._create_intervention_rules()
        self.kmeans_model = None
        
    def _create_intervention_rules(self):
        """Buat aturan intervensi berdasarkan faktor risiko"""
        rules = {
            'low_study_time': {
                'condition': lambda x: x.get('StudyTimeWeekly', x.get('study_time_weekly', 0)) < 0,  # Normalized value
                'recommendations': [
                    "Buat jadwal belajar terstruktur minimal 2 jam per hari",
                    "Gunakan teknik Pomodoro untuk fokus belajar",
                    "Cari ruang belajar yang tenang dan bebas distraksi"
                ]
            },
            'high_absences': {
                'condition': lambda x: x.get('Absences', x.get('absences', 0)) > 0.5,  # Normalized value
                'recommendations': [
                    "Konsultasi dengan konselor sekolah mengenai absensi",
                    "Identifikasi penyebab sering tidak masuk sekolah",
                    "Buat komitmen untuk hadir sekolah secara teratur"
                ]
            },
            'low_parental_support': {
                'condition': lambda x: x.get('ParentalSupport', x.get('parental_support', 2)) <= 1,
                'recommendations': [
                    "Libatkan orang tua dalam diskusi prestasi akademik",
                    "Minta bantuan orang tua untuk memonitor jadwal belajar",
                    "Ajak orang tua menghadiri pertemuan dengan guru"
                ]
            },
            'no_tutoring': {
                'condition': lambda x: x.get('Tutoring', x.get('tutoring', 0)) == 0,
                'recommendations': [
                    "Pertimbangkan mengikuti les tambahan untuk mata pelajaran sulit",
                    "Manfaatkan tutor sebaya atau study group",
                    "Ikuti kelas remedial yang disediakan sekolah"
                ]
            },
            'low_extracurricular': {
                'condition': lambda x: (
                    x.get('Extracurricular', 0) + 
                    x.get('Sports', 0) + 
                    x.get('Music', 0) + 
                    x.get('Volunteering', 0)
                ) == 0,
                'recommendations': [
                    "Ikuti minimal satu kegiatan ekstrakurikuler",
                    "Pilih kegiatan yang sesuai dengan minat dan bakat",
                    "Kegiatan ekstrakurikuler dapat meningkatkan soft skills"
                ]
            },
            'low_gpa': {
                'condition': lambda x: x.get('GPA', x.get('gpa', 0)) < 0,  # Normalized negative value
                'recommendations': [
                    "Konsultasi rutin dengan guru mata pelajaran",
                    "Buat target perbaikan nilai untuk setiap mata pelajaran",
                    "Ikuti program remedial untuk mata pelajaran bermasalah"
                ]
            },
            'general_academic': {
                'condition': lambda x: True,  # Selalu applicable
                'recommendations': [
                    "Konsultasi rutin dengan guru mata pelajaran",
                    "Manfaatkan perpustakaan sekolah untuk belajar",
                    "Bergabung dengan kelompok belajar",
                    "Tetapkan target nilai untuk setiap mata pelajaran"
                ]
            }
        }
        return rules
    
    def analyze_risk_factors(self, student_profile):
        """Analisis faktor risiko dari profil siswa"""
        risk_factors = []
        
        # Konversi student_profile ke dict jika berupa Series atau array
        if hasattr(student_profile, 'to_dict'):
            profile_dict = student_profile.to_dict()
        elif isinstance(student_profile, (list, np.ndarray)):
            # Jika array, buat mapping dengan nama kolom umum
            feature_names = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
                           'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
                           'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA']
            profile_dict = dict(zip(feature_names[:len(student_profile)], student_profile))
        else:
            profile_dict = student_profile
        
        for factor_name, rule in self.intervention_rules.items():
            try:
                if rule['condition'](profile_dict):
                    risk_factors.append(factor_name)
            except (KeyError, TypeError) as e:
                logger.warning(f"Error evaluating condition for {factor_name}: {e}")
                continue
        
        return risk_factors
    
    def generate_student_recommendations(self, student_profile, risk_level):
        """Generate rekomendasi berdasarkan profil siswa individual"""
        risk_factors = self.analyze_risk_factors(student_profile)
        
        recommendations = []
        
        # Tambahkan rekomendasi berdasarkan faktor risiko
        for factor in risk_factors:
            if factor in self.intervention_rules:
                recommendations.extend(self.intervention_rules[factor]['recommendations'])
        
        # Tambahkan rekomendasi khusus berdasarkan tingkat risiko
        if risk_level == 'Sangat Tinggi':
            recommendations.extend([
                "Segera konsultasi dengan konselor sekolah",
                "Pertimbangkan program remedial intensif",
                "Buat rencana pembelajaran individual dengan guru"
            ])
        elif risk_level == 'Tinggi':
            recommendations.extend([
                "Tingkatkan waktu belajar dan fokus pada mata pelajaran bermasalah",
                "Minta bantuan guru untuk penjelasan tambahan"
            ])
        
        # Hapus duplikasi
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def generate_recommendations(self, model_data, processed_data):
        """Method utama yang dipanggil dari main.py"""
        try:
            logger.info("Generating recommendations for students...")
            
            # Extract data
            model = model_data['model']
            X_test = processed_data['X_test']
            
            # Prediksi untuk sample data
            sample_data = X_test.head(10)  # Ambil 10 siswa pertama
            # Di recommendation_system.py, line 107
            predictions = model.predict(sample_data)
            pred_grade_numeric = predictions[i]

            # ADD THIS MAPPING:
            grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
            pred_grade = grade_mapping[pred_grade_numeric]
            
            recommendations_list = []
            
            for i, (_, student_row) in enumerate(sample_data.iterrows()):
                pred_grade = predictions[i]
                risk_level = grade_mapping.get(pred_grade, 'Sedang')
                
                # Generate recommendations untuk siswa ini
                student_recommendations = self.generate_student_recommendations(
                    student_row, risk_level
                )
                
                student_plan = {
                    'student_id': sample_data.index[i],
                    'predicted_grade': pred_grade,
                    'risk_level': risk_level,
                    'recommendations': student_recommendations[:8],  # Limit to 8
                    'priority': 'URGENT' if risk_level in ['Sangat Tinggi', 'Tinggi'] else 'MODERATE'
                }
                
                recommendations_list.append(student_plan)
            
            logger.info(f"Generated recommendations for {len(recommendations_list)} students")
            return recommendations_list
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def save_recommendations(self, recommendations, output_dir):
        """Simpan rekomendasi ke file"""
        import os
        
        try:
            # Buat direktori jika belum ada
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert numpy types to native Python types
            clean_recommendations = []
            for rec in recommendations:
                clean_rec = {}
                for key, value in rec.items():
                    if isinstance(value, (np.integer, np.floating)):
                        clean_rec[key] = value.item()
                    elif isinstance(value, np.ndarray):
                        clean_rec[key] = value.tolist()
                    else:
                        clean_rec[key] = value
                clean_recommendations.append(clean_rec)
            
            # Simpan ke JSON
            output_file = os.path.join(output_dir, "student_recommendations.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_recommendations, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Recommendations saved to {output_file}")
            
            # Buat summary
            self._create_summary_report(clean_recommendations, output_dir)
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
    
    def _create_summary_report(self, recommendations, output_dir):
        """Buat laporan ringkasan"""
        try:
            # Hitung statistik
            total_students = len(recommendations)
            risk_distribution = {}
            priority_distribution = {}
            
            for rec in recommendations:
                risk_level = rec.get('risk_level', 'Unknown')
                priority = rec.get('priority', 'Unknown')
                
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
            
            summary = {
                'total_students_analyzed': total_students,
                'risk_level_distribution': risk_distribution,
                'priority_distribution': priority_distribution,
                'high_risk_students': [
                    rec['student_id'] for rec in recommendations 
                    if rec.get('risk_level') in ['Sangat Tinggi', 'Tinggi']
                ]
            }
            
            # Simpan summary
            summary_file = os.path.join(output_dir, "recommendations_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error creating summary report: {e}")
    
    def cluster_students(self, X_data, n_clusters=5):
        """Clustering siswa berdasarkan profil akademik"""
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans_model.fit_predict(X_data)
        
        # Analisis karakteristik setiap cluster
        cluster_profiles = {}
        for i in range(n_clusters):
            cluster_mask = clusters == i
            cluster_data = X_data[cluster_mask]
            
            cluster_profiles[i] = {
                'size': len(cluster_data),
                'mean_profile': cluster_data.mean(axis=0),
                'std_profile': cluster_data.std(axis=0)
            }
        
        return clusters, cluster_profiles

if __name__ == "__main__":
    # Test recommendation system
    print("Testing Recommendation System...")
    
    # Create sample data
    sample_profile = {
        'StudyTimeWeekly': -0.5,
        'Absences': 0.3,
        'ParentalSupport': 1,
        'Tutoring': 0,
        'Extracurricular': 0,
        'Sports': 0,
        'Music': 0,
        'Volunteering': 0,
        'GPA': -0.2
    }
    
    rec_system = RecommendationSystem()
    recommendations = rec_system.generate_student_recommendations(sample_profile, 'Tinggi')
    
    print("Sample recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. {rec}")