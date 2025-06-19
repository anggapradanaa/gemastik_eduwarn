import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load dataset dari file CSV"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data berhasil dimuat: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data(self):
        """Eksplorasi dasar dataset"""
        print("=== DATA EXPLORATION ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nInfo:")
        print(self.df.info())
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        print(f"\nTarget distribution:")
        print(self.df['GradeClass'].value_counts())
        
    def clean_data(self):
        """Pembersihan data dasar"""
        # Handle missing values
        if self.df.isnull().sum().sum() > 0:
            self.df = self.df.fillna(self.df.median(numeric_only=True))
        
        # Remove outliers menggunakan IQR method untuk numerical columns
        numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
        
        print(f"Data setelah cleaning: {self.df.shape}")
        return self.df
    
    def prepare_features(self):
        """Persiapan features untuk modeling"""
        # Pisahkan features dan target
        feature_cols = ['Age', 'Gender', 'Ethnicity', 'ParentalEducation', 
                    'StudyTimeWeekly', 'Absences', 'Tutoring', 
                    'ParentalSupport', 'Extracurricular', 'Sports', 
                    'Music', 'Volunteering', 'GPA']
        
        X = self.df[feature_cols].copy()  # Tambahkan .copy() untuk menghindari warning
        y = self.df['GradeClass']
        
        # Normalisasi numerical features
        numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data untuk training dan testing"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor("D:\Perkuliahan\GEMASTIK\EDUWARN\data\raw\Training.csv")
    df = preprocessor.load_data()
    preprocessor.explore_data()
    df_clean = preprocessor.clean_data()
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)