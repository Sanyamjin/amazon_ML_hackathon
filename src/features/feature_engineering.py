import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.tfidf_vectorizer = None
        self.is_fitted = False
    
    def engineer_numerical_features(self, df):
        """Create advanced numerical features"""
        engineered_df = df.copy()
        
        # Price per unit calculations
        if 'value' in df.columns and 'price' in df.columns:
            engineered_df['price_per_unit'] = df['price'] / df['value'].replace(0, 1)
        
        # Volume-based features
        if 'item_total_volume_oz' in df.columns:
            engineered_df['price_per_oz'] = df['price'] / df['item_total_volume_oz'].replace(0, 1)
        
        # Pack size efficiency
        if 'item_pack_size' in df.columns:
            engineered_df['bulk_factor'] = np.log1p(df['item_pack_size'])
            engineered_df['is_bulk'] = (df['item_pack_size'] > 1).astype(int)
        
        # Text complexity features
        if 'combined_word_count' in df.columns and 'combined_unique_words' in df.columns:
            engineered_df['text_complexity_ratio'] = (
                df['combined_unique_words'] / df['combined_word_count'].replace(0, 1)
            )
        
        # Premium indicators
        premium_cols = [col for col in df.columns if 'premium_count' in col or 'organic_count' in col]
        if premium_cols:
            engineered_df['total_premium_indicators'] = df[premium_cols].sum(axis=1)
            engineered_df['has_premium_indicators'] = (engineered_df['total_premium_indicators'] > 0).astype(int)
        
        # Category diversity
        category_cols = [col for col in df.columns if col.startswith('category_')]
        if category_cols:
            engineered_df['category_diversity'] = df[category_cols].sum(axis=1)
        
        return engineered_df
    
    def create_interaction_features(self, df):
        """Create interaction features between important variables"""
        interaction_df = df.copy()
        
        # Size × Premium interaction
        if 'item_total_volume_oz' in df.columns and 'total_premium_indicators' in df.columns:
            interaction_df['premium_size_interaction'] = (
                df['item_total_volume_oz'] * df['total_premium_indicators']
            )
        
        # Pack size × Category interaction
        if 'item_pack_size' in df.columns:
            for cat_col in [col for col in df.columns if col.startswith('category_')]:
                interaction_df[f'pack_{cat_col}_interaction'] = df['item_pack_size'] * df[cat_col]
        
        # Text length × Premium interaction
        if 'combined_text_length' in df.columns and 'total_premium_indicators' in df.columns:
            interaction_df['text_premium_interaction'] = (
                np.log1p(df['combined_text_length']) * df['total_premium_indicators']
            )
        
        return interaction_df
    
    def fit_transform(self, df, target_col='price'):
        """Fit transformers and transform the data"""
        self.is_fitted = True
        
        # Engineer features
        df_engineered = self.engineer_numerical_features(df)
        df_final = self.create_interaction_features(df_engineered)
        
        # Identify numerical columns for scaling
        numerical_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if 'prod_id' in numerical_cols:
            numerical_cols.remove('prod_id')
        
        # Fit and transform numerical features
        for col in numerical_cols:
            scaler = StandardScaler()
            df_final[f'{col}_scaled'] = scaler.fit_transform(df_final[[col]])
            self.scalers[col] = scaler
        
        return df_final
    
    def transform(self, df):
        """Transform new data using fitted transformers"""
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
        
        # Engineer features
        df_engineered = self.engineer_numerical_features(df)
        df_final = self.create_interaction_features(df_engineered)
        
        # Transform numerical features
        for col, scaler in self.scalers.items():
            if col in df_final.columns:
                df_final[f'{col}_scaled'] = scaler.transform(df_final[[col]])
            else:
                df_final[f'{col}_scaled'] = 0  # Handle missing columns
        
        return df_final