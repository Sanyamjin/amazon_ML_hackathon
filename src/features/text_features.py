from __future__ import annotations
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from textstat import flesch_reading_ease, flesch_kincaid_grade
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def ensure_nltk_resources():
    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            nltk.download(resource)

@dataclass
class TextFeatureConfig:
    def __getattr__(self, item):
        return None

class TextFeatureExtractor:
    def __init__(self, config: Optional[TextFeatureConfig] = None):
        ensure_nltk_resources()
        self.config = config or TextFeatureConfig()
        self.stop_words = set(stopwords.words('english'))
        
        # Price-relevant patterns
        self.brand_indicators = {
            'premium': ['premium', 'gourmet', 'artisan', 'craft', 'luxury', 'elite'],
            'organic': ['organic', 'natural', 'bio', 'eco', 'green'],
            'budget': ['value', 'economy', 'basic', 'essential', 'budget'],
            'international': ['imported', 'italian', 'french', 'german', 'japanese']
        }
        
        self.size_patterns = re.compile(
            r'\b(\d+\.?\d*)\s*(oz|ounce|lb|pound|ml|liter|count|pack|ct|g|kg|fl\.?\s*oz|piece|serving)\b',
            re.IGNORECASE
        )
        
        self.quantity_patterns = re.compile(
            r'\b(pack\s*of\s*\d+|case\s*of\s*\d+|\d+\s*pack|\d+\s*count|\d+\s*ct)\b',
            re.IGNORECASE
        )
    
    def extract_brand_indicators(self, text):
        """Extract brand positioning indicators from text"""
        if pd.isna(text):
            return {f'{category}_count': 0 for category in self.brand_indicators}
        
        text_lower = str(text).lower()
        features = {}
        
        for category, keywords in self.brand_indicators.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            features[f'{category}_count'] = count
            
        return features
    
    def extract_size_quantity_features(self, text):
        """Extract size and quantity information"""
        if pd.isna(text):
            return {
                'size_mentions': 0,
                'quantity_mentions': 0,
                'total_volume_oz': 0,
                'pack_size': 0
            }
        
        text = str(text)
        features = {}
        
        # Size mentions
        size_matches = self.size_patterns.findall(text)
        features['size_mentions'] = len(size_matches)
        
        # Calculate total volume in oz (rough conversion)
        total_volume = 0
        for value, unit in size_matches:
            try:
                val = float(value)
                if unit.lower() in ['oz', 'ounce', 'fl oz']:
                    total_volume += val
                elif unit.lower() in ['lb', 'pound']:
                    total_volume += val * 16  # Convert to oz
                elif unit.lower() in ['ml']:
                    total_volume += val * 0.033814  # Convert to oz
                elif unit.lower() in ['liter']:
                    total_volume += val * 33.814
            except ValueError:
                continue
        
        features['total_volume_oz'] = total_volume
        
        # Quantity/pack information
        quantity_matches = self.quantity_patterns.findall(text)
        features['quantity_mentions'] = len(quantity_matches)
        
        # Extract pack size
        pack_size = 0
        pack_pattern = re.search(r'pack\s*of\s*(\d+)|case\s*of\s*(\d+)|(\d+)\s*pack', text, re.IGNORECASE)
        if pack_pattern:
            pack_size = int([g for g in pack_pattern.groups() if g][0])
        
        features['pack_size'] = pack_size
        
        return features
    
    def extract_text_statistics(self, text):
        """Extract general text statistics"""
        if pd.isna(text):
            return {
                'text_length': 0,
                'word_count': 0,
                'unique_words': 0,
                'avg_word_length': 0,
                'readability_score': 0,
                'exclamation_count': 0,
                'uppercase_ratio': 0
            }
        
        text = str(text)
        
        # Basic statistics
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Word-level features
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha() and w not in self.stop_words]
        
        features['unique_words'] = len(set(words))
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Readability
        try:
            features['readability_score'] = flesch_reading_ease(text)
        except:
            features['readability_score'] = 0
        
        return features
    
    def extract_product_category_features(self, text):
        """Extract product category indicators"""
        if pd.isna(text):
            return {'category_food': 0, 'category_beverage': 0, 'category_personal_care': 0, 'category_household': 0}
        
        text = str(text).lower()
        
        categories = {
            'category_food': ['food', 'snack', 'meal', 'sauce', 'spice', 'cooking', 'baking', 'ingredient'],
            'category_beverage': ['drink', 'beverage', 'juice', 'soda', 'water', 'coffee', 'tea', 'wine'],
            'category_personal_care': ['shampoo', 'soap', 'lotion', 'cream', 'care', 'beauty', 'health'],
            'category_household': ['clean', 'detergent', 'paper', 'tissue', 'home', 'kitchen', 'bathroom']
        }
        
        features = {}
        for category, keywords in categories.items():
            features[category] = int(any(keyword in text for keyword in keywords))
        
        return features
    
    def process_text_features(self, df, text_columns=['item_name', 'prod_desc', 'bullet_points']):
        """Process all text features for a dataframe"""
        all_features = []
        
        for idx, row in df.iterrows():
            # Combine all text fields
            combined_text = ''
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    combined_text += ' ' + str(row[col])
            
            # Extract features from individual columns and combined text
            features = {'prod_id': row.get('prod_id', idx)}
            
            # Features from item name
            if 'item_name' in df.columns:
                item_features = self.extract_brand_indicators(row['item_name'])
                features.update({f'item_{k}': v for k, v in item_features.items()})
                
                size_features = self.extract_size_quantity_features(row['item_name'])
                features.update({f'item_{k}': v for k, v in size_features.items()})
            
            # Features from combined text
            combined_features = self.extract_text_statistics(combined_text)
            features.update({f'combined_{k}': v for k, v in combined_features.items()})
            
            brand_features = self.extract_brand_indicators(combined_text)
            features.update({f'combined_{k}': v for k, v in brand_features.items()})
            
            category_features = self.extract_product_category_features(combined_text)
            features.update(category_features)
            
            all_features.append(features)
        
        return pd.DataFrame(all_features)