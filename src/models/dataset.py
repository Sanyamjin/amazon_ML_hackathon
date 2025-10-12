import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import numpy as np

class ProductPriceDataset(Dataset):
    def __init__(self, df, tokenizer_name='bert-base-uncased', max_length=512, is_training=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.is_training = is_training
        
        # Prepare text inputs
        self._prepare_text()
        
        # Prepare numerical features
        self._prepare_numerical_features()
    
    def _prepare_text(self):
        """Combine text fields into enhanced input"""
        enhanced_texts = []
        
        for idx, row in self.df.iterrows():
            text_parts = []
            
            # Item name
            if pd.notna(row.get('item_name', '')):
                text_parts.append(f"PRODUCT: {row['item_name']}")
            
            # Product description
            if pd.notna(row.get('prod_desc', '')) and str(row['prod_desc']).strip():
                desc = str(row['prod_desc'])[:300]  # Limit description length
                text_parts.append(f"DESCRIPTION: {desc}")
            
            # Bullet points
            if pd.notna(row.get('bullet_points', '')) and str(row['bullet_points']).strip():
                bullets = str(row['bullet_points'])[:200]  # Limit bullet length
                text_parts.append(f"FEATURES: {bullets}")
            
            # Value and unit information
            if pd.notna(row.get('value', '')) and pd.notna(row.get('unit', '')):
                text_parts.append(f"SIZE: {row['value']} {row['unit']}")
            
            enhanced_text = ' [SEP] '.join(text_parts)
            enhanced_texts.append(enhanced_text)
        
        self.texts = enhanced_texts
    
    def _prepare_numerical_features(self):
        """Prepare numerical features for the model"""
        # Select relevant numerical features
        feature_cols = [
            # Original features
            'value', 'num_bullets', 'has_description',
            
            # Text statistics
            'combined_text_length', 'combined_word_count', 'combined_unique_words',
            'combined_avg_word_length', 'combined_readability_score',
            
            # Brand indicators
            'combined_premium_count', 'combined_organic_count', 'combined_budget_count',
            
            # Size features
            'item_size_mentions', 'item_total_volume_oz', 'item_pack_size',
            
            # Category features
            'category_food', 'category_beverage', 'category_personal_care',
            
            # Engineered features
            'total_premium_indicators', 'category_diversity', 'bulk_factor'
        ]
        
        # Extract available features
        available_features = []
        for col in feature_cols:
            if col in self.df.columns:
                values = self.df[col].fillna(0).values
                available_features.append(values)
            else:
                # Create zero column for missing features
                available_features.append(np.zeros(len(self.df)))
        
        self.numerical_features = np.column_stack(available_features).astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Tokenize text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'numerical_features': torch.tensor(self.numerical_features[idx], dtype=torch.float32),
            'prod_id': self.df.iloc[idx].get('prod_id', idx)
        }
        
        if self.is_training and 'price' in self.df.columns:
            item['price'] = torch.tensor(self.df.iloc[idx]['price'], dtype=torch.float32)
        
        return item