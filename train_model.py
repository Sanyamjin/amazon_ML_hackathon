import sys
import os
sys.path.append('src')

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from features.text_features import TextFeatureExtractor
from features.feature_engineering import FeatureEngineer
from models.dataset import ProductPriceDataset
from models.bert_model import BERTPricePredictor
from models.trainer import ModelTrainer

def main():
    print("=== Text-Only Price Prediction Training ===")
    
    # Configuration
    config = {
        'batch_size': 16,
        'bert_lr': 2e-5,          # Lower learning rate for BERT
        'other_lr': 1e-3,         # Higher learning rate for other layers
        'weight_decay': 0.01,
        'epochs': 15,
        'warmup_steps': 500,
        'patience': 3,
        'max_length': 512
    }
    
    # 1. Load cleaned data
    print("Loading cleaned data...")
    train_df = pd.read_csv('dataset/cleaned_train.csv')
    
    print(f"Loaded {len(train_df)} training samples")
    
    # 2. Extract text features
    print("Extracting text features...")
    text_extractor = TextFeatureExtractor()
    text_features = text_extractor.process_text_features(train_df)
    
    # 3. Merge with original data
    enhanced_df = train_df.merge(text_features, on='prod_id', how='left')
    
    # 4. Feature engineering
    print("Engineering additional features...")
    feature_engineer = FeatureEngineer()
    final_df = feature_engineer.fit_transform(enhanced_df, target_col='price')
    
    print(f"Final feature count: {len(final_df.columns)}")
    
    # 5. Train-validation split
    price_bins = None
    try:
        price_bins = pd.qcut(final_df["price"], q=5, duplicates="drop")
        if price_bins.value_counts().min() < 2:
            price_bins = None
    except ValueError:
        price_bins = None

    train_df_final, val_df_final = train_test_split(
        final_df,
        test_size=0.2,
        random_state=42,
        stratify=price_bins,
    )
    
    print(f"Train: {len(train_df_final)}, Validation: {len(val_df_final)}")
    
    # 6. Create datasets
    print("Creating PyTorch datasets...")
    train_dataset = ProductPriceDataset(train_df_final, max_length=config['max_length'], is_training=True)
    val_dataset = ProductPriceDataset(val_df_final, max_length=config['max_length'], is_training=True)
    
    # 7. Initialize model
    print("Initializing BERT model...")
    model = BERTPricePredictor(
        bert_model_name='bert-base-uncased',
        num_numerical_features=train_dataset.numerical_features.shape[1],
        dropout_rate=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 8. Train model
    print("Starting training...")
    trainer = ModelTrainer(model, train_dataset, val_dataset, config)
    best_smape = trainer.train()
    
    print(f"Training completed! Best validation SMAPE: {best_smape:.2f}%")
    
    # 9. Save feature engineer for inference
    import joblib
    joblib.dump(feature_engineer, 'checkpoints/feature_engineer.pkl')
    joblib.dump(text_extractor, 'checkpoints/text_extractor.pkl')
    
    print("Feature processors saved for inference!")

if __name__ == "__main__":
    main()