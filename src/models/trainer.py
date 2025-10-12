import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import mean_absolute_error
import os

class SMAPELoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, min=self.epsilon)
        targets = torch.clamp(targets, min=self.epsilon)
        
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(predictions) + torch.abs(targets)) / 2
        smape = (numerator / denominator).mean() * 100
        
        return smape

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimizer with different learning rates for BERT and other layers
        bert_params = list(self.model.bert.parameters())
        other_params = [p for p in self.model.parameters() if not any(p is bp for bp in bert_params)]
        
        self.optimizer = AdamW([
            {'params': bert_params, 'lr': config['bert_lr']},
            {'params': other_params, 'lr': config['other_lr']}
        ], weight_decay=config['weight_decay'])
        
        # Scheduler
        total_steps = len(self.train_loader) * config['epochs']
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # Loss functions
        self.smape_loss = SMAPELoss()
        self.mse_loss = nn.MSELoss()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_smape = 0
        predictions, targets = [], []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            prices = batch['price'].to(self.device)
            
            # Forward pass
            pred_prices = self.model(input_ids, attention_mask, numerical_features)
            
            # Combined loss (MSE for stability + SMAPE for target metric)
            mse_loss = self.mse_loss(pred_prices, prices)
            smape_loss = self.smape_loss(pred_prices, prices)
            loss = 0.7 * mse_loss + 0.3 * smape_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            total_loss += loss.item()
            total_smape += smape_loss.item()
            predictions.extend(pred_prices.detach().cpu().numpy())
            targets.extend(prices.detach().cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        avg_smape = total_smape / len(self.train_loader)
        mae = mean_absolute_error(targets, predictions)
        
        return avg_loss, avg_smape, mae
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_smape = 0
        predictions, targets = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                prices = batch['price'].to(self.device)
                
                pred_prices = self.model(input_ids, attention_mask, numerical_features)
                
                mse_loss = self.mse_loss(pred_prices, prices)
                smape_loss = self.smape_loss(pred_prices, prices)
                loss = 0.7 * mse_loss + 0.3 * smape_loss
                
                total_loss += loss.item()
                total_smape += smape_loss.item()
                predictions.extend(pred_prices.cpu().numpy())
                targets.extend(prices.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        avg_smape = total_smape / len(self.val_loader)
        mae = mean_absolute_error(targets, predictions)
        
        return avg_loss, avg_smape, mae
    
    def train(self):
        best_val_smape = float('inf')
        patience = 0
        
        for epoch in range(self.config['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Training
            train_loss, train_smape, train_mae = self.train_epoch()
            
            # Validation
            val_loss, val_smape, val_mae = self.validate()
            
            self.logger.info(
                f"Train - Loss: {train_loss:.4f}, SMAPE: {train_smape:.2f}%, MAE: {train_mae:.2f}"
            )
            self.logger.info(
                f"Val - Loss: {val_loss:.4f}, SMAPE: {val_smape:.2f}%, MAE: {val_mae:.2f}"
            )
            
            # Save best model
            if val_smape < best_val_smape:
                best_val_smape = val_smape
                self.save_checkpoint(epoch, val_smape, 'best_model.pth')
                patience = 0
            else:
                patience += 1
            
            # Early stopping
            if patience >= self.config['patience']:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_smape
    
    def save_checkpoint(self, epoch, val_smape, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_smape': val_smape,
            'config': self.config
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/{filename}')
        self.logger.info(f"Model saved: checkpoints/{filename}")