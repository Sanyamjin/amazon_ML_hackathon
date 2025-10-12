import torch
import torch.nn as nn
from transformers import BertModel

class BERTPricePredictor(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_numerical_features=20, 
                 dropout_rate=0.3, hidden_dim=768):
        super().__init__()
        
        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Numerical feature processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Price prediction head
        self.price_head = nn.Linear(128, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize additional layers with Xavier initialization"""
        for module in [self.numerical_processor, self.fusion_layer, self.price_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, numerical_features):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # [batch, 768]
        
        # Numerical feature processing
        numerical_emb = self.numerical_processor(numerical_features)  # [batch, 64]
        
        # Feature fusion
        combined_features = torch.cat([text_features, numerical_emb], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        
        # Price prediction (ensure positive output)
        price = torch.relu(self.price_head(fused_features)).squeeze(-1) + 0.01
        
        return price