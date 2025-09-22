#!/usr/bin/env python3
"""
Advanced BERT-based Models for Trust and Emotion Detection
- BERT fine-tuning for trust regression
- RoBERTa for emotion classification  
- DistilBERT for lightweight inference
- Cross-attention mechanisms for conversation context
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    BertModel, RobertaModel, DistilBertModel,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ConversationalTrustDataset(Dataset):
    """Dataset for BERT-based trust prediction with conversation context"""
    
    def __init__(self, texts: List[str], labels: List[float], 
                 conversation_ids: List[str], turn_ids: List[int],
                 tokenizer, max_length: int = 512, include_context: bool = True):
        self.texts = texts
        self.labels = labels
        self.conversation_ids = conversation_ids  
        self.turn_ids = turn_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context
        
        # Build conversation context mapping
        self.context_map = self._build_context_map()
        
    def _build_context_map(self) -> Dict[Tuple[str, int], str]:
        """Build mapping of (conv_id, turn_id) to previous context"""
        context_map = {}
        conv_texts = {}
        
        # Group texts by conversation
        for i, (conv_id, turn_id) in enumerate(zip(self.conversation_ids, self.turn_ids)):
            if conv_id not in conv_texts:
                conv_texts[conv_id] = {}
            conv_texts[conv_id][turn_id] = self.texts[i]
        
        # Build context for each turn
        for conv_id, turns in conv_texts.items():
            for turn_id in sorted(turns.keys()):
                if self.include_context and turn_id > 1:
                    # Include previous 2 turns as context
                    context_turns = []
                    for prev_turn in range(max(1, turn_id - 2), turn_id):
                        if prev_turn in turns:
                            context_turns.append(turns[prev_turn])
                    context = " [SEP] ".join(context_turns) if context_turns else ""
                    context_map[(conv_id, turn_id)] = context
                else:
                    context_map[(conv_id, turn_id)] = ""
        
        return context_map
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        conv_id = self.conversation_ids[idx]
        turn_id = self.turn_ids[idx]
        
        # Get context
        context = self.context_map.get((conv_id, turn_id), "")
        
        # Combine context and current text
        if context:
            full_text = f"{context} [SEP] {text}"
        else:
            full_text = text
            
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class BERTTrustRegressor(nn.Module):
    """BERT-based trust score regressor"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.3):
        super(BERTTrustRegressor, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
        # Trust score should be between 1-7
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.regressor(output)
        
        # Scale to 1-7 range
        output = 1 + 6 * self.sigmoid(output)
        return output

class MultiTaskBERT(nn.Module):
    """Multi-task BERT for trust + emotion prediction"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_emotions: int = 7, dropout: float = 0.3):
        super(MultiTaskBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Task-specific heads
        self.trust_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.emotion_head = nn.Linear(self.bert.config.hidden_size, num_emotions)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, task='both'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        shared_features = self.dropout(pooled_output)
        
        results = {}
        
        if task in ['trust', 'both']:
            trust_output = self.trust_head(shared_features)
            trust_output = 1 + 6 * self.sigmoid(trust_output)  # Scale to 1-7
            results['trust'] = trust_output
            
        if task in ['emotion', 'both']:
            emotion_output = self.emotion_head(shared_features)
            results['emotion'] = emotion_output
            
        return results

class ConversationContextBERT(nn.Module):
    """BERT with explicit conversation context modeling"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 max_context_turns: int = 5, dropout: float = 0.3):
        super(ConversationContextBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_context_turns = max_context_turns
        
        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.trust_head = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, context_embeddings=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Apply context attention if available
        if context_embeddings is not None:
            attended_output, _ = self.context_attention(
                query=pooled_output.unsqueeze(1),
                key=context_embeddings,
                value=context_embeddings
            )
            pooled_output = self.layer_norm(pooled_output + attended_output.squeeze(1))
        
        output = self.dropout(pooled_output)
        output = self.trust_head(output)
        output = 1 + 6 * self.sigmoid(output)
        
        return output

class BERTTrainer:
    """Trainer for BERT-based models"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def train_trust_regressor(self, train_dataset, val_dataset, 
                            epochs: int = 3, batch_size: int = 16,
                            learning_rate: float = 2e-5):
        """Train BERT trust regressor"""
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Learning rate scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    total_val_loss += loss.item()
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Calculate R²
            val_r2 = 1 - (np.sum((np.array(val_true) - np.array(val_predictions))**2) / 
                         np.sum((np.array(val_true) - np.mean(val_true))**2))
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'  Train Loss: {avg_train_loss:.4f}')
            logger.info(f'  Val Loss: {avg_val_loss:.4f}')
            logger.info(f'  Val R²: {val_r2:.4f}')
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_r2': val_r2
        }
    
    def train_multitask(self, train_dataset, val_dataset, 
                       epochs: int = 3, batch_size: int = 16,
                       learning_rate: float = 2e-5, 
                       trust_weight: float = 1.0, emotion_weight: float = 0.5):
        """Train multi-task BERT model"""
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        trust_criterion = nn.MSELoss()
        emotion_criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                trust_labels = batch['trust_labels'].to(self.device)
                emotion_labels = batch['emotion_labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask, task='both')
                
                trust_loss = trust_criterion(outputs['trust'].squeeze(), trust_labels)
                emotion_loss = emotion_criterion(outputs['emotion'], emotion_labels)
                
                total_loss_batch = (trust_weight * trust_loss + 
                                  emotion_weight * emotion_loss)
                
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            logger.info(f'Epoch {epoch+1}/{epochs}: Loss: {total_loss/len(train_loader):.4f}')
    
    def predict(self, text: str, context: str = "") -> float:
        """Make prediction for single text"""
        self.model.eval()
        
        # Prepare input
        if context:
            full_text = f"{context} [SEP] {text}"
        else:
            full_text = text
            
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            return output.item()

def create_bert_models() -> Dict[str, nn.Module]:
    """Create different BERT model variants"""
    
    models = {
        'bert_trust': BERTTrustRegressor('bert-base-uncased'),
        'distilbert_trust': BERTTrustRegressor('distilbert-base-uncased'),
        'roberta_trust': BERTTrustRegressor('roberta-base'),
        'multitask_bert': MultiTaskBERT('bert-base-uncased'),
        'context_bert': ConversationContextBERT('bert-base-uncased')
    }
    
    return models

def prepare_bert_data(df: pd.DataFrame, tokenizer) -> Tuple[Dataset, Dataset]:
    """Prepare data for BERT training"""
    
    # Filter agent turns with trust scores
    agent_df = df[(df['speaker'] == 'agent') & (df['trust_score'].notna())].copy()
    
    texts = agent_df['utterance'].tolist()
    labels = agent_df['trust_score'].tolist()
    conv_ids = agent_df['conversation_id'].tolist()
    turn_ids = agent_df['turn_id'].tolist()
    
    # Split data
    split_idx = int(0.8 * len(texts))
    
    train_dataset = ConversationalTrustDataset(
        texts[:split_idx], labels[:split_idx], 
        conv_ids[:split_idx], turn_ids[:split_idx],
        tokenizer
    )
    
    val_dataset = ConversationalTrustDataset(
        texts[split_idx:], labels[split_idx:],
        conv_ids[split_idx:], turn_ids[split_idx:],
        tokenizer
    )
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BERTTrustRegressor()
    trainer = BERTTrainer(model, tokenizer)
    
    print("Advanced BERT models ready for training!")
