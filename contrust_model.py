#!/usr/bin/env python3
"""
ConTrust: Novel Conversational Trust Prediction Architecture

A hybrid deep learning model combining:
1. Hierarchical Attention Transformers for conversation understanding
2. Trust Evolution Memory for temporal dynamics
3. Multi-Modal Fusion for text, temporal, and behavioral features
4. Trust-Aware Self-Attention mechanism
5. Adaptive Multi-Task Learning with dynamic loss weighting
6. Conversation Context Encoder for global conversation understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, LayerNorm, Dropout
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class TrustAwareSelfAttention(nn.Module):
    """
    Novel Trust-Aware Self-Attention mechanism that incorporates trust relationships
    between conversation turns to enhance attention weights
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Trust-aware components
        self.trust_projection = nn.Linear(1, self.head_dim)
        self.trust_gate = nn.Linear(d_model + 1, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, trust_scores: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Standard attention computation
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Trust-aware attention modification
        trust_scores_padded = None
        if trust_scores is not None:
            # Ensure trust_scores matches the expected sequence length
            if trust_scores.size(1) != seq_len:
                if trust_scores.size(1) < seq_len:
                    # Pad trust scores to match sequence length
                    padding = torch.zeros(trust_scores.size(0), seq_len - trust_scores.size(1), 
                                        device=trust_scores.device)
                    trust_scores_padded = torch.cat([trust_scores, padding], dim=1)
                else:
                    # Truncate trust scores to match sequence length
                    trust_scores_padded = trust_scores[:, :seq_len]
            else:
                trust_scores_padded = trust_scores
            
            # Project trust scores to head dimension
            trust_proj = self.trust_projection(trust_scores_padded.unsqueeze(-1))  # [batch, seq_len, head_dim]
            trust_proj = trust_proj.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # [batch, heads, seq_len, head_dim]
            
            # Create trust affinity matrix
            trust_affinity = torch.matmul(trust_proj, trust_proj.transpose(-2, -1))
            trust_affinity = torch.sigmoid(trust_affinity)  # Normalize to [0,1]
            
            # Modulate attention scores with trust relationships
            scores = scores + 0.1 * trust_affinity  # Small trust boost
        
        if mask is not None:
            # Expand mask to match scores dimensions [batch, heads, seq_len, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Trust gating
        if trust_scores is not None:
            # Use the padded trust scores if available, otherwise use original
            trust_for_gating = trust_scores_padded if trust_scores_padded is not None else trust_scores
            
            # Ensure trust scores match attended sequence length
            if trust_for_gating.size(1) != attended.size(1):
                # If mismatch, pad or truncate trust_scores to match
                if trust_for_gating.size(1) < attended.size(1):
                    padding = torch.zeros(trust_for_gating.size(0), 
                                        attended.size(1) - trust_for_gating.size(1),
                                        device=trust_for_gating.device)
                    trust_for_gating = torch.cat([trust_for_gating, padding], dim=1)
                else:
                    trust_for_gating = trust_for_gating[:, :attended.size(1)]
            
            gate_input = torch.cat([attended, trust_for_gating.unsqueeze(-1)], dim=-1)
            gate = torch.sigmoid(self.trust_gate(gate_input))
            attended = attended * gate
        
        output = self.out_linear(attended)
        return output, attention_weights

class ConversationMemory(nn.Module):
    """
    Memory mechanism to track trust evolution throughout the conversation
    Uses a combination of LSTM and attention to maintain conversation state
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1, bidirectional=True)
        
        # Memory update mechanism
        self.memory_gate = nn.Linear(hidden_dim * 2 + input_dim, hidden_dim * 2)
        self.memory_transform = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # Trust evolution predictor
        self.trust_evolution = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x: torch.Tensor, conversation_lengths: torch.Tensor):
        batch_size, max_len, _ = x.shape
        
        # Pack padded sequence for efficient LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, conversation_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Memory update with gating
        memory_input = torch.cat([lstm_output, x], dim=-1)
        gate = torch.sigmoid(self.memory_gate(memory_input))
        
        # Update memory state
        memory_candidate = torch.tanh(self.memory_transform(lstm_output))
        memory_state = gate * memory_candidate + (1 - gate) * lstm_output
        
        # Predict trust evolution
        trust_evolution = self.trust_evolution(memory_state)
        
        return memory_state, trust_evolution

class MultiModalFusionEncoder(nn.Module):
    """
    Multi-modal encoder that processes text, temporal, and behavioral features separately
    then fuses them using cross-attention and gating mechanisms
    """
    def __init__(self, text_dim: int = 768, temporal_dim: int = 64, 
                 behavioral_dim: int = 128, fusion_dim: int = 512):
        super().__init__()
        
        # Individual encoders
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(text_dim, nhead=8, dim_feedforward=2048, dropout=0.1),
            num_layers=2
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        self.behavioral_encoder = nn.Sequential(
            nn.Linear(behavioral_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.text_to_temporal_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.text_to_behavioral_attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        
        # Fusion layers
        self.fusion_transform = nn.Linear(text_dim + 256 + 256, fusion_dim)
        self.fusion_gate = nn.Linear(fusion_dim, 3)  # 3 modalities
        
        # Projection layers
        self.text_proj = nn.Linear(text_dim, 256)
        
    def forward(self, text_features: torch.Tensor, temporal_features: torch.Tensor, 
                behavioral_features: torch.Tensor):
        
        # Encode each modality
        text_encoded = self.text_encoder(text_features.transpose(0, 1)).transpose(0, 1)
        
        # Handle temporal features: they are already [batch, seq_len, temporal_dim]
        # We encode each timestep
        batch_size, seq_len, temp_dim = temporal_features.shape
        temporal_flat = temporal_features.view(-1, temp_dim)
        temporal_encoded_flat = self.temporal_encoder(temporal_flat)
        temporal_encoded = temporal_encoded_flat.view(batch_size, seq_len, -1)
        
        # Behavioral features are [batch, behavioral_dim] - encode once per conversation
        behavioral_encoded = self.behavioral_encoder(behavioral_features)
        
        # Project text to same dimension
        text_proj = self.text_proj(text_encoded)
        
        # Cross-modal attention
        # Expand behavioral to match sequence length, temporal is already the right shape
        behavioral_expanded = behavioral_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        
        text_temp_attended, _ = self.text_to_temporal_attn(
            text_proj.transpose(0, 1), 
            temporal_encoded.transpose(0, 1),
            temporal_encoded.transpose(0, 1)
        )
        text_temp_attended = text_temp_attended.transpose(0, 1)
        
        text_behav_attended, _ = self.text_to_behavioral_attn(
            text_proj.transpose(0, 1),
            behavioral_expanded.transpose(0, 1),
            behavioral_expanded.transpose(0, 1)
        )
        text_behav_attended = text_behav_attended.transpose(0, 1)
        
        # Fusion with gating
        fused_features = torch.cat([
            text_encoded,
            text_temp_attended,
            text_behav_attended
        ], dim=-1)
        
        fused = self.fusion_transform(fused_features)
        
        # Adaptive gating based on modality importance
        gate_weights = F.softmax(self.fusion_gate(fused), dim=-1)
        
        return fused, gate_weights

class ConTrustModel(nn.Module):
    """
    Novel ConTrust: Conversational Trust Prediction Model
    
    Architecture combines:
    - Hierarchical attention transformers
    - Trust evolution memory
    - Multi-modal fusion
    - Trust-aware self-attention
    - Multi-task learning
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.n_layers = config.get('n_layers', 6)
        
        # Text encoder (BERT-based)
        self.text_model = AutoModel.from_pretrained(config.get('text_model', 'distilbert-base-uncased'))
        self.text_projection = nn.Linear(self.text_model.config.hidden_size, self.d_model)
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusionEncoder(
            text_dim=self.text_model.config.hidden_size,
            temporal_dim=config.get('temporal_dim', 64),
            behavioral_dim=config.get('behavioral_dim', 128),
            fusion_dim=self.d_model
        )
        
        # Trust-aware attention layers
        self.trust_attention_layers = nn.ModuleList([
            TrustAwareSelfAttention(self.d_model, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Conversation memory
        self.conversation_memory = ConversationMemory(self.d_model, hidden_dim=256)
        
        # Hierarchical attention (turn-level and conversation-level)
        self.turn_attention = nn.MultiheadAttention(self.d_model, self.n_heads)
        self.conversation_attention = nn.MultiheadAttention(self.d_model, self.n_heads)
        
        # Position encoding for conversation turns
        self.position_encoding = nn.Parameter(torch.randn(1000, self.d_model))
        
        # Multi-task heads
        self.trust_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Trust score 1-7
        )
        
        self.trust_categories_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3)  # Competence, Benevolence, Integrity
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # 7 emotion classes
        )
        
        self.engagement_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Engagement score
        )
        
        # Adaptive loss weighting
        self.loss_weights = nn.Parameter(torch.ones(4))  # 4 tasks
        
        # Layer normalization and dropout
        self.layer_norm = LayerNorm(self.d_model)
        self.dropout = Dropout(0.1)
        
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Forward pass through ConTrust model
        
        Args:
            batch: Dictionary containing:
                - input_ids: Text token IDs [batch, seq_len, tokens]
                - attention_mask: Text attention mask
                - temporal_features: Temporal features [batch, seq_len, temp_dim]
                - behavioral_features: Behavioral features [batch, behavioral_dim]
                - trust_scores: Previous trust scores [batch, seq_len]
                - conversation_lengths: Actual lengths of conversations
        """
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        temporal_features = batch['temporal_features']
        behavioral_features = batch['behavioral_features']
        trust_scores = batch.get('trust_scores', None)
        conversation_lengths = batch['conversation_lengths']
        
        batch_size, seq_len, max_tokens = input_ids.shape
        
        # Process text through BERT for each turn
        text_features = []
        for i in range(seq_len):
            turn_input_ids = input_ids[:, i, :]
            turn_attention_mask = attention_mask[:, i, :]
            
            text_output = self.text_model(
                input_ids=turn_input_ids,
                attention_mask=turn_attention_mask
            )
            text_features.append(text_output.last_hidden_state.mean(dim=1))  # Pool over tokens
        
        text_features = torch.stack(text_features, dim=1)  # [batch, seq_len, hidden]
        
        # Multi-modal fusion
        fused_features, modality_weights = self.multimodal_fusion(
            text_features, temporal_features, behavioral_features
        )
        
        # Add positional encoding
        seq_positions = torch.arange(seq_len, device=fused_features.device)
        pos_encoding = self.position_encoding[seq_positions].unsqueeze(0)
        fused_features = fused_features + pos_encoding
        
        # Trust-aware self-attention layers
        attended_features = fused_features
        attention_weights_list = []
        
        for attention_layer in self.trust_attention_layers:
            # Create proper attention mask: 1 for valid positions, 0 for padding
            seq_mask = torch.arange(seq_len, device=fused_features.device).unsqueeze(0) < conversation_lengths.unsqueeze(1)
            
            attended_features, attn_weights = attention_layer(
                attended_features, trust_scores, seq_mask
            )
            attended_features = self.layer_norm(attended_features + fused_features)
            attention_weights_list.append(attn_weights)
        
        # Conversation memory for trust evolution
        memory_state, trust_evolution = self.conversation_memory(
            attended_features, conversation_lengths
        )
        
        # Hierarchical attention
        # Turn-level attention
        turn_attended, turn_weights = self.turn_attention(
            attended_features.transpose(0, 1),
            attended_features.transpose(0, 1),
            attended_features.transpose(0, 1)
        )
        turn_attended = turn_attended.transpose(0, 1)
        
        # Conversation-level representation (aggregate turns)
        conversation_mask = torch.arange(seq_len, device=fused_features.device).unsqueeze(0) < conversation_lengths.unsqueeze(1)
        conversation_repr = (turn_attended * conversation_mask.unsqueeze(-1).float()).sum(dim=1) / conversation_lengths.unsqueeze(-1).float()
        
        # Multi-task predictions
        # Use last turn representation for current predictions
        last_turn_indices = (conversation_lengths - 1).long()
        last_turn_features = turn_attended[torch.arange(batch_size), last_turn_indices]
        
        # Trust prediction (main task)
        trust_pred = self.trust_head(last_turn_features)
        trust_categories_pred = self.trust_categories_head(last_turn_features)
        
        # Auxiliary tasks
        emotion_pred = self.emotion_head(last_turn_features)
        engagement_pred = self.engagement_head(conversation_repr)
        
        outputs = {
            'trust_score': trust_pred,
            'trust_categories': trust_categories_pred,
            'emotion': emotion_pred,
            'engagement': engagement_pred,
            'trust_evolution': trust_evolution,
            'attention_weights': attention_weights_list,
            'modality_weights': modality_weights,
            'conversation_representation': conversation_repr
        }
        
        return outputs
    
    def compute_loss(self, predictions: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive multi-task loss with dynamic weighting
        """
        losses = {}
        
        # Main trust prediction loss
        if 'trust_score' in targets and targets['trust_score'] is not None:
            trust_loss = F.mse_loss(predictions['trust_score'].squeeze(), targets['trust_score'])
            losses['trust'] = trust_loss
        
        # Trust categories loss
        if 'trust_categories' in targets and targets['trust_categories'] is not None:
            # Ensure both prediction and target have the same shape
            pred_cats = predictions['trust_categories']  # Should be [batch_size, 3]
            target_cats = targets['trust_categories']    # Should be [batch_size, 3]
            
            try:
                # If target is concatenated, reshape it
                if target_cats.dim() == 1 and pred_cats.dim() == 2:
                    target_cats = target_cats.view(pred_cats.size(0), -1)
                
                # Only compute loss if shapes match
                if target_cats.size() == pred_cats.size():
                    trust_cat_loss = F.mse_loss(pred_cats, target_cats)
                    losses['trust_categories'] = trust_cat_loss
                else:
                    logger.warning(f"Skipping trust categories loss due to shape mismatch: pred {pred_cats.shape}, target {target_cats.shape}")
            except Exception as e:
                logger.warning(f"Error in trust categories loss computation: {e}")
        
        # Emotion classification loss
        if 'emotion' in targets:
            emotion_loss = F.cross_entropy(predictions['emotion'], targets['emotion'])
            losses['emotion'] = emotion_loss
        
        # Engagement prediction loss
        if 'engagement' in targets:
            engagement_loss = F.mse_loss(predictions['engagement'].squeeze(), targets['engagement'])
            losses['engagement'] = engagement_loss
        
        # Trust evolution loss (temporal consistency)
        if 'trust_evolution' in targets and targets['trust_evolution'] is not None:
            try:
                pred_evo = predictions['trust_evolution'].squeeze(-1)  # Remove last dim if size 1
                target_evo = targets['trust_evolution']
                
                # Align shapes by truncating prediction to match target length
                batch_size = target_evo.size(0)
                target_seq_len = target_evo.size(1)
                
                if pred_evo.size(1) != target_seq_len:
                    if pred_evo.size(1) > target_seq_len:
                        # Truncate prediction to match target length
                        pred_evo = pred_evo[:, :target_seq_len]
                    else:
                        # Pad prediction to match target length
                        pad_length = target_seq_len - pred_evo.size(1)
                        padding = torch.zeros(batch_size, pad_length, device=pred_evo.device)
                        pred_evo = torch.cat([pred_evo, padding], dim=1)
                
                # Now shapes should match
                if pred_evo.shape == target_evo.shape:
                    evolution_loss = F.mse_loss(pred_evo, target_evo)
                    losses['evolution'] = evolution_loss
                else:
                    logger.debug(f"Evolution shapes still don't match after alignment: pred {pred_evo.shape}, target {target_evo.shape}")
            except Exception as e:
                logger.warning(f"Error in trust evolution loss computation: {e}")
        
        # Adaptive loss weighting using uncertainty
        loss_weights = F.softmax(self.loss_weights, dim=0)
        
        # Create a list of loss items to avoid dictionary modification during iteration
        loss_items = list(losses.items())
        
        total_loss = 0
        for i, (task, loss) in enumerate(loss_items):
            if i < len(loss_weights):
                weighted_loss = loss_weights[i] * loss
                total_loss += weighted_loss
                losses[f'{task}_weighted'] = weighted_loss
        
        losses['total'] = total_loss
        losses['weights'] = loss_weights
        
        return losses

class ConTrustTrainer:
    """
    Training pipeline for ConTrust model
    """
    def __init__(self, model: ConTrustModel, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer with different learning rates for different components
        bert_params = list(self.model.text_model.parameters())
        bert_param_ids = {id(p) for p in bert_params}
        other_params = [p for p in self.model.parameters() if id(p) not in bert_param_ids]
        
        self.optimizer = torch.optim.AdamW([
            {'params': bert_params, 'lr': config.get('bert_lr', 2e-5)},
            {'params': other_params, 'lr': config.get('lr', 1e-3)}
        ], weight_decay=config.get('weight_decay', 0.01))
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 10)
        
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {}
        num_batches = 0
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch)
            
            # Compute losses
            targets = {
                'trust_score': batch.get('target_trust'),  # Fix key mapping
                'trust_categories': batch.get('target_trust_categories'),
                'emotion': batch.get('target_emotion'),
                'engagement': batch.get('target_engagement'),
                'trust_evolution': batch.get('target_evolution')
            }
            
            losses = self.model.compute_loss(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if torch.is_tensor(value):
                    # Ensure scalar before calling .item()
                    if value.numel() == 1:
                        total_losses[key] = total_losses.get(key, 0) + value.item()
                    else:
                        # If tensor has multiple elements, take mean
                        total_losses[key] = total_losses.get(key, 0) + value.mean().item()
            
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.scheduler.step()
        
        return avg_losses
    
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                predictions = self.model(batch)
                
                targets = {
                    'trust_score': batch['target_trust'],
                    'trust_categories': batch.get('target_trust_categories'),
                    'emotion': batch.get('target_emotion'),
                    'engagement': batch.get('target_engagement'),
                    'trust_evolution': batch.get('target_evolution')
                }
                
                losses = self.model.compute_loss(predictions, targets)
                
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        # Ensure scalar before calling .item()
                        if value.numel() == 1:
                            total_losses[key] = total_losses.get(key, 0) + value.item()
                        else:
                            # If tensor has multiple elements, take mean
                            total_losses[key] = total_losses.get(key, 0) + value.mean().item()
                
                num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def train(self, train_dataloader, val_dataloader, epochs: int) -> Dict:
        """Full training loop"""
        history = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_losses = self.train_epoch(train_dataloader)
            history['train'].append(train_losses)
            
            # Validation
            val_losses = self.evaluate(val_dataloader)
            history['val'].append(val_losses)
            
            logger.info(f"Train Loss: {train_losses['total']:.4f}, Val Loss: {val_losses['total']:.4f}")
            
            # Early stopping
            if val_losses['total'] < self.best_loss:
                self.best_loss = val_losses['total']
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'contrust_best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history

def create_contrust_config() -> Dict:
    """Create default configuration for ConTrust model"""
    return {
        'text_model': 'distilbert-base-uncased',
        'd_model': 768,  # Match DistilBERT output dimension
        'n_heads': 12,   # Divisible by 768
        'n_layers': 6,
        'temporal_dim': 4,   # Match actual temporal features: response_time, turn_pos, rel_pos, conv_length
        'behavioral_dim': 8,  # Match actual behavioral features: 6 numeric + 2 categorical
        'bert_lr': 2e-5,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'patience': 10,
        'max_seq_length': 512,
        'max_conversation_length': 50
    }

if __name__ == "__main__":
    # Example usage
    config = create_contrust_config()
    model = ConTrustModel(config)
    
    print("🚀 ConTrust Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size, seq_len, max_tokens = 2, 5, 64
    
    test_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len, max_tokens)),
        'attention_mask': torch.ones((batch_size, seq_len, max_tokens)),
        'temporal_features': torch.randn(batch_size, seq_len, 64),
        'behavioral_features': torch.randn(batch_size, 128),
        'trust_scores': torch.randn(batch_size, seq_len),
        'conversation_lengths': torch.tensor([5, 3])
    }
    
    outputs = model(test_batch)
    print(f"✅ Forward pass successful!")
    print(f"Trust prediction shape: {outputs['trust_score'].shape}")
    print(f"Trust categories shape: {outputs['trust_categories'].shape}")
    print(f"Emotion prediction shape: {outputs['emotion'].shape}")
    print(f"Engagement prediction shape: {outputs['engagement'].shape}")
