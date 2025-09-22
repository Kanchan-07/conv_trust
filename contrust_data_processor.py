#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for ConTrust Model

Prepares conversational trust dataset for training the novel ConTrust architecture
with proper tokenization, feature extraction, and conversation-level processing
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

logger = logging.getLogger(__name__)

class ConTrustDataProcessor:
    """
    Data processor for ConTrust model that handles conversation-level preprocessing
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.get('text_model', 'distilbert-base-uncased'))
        
        # Add special tokens for conversation structure
        special_tokens = ['[TURN_SEP]', '[CONV_START]', '[CONV_END]', '[USER]', '[AGENT]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Label encoders
        self.emotion_encoder = LabelEncoder()
        self.scenario_encoder = LabelEncoder()
        self.model_encoder = LabelEncoder()
        
        # Feature scalers
        self.temporal_scaler = StandardScaler()
        self.behavioral_scaler = StandardScaler()
        
        # Conversation statistics for normalization
        self.conversation_stats = {}
        
        # Fitted status
        self.fitted = False
        
    def load_conversations(self, data_dir: str) -> pd.DataFrame:
        """Load and merge conversation and metadata files"""
        from train_trust_models import TrustDatasetLoader
        
        loader = TrustDatasetLoader(data_dir)
        dataset = loader.load_data()
        
        # Filter to agent turns with trust scores for training
        agent_data = dataset[
            (dataset['speaker'] == 'agent') & 
            (dataset['trust_score'].notna())
        ].copy()
        
        logger.info(f"Loaded {len(dataset)} total turns, {len(agent_data)} agent turns with trust scores")
        return dataset, agent_data
    
    def extract_conversation_features(self, conversations_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract conversation-level features for each conversation"""
        
        conversation_features = {}
        
        for conv_id in tqdm(conversations_df['conversation_id'].unique(), desc="Processing conversations"):
            conv_data = conversations_df[conversations_df['conversation_id'] == conv_id].sort_values('turn_id')
            
            # Basic conversation statistics
            conv_length = len(conv_data)
            agent_turns = conv_data[conv_data['speaker'] == 'agent']
            user_turns = conv_data[conv_data['speaker'] == 'user']
            
            # Text features per turn
            turn_texts = []
            turn_speakers = []
            turn_emotions = []
            turn_trust_scores = []
            turn_trust_categories = []
            
            # Temporal features
            response_times = []
            turn_positions = []
            
            for idx, turn in conv_data.iterrows():
                # Text processing
                speaker_token = '[AGENT]' if turn['speaker'] == 'agent' else '[USER]'
                turn_text = f"{speaker_token} {turn['utterance']}"
                turn_texts.append(turn_text)
                turn_speakers.append(turn['speaker'])
                
                # Emotion and trust
                turn_emotions.append(turn.get('emotion_detected', 'neutral'))
                
                if turn['speaker'] == 'agent' and pd.notna(turn['trust_score']):
                    turn_trust_scores.append(float(turn['trust_score']))
                    
                    # Trust categories
                    trust_cats = [
                        turn.get('competence_score', 0.0),
                        turn.get('benevolence_score', 0.0),
                        turn.get('integrity_score', 0.0)
                    ]
                    turn_trust_categories.append(trust_cats)
                else:
                    turn_trust_scores.append(None)
                    turn_trust_categories.append([0.0, 0.0, 0.0])
                
                # Temporal features
                response_times.append(turn.get('response_time', 0.0))
                turn_positions.append(turn.get('turn_id', 0))
            
            # Conversation-level aggregated features
            avg_response_time = np.mean([rt for rt in response_times if rt > 0])
            trust_evolution = [ts for ts in turn_trust_scores if ts is not None]
            
            # Behavioral features
            behavioral_features = [
                conv_length,  # Conversation length
                len(agent_turns) / conv_length if conv_length > 0 else 0,  # Agent turn ratio
                avg_response_time if not np.isnan(avg_response_time) else 2.0,  # Avg response time
                len(set(turn_emotions)),  # Emotion diversity
                np.std(trust_evolution) if len(trust_evolution) > 1 else 0.0,  # Trust variance
                len(trust_evolution),  # Number of trust scores
                conv_data['scenario'].iloc[0] if 'scenario' in conv_data.columns else 'general',  # Scenario
                conv_data['agent_model'].iloc[0] if 'agent_model' in conv_data.columns else 'default',  # Model
            ]
            
            conversation_features[conv_id] = {
                'texts': turn_texts,
                'speakers': turn_speakers,
                'emotions': turn_emotions,
                'trust_scores': turn_trust_scores,
                'trust_categories': turn_trust_categories,
                'response_times': response_times,
                'turn_positions': turn_positions,
                'behavioral_features': behavioral_features,
                'conversation_length': conv_length,
                'trust_evolution': trust_evolution
            }
        
        return conversation_features
    
    def fit_encoders(self, conversation_features: Dict):
        """Fit label encoders and scalers on the dataset"""
        
        all_emotions = []
        all_scenarios = []
        all_models = []
        all_temporal_features = []
        all_behavioral_features = []
        
        for conv_id, features in conversation_features.items():
            all_emotions.extend(features['emotions'])
            
            # Extract scenario and model from behavioral features
            scenario = features['behavioral_features'][6]
            model = features['behavioral_features'][7]
            
            all_scenarios.append(scenario)
            all_models.append(model)
            
            # Temporal features per turn
            for i in range(len(features['texts'])):
                temporal_feats = [
                    features['response_times'][i],
                    features['turn_positions'][i],
                    i / len(features['texts']),  # Relative position
                    len(features['texts']),  # Conversation length context
                ]
                all_temporal_features.append(temporal_feats)
            
            # Behavioral features (numerical part + categorical encoded)
            behavioral_numeric = features['behavioral_features'][:6]  # First 6 are numeric
            scenario = features['behavioral_features'][6]
            model = features['behavioral_features'][7]
            
            # We need to encode these during fitting to get consistent dimensions
            scenario_encoded = len(all_scenarios)  # Temporary encoding for fitting
            model_encoded = len(all_models)  # Temporary encoding for fitting
            
            behavioral_full = behavioral_numeric + [scenario_encoded, model_encoded]
            all_behavioral_features.append(behavioral_full)
        
        # Fit encoders
        self.emotion_encoder.fit(all_emotions)
        self.scenario_encoder.fit(all_scenarios)
        self.model_encoder.fit(all_models)
        
        # Fit scalers
        self.temporal_scaler.fit(all_temporal_features)
        self.behavioral_scaler.fit(all_behavioral_features)
        
        self.fitted = True
        logger.info("Encoders and scalers fitted")
        logger.info(f"Emotions: {len(self.emotion_encoder.classes_)}")
        logger.info(f"Scenarios: {len(self.scenario_encoder.classes_)}")
        logger.info(f"Models: {len(self.model_encoder.classes_)}")
    
    def tokenize_conversation(self, conversation_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize conversation turns"""
        
        max_length = self.config.get('max_seq_length', 128)
        max_turns = self.config.get('max_conversation_length', 50)
        
        # Truncate conversation if too long
        if len(conversation_texts) > max_turns:
            conversation_texts = conversation_texts[-max_turns:]
        
        # Tokenize each turn
        tokenized_turns = []
        attention_masks = []
        
        for text in conversation_texts:
            # Clean text to avoid out-of-vocabulary issues
            clean_text = text.replace('[TURN_SEP]', ' ').replace('[CONV_START]', ' ').replace('[CONV_END]', ' ')
            clean_text = clean_text.replace('[USER]', 'User:').replace('[AGENT]', 'Agent:')
            
            tokens = self.tokenizer(
                clean_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            # Validate token IDs are within vocab range
            token_ids = tokens['input_ids'].squeeze(0)
            vocab_size = self.tokenizer.vocab_size
            
            # Clip any token IDs that are out of range
            token_ids = torch.clamp(token_ids, 0, vocab_size - 1)
            
            tokenized_turns.append(token_ids)
            attention_masks.append(tokens['attention_mask'].squeeze(0))
        
        # Pad conversation to max_turns
        while len(tokenized_turns) < max_turns:
            # Add padding turn
            padding_tokens = torch.zeros_like(tokenized_turns[0])
            padding_mask = torch.zeros_like(attention_masks[0])
            
            tokenized_turns.append(padding_tokens)
            attention_masks.append(padding_mask)
        
        return {
            'input_ids': torch.stack(tokenized_turns),
            'attention_mask': torch.stack(attention_masks)
        }
    
    def process_conversation(self, conv_features: Dict) -> Dict[str, torch.Tensor]:
        """Process a single conversation into model inputs"""
        
        if not self.fitted:
            raise ValueError("Processor must be fitted before processing conversations")
        
        # Tokenize conversation
        tokenized = self.tokenize_conversation(conv_features['texts'])
        
        # Process temporal features
        temporal_features = []
        for i in range(len(conv_features['texts'])):
            temp_feats = [
                conv_features['response_times'][i],
                conv_features['turn_positions'][i],
                i / len(conv_features['texts']),  # Relative position
                len(conv_features['texts']),  # Conversation length context
            ]
            temporal_features.append(temp_feats)
        
        # Pad temporal features to max conversation length
        max_turns = self.config.get('max_conversation_length', 50)
        while len(temporal_features) < max_turns:
            temporal_features.append([0.0, 0.0, 0.0, 0.0])
        
        temporal_features = temporal_features[:max_turns]
        temporal_scaled = self.temporal_scaler.transform(temporal_features)
        
        # Process behavioral features
        behavioral_numeric = conv_features['behavioral_features'][:6]
        scenario = conv_features['behavioral_features'][6]
        model = conv_features['behavioral_features'][7]
        
        # Encode categorical features
        scenario_encoded = self.scenario_encoder.transform([scenario])[0]
        model_encoded = self.model_encoder.transform([model])[0]
        
        # Combine behavioral features
        behavioral_combined = behavioral_numeric + [scenario_encoded, model_encoded]
        
        # Ensure consistent feature count for behavioral features
        expected_behavioral_size = 8  # 6 numeric + 2 categorical
        if len(behavioral_combined) != expected_behavioral_size:
            # Pad or truncate to expected size
            if len(behavioral_combined) < expected_behavioral_size:
                behavioral_combined.extend([0.0] * (expected_behavioral_size - len(behavioral_combined)))
            else:
                behavioral_combined = behavioral_combined[:expected_behavioral_size]
        
        behavioral_scaled = self.behavioral_scaler.transform([behavioral_combined])[0]
        
        # Process emotions
        emotion_encoded = []
        for emotion in conv_features['emotions']:
            try:
                emotion_enc = self.emotion_encoder.transform([emotion])[0]
            except ValueError:
                emotion_enc = 0  # Unknown emotion
            emotion_encoded.append(emotion_enc)
        
        # Pad emotions
        while len(emotion_encoded) < max_turns:
            emotion_encoded.append(0)
        
        emotion_encoded = emotion_encoded[:max_turns]
        
        # Process trust scores and categories
        trust_scores = []
        trust_categories = []
        
        for i in range(len(conv_features['texts'])):
            if i < len(conv_features['trust_scores']) and conv_features['trust_scores'][i] is not None:
                trust_scores.append(conv_features['trust_scores'][i])
                trust_categories.append(conv_features['trust_categories'][i])
            else:
                trust_scores.append(0.0)
                trust_categories.append([0.0, 0.0, 0.0])
        
        # Pad to max turns
        while len(trust_scores) < max_turns:
            trust_scores.append(0.0)
            trust_categories.append([0.0, 0.0, 0.0])
        
        trust_scores = trust_scores[:max_turns]
        trust_categories = trust_categories[:max_turns]
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'temporal_features': torch.FloatTensor(temporal_scaled),
            'behavioral_features': torch.FloatTensor(behavioral_scaled),
            'trust_scores': torch.FloatTensor(trust_scores),
            'emotions': torch.LongTensor(emotion_encoded),
            'trust_categories': torch.FloatTensor(trust_categories),
            'conversation_length': len(conv_features['texts']),
            'trust_evolution': conv_features['trust_evolution']
        }
    
    def save(self, path: str):
        """Save fitted processor"""
        processor_data = {
            'config': self.config,
            'emotion_encoder': self.emotion_encoder,
            'scenario_encoder': self.scenario_encoder,
            'model_encoder': self.model_encoder,
            'temporal_scaler': self.temporal_scaler,
            'behavioral_scaler': self.behavioral_scaler,
            'fitted': self.fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(processor_data, f)
        
        # Also save tokenizer
        tokenizer_path = Path(path).parent / 'tokenizer'
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Processor saved to {path}")
    
    @classmethod
    def load(cls, path: str, config: Dict):
        """Load fitted processor"""
        processor = cls(config)
        
        with open(path, 'rb') as f:
            processor_data = pickle.load(f)
        
        processor.emotion_encoder = processor_data['emotion_encoder']
        processor.scenario_encoder = processor_data['scenario_encoder']
        processor.model_encoder = processor_data['model_encoder']
        processor.temporal_scaler = processor_data['temporal_scaler']
        processor.behavioral_scaler = processor_data['behavioral_scaler']
        processor.fitted = processor_data['fitted']
        
        # Load tokenizer
        tokenizer_path = Path(path).parent / 'tokenizer'
        if tokenizer_path.exists():
            processor.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"✓ Processor loaded from {path}")
        return processor

class ConTrustDataset(Dataset):
    """
    PyTorch Dataset for ConTrust model training
    """
    
    def __init__(self, conversation_features: Dict, processor: ConTrustDataProcessor, 
                 target_type: str = 'current'):
        self.conversation_features = conversation_features
        self.processor = processor
        self.target_type = target_type
        
        # Filter conversations that have trust scores
        self.valid_conversations = []
        for conv_id, features in conversation_features.items():
            if any(ts is not None for ts in features['trust_scores']):
                self.valid_conversations.append(conv_id)
        
        logger.info(f"Dataset created with {len(self.valid_conversations)} valid conversations")
    
    def __len__(self):
        return len(self.valid_conversations)
    
    def __getitem__(self, idx):
        conv_id = self.valid_conversations[idx]
        conv_features = self.conversation_features[conv_id]
        
        # Process conversation
        processed = self.processor.process_conversation(conv_features)
        
        # Determine target based on target_type
        if self.target_type == 'current':
            # Predict trust score of the last agent turn
            agent_trust_scores = [ts for ts in conv_features['trust_scores'] if ts is not None]
            if agent_trust_scores:
                target_trust = agent_trust_scores[-1]
                target_trust_cats = [tc for tc in conv_features['trust_categories'] 
                                   if tc != [0.0, 0.0, 0.0]][-1] if any(tc != [0.0, 0.0, 0.0] for tc in conv_features['trust_categories']) else [0.0, 0.0, 0.0]
            else:
                target_trust = 4.0  # Default middle value
                target_trust_cats = [0.0, 0.0, 0.0]
        
        elif self.target_type == 'evolution':
            # Predict trust evolution over conversation
            target_trust = np.mean(conv_features['trust_evolution']) if conv_features['trust_evolution'] else 4.0
            target_trust_cats = [0.0, 0.0, 0.0]  # Not used for evolution prediction
        
        # Engagement score (simplified)
        engagement = len(conv_features['trust_evolution']) / len(conv_features['texts']) if len(conv_features['texts']) > 0 else 0.5
        
        # Last emotion for prediction
        last_emotion_idx = len([e for e in conv_features['emotions'] if e != 'neutral']) - 1
        if last_emotion_idx >= 0:
            try:
                target_emotion = self.processor.emotion_encoder.transform([conv_features['emotions'][-1]])[0]
            except ValueError:
                target_emotion = 0
        else:
            target_emotion = 0
        
        # Filter out None values from trust scores for evolution target
        evolution_scores = [ts for ts in conv_features['trust_scores'] if ts is not None]
        if not evolution_scores:
            evolution_scores = [4.0]  # Default value if no trust scores
        
        # Create target dictionary
        targets = {
            'target_trust': torch.FloatTensor([target_trust]),
            'target_trust_categories': torch.FloatTensor(target_trust_cats),
            'target_emotion': torch.LongTensor([target_emotion]),
            'target_engagement': torch.FloatTensor([engagement]),
            'target_evolution': torch.FloatTensor(evolution_scores)
        }
        
        # Combine inputs and targets
        batch_item = {**processed, **targets}
        batch_item['conversation_lengths'] = torch.LongTensor([processed['conversation_length']])
        
        return batch_item

def create_contrust_dataloaders(data_dir: str, config: Dict, 
                               batch_size: int = 8, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, ConTrustDataProcessor]:
    """
    Create DataLoaders for ConTrust training
    """
    
    # Initialize processor
    processor = ConTrustDataProcessor(config)
    
    # Load and process data
    full_dataset, agent_data = processor.load_conversations(data_dir)
    conversation_features = processor.extract_conversation_features(full_dataset)
    
    # Fit encoders
    processor.fit_encoders(conversation_features)
    
    # Split conversations
    conv_ids = list(conversation_features.keys())
    np.random.seed(42)
    np.random.shuffle(conv_ids)
    
    split_idx = int(len(conv_ids) * (1 - val_split))
    train_conv_ids = conv_ids[:split_idx]
    val_conv_ids = conv_ids[split_idx:]
    
    train_features = {cid: conversation_features[cid] for cid in train_conv_ids}
    val_features = {cid: conversation_features[cid] for cid in val_conv_ids}
    
    # Create datasets
    train_dataset = ConTrustDataset(train_features, processor)
    val_dataset = ConTrustDataset(val_features, processor)
    
    # Create dataloaders with custom collate function
    def collate_fn(batch):
        """Custom collate function for variable length conversations"""
        
        # Find max conversation length in batch
        max_conv_len = max(item['conversation_length'] for item in batch)
        
        # Pad all items to max length
        padded_batch = {}
        
        for key in batch[0].keys():
            if key == 'conversation_length':
                padded_batch[key] = torch.LongTensor([item[key] for item in batch])
            elif key == 'conversation_lengths':
                padded_batch[key] = torch.cat([item[key] for item in batch])
            elif key.startswith('target_'):
                # Handle targets with special case for evolution
                if key == 'target_evolution':
                    # Pad evolution targets to same length
                    max_evo_len = max(len(item[key]) for item in batch)
                    padded_evos = []
                    for item in batch:
                        evo = item[key]
                        if len(evo) < max_evo_len:
                            # Pad with last value or 4.0 if empty
                            pad_val = evo[-1] if len(evo) > 0 else 4.0
                            padding = torch.full((max_evo_len - len(evo),), pad_val)
                            evo = torch.cat([evo, padding])
                        elif len(evo) > max_evo_len:
                            evo = evo[:max_evo_len]
                        padded_evos.append(evo)
                    padded_batch[key] = torch.stack(padded_evos)
                else:
                    # Stack other targets normally
                    if batch[0][key].dim() == 1:
                        padded_batch[key] = torch.cat([item[key] for item in batch])
                    else:
                        padded_batch[key] = torch.stack([item[key] for item in batch])
            else:
                # Stack other tensors - handle both tensors and lists/scalars
                first_item = batch[0][key]
                if torch.is_tensor(first_item):
                    if len(first_item.shape) > 1:
                        # Multi-dimensional tensor - truncate/pad to max_conv_len
                        padded_items = []
                        for item in batch:
                            tensor = item[key]
                            if tensor.size(0) > max_conv_len:
                                tensor = tensor[:max_conv_len]
                            elif tensor.size(0) < max_conv_len:
                                # Pad
                                pad_size = [0] * (2 * len(tensor.shape))
                                pad_size[-1] = max_conv_len - tensor.size(0)
                                tensor = F.pad(tensor, pad_size)
                            padded_items.append(tensor)
                        
                        padded_batch[key] = torch.stack(padded_items)
                    else:
                        # 1D tensor or scalar tensor
                        padded_batch[key] = torch.stack([item[key] for item in batch])
                elif isinstance(first_item, (list, tuple)):
                    # Convert lists to tensors
                    try:
                        padded_batch[key] = torch.stack([torch.tensor(item[key]) for item in batch])
                    except:
                        # If conversion fails, just pass as is
                        padded_batch[key] = [item[key] for item in batch]
                else:
                    # Scalar values
                    padded_batch[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        
        return padded_batch
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0  # Disable multiprocessing for Windows compatibility
    )
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader, processor

if __name__ == "__main__":
    # Test data processing
    from contrust_model import create_contrust_config
    
    config = create_contrust_config()
    
    try:
        train_loader, val_loader, processor = create_contrust_dataloaders(
            'data', config, batch_size=2
        )
        
        # Test a batch
        for batch in train_loader:
            print("✅ Batch processing successful!")
            print(f"Input IDs shape: {batch['input_ids'].shape}")
            print(f"Temporal features shape: {batch['temporal_features'].shape}")
            print(f"Behavioral features shape: {batch['behavioral_features'].shape}")
            print(f"Target trust shape: {batch['target_trust'].shape}")
            break
            
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
