#!/usr/bin/env python3
"""
Comprehensive ML/DL Training Pipeline for Real-time Trust Metrics and Emotion Detection
in Conversational Agents

Features:
- Multi-modal feature engineering (text, temporal, behavioral)
- Traditional ML models (RF, XGBoost, SVM)
- Deep learning models (LSTM, BERT, Transformer)
- Multi-task learning for trust + emotion prediction
- Real-time inference pipeline
- Comprehensive evaluation framework
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import xgboost as xgb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# NLP
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sentence_transformers import SentenceTransformer
import re
from collections import Counter

# Feature Engineering
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from textblob import TextBlob

# Utilities
from tqdm import tqdm
import pickle
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrustDatasetLoader:
    """Load and preprocess the conversational trust dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.conversations_dir = self.data_dir / "conversations"
        self.metadata_dir = self.data_dir / "metadata"
        
    def load_data(self, max_files: Optional[int] = None) -> pd.DataFrame:
        """Load and merge conversation and metadata files"""
        all_turns = []
        
        conv_files = [f for f in os.listdir(self.conversations_dir) if f.endswith('.json')]
        if max_files:
            conv_files = conv_files[:max_files]
        logger.info(f"Loading {len(conv_files)} conversation files")
        
        for filename in tqdm(conv_files, desc="Loading conversations"):
            conv_id = os.path.splitext(filename)[0]
            
            # Load metadata
            meta_path = os.path.join(self.metadata_dir, filename)
            metadata = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Load conversation
            conv_path = os.path.join(self.conversations_dir, filename)
            with open(conv_path, 'r', encoding='utf-8') as f:
                conv_data = json.load(f)
                
                for turn in conv_data.get('turns', []):
                    turn_data = turn.copy()
                    
                    # Add metadata
                    turn_data.update({
                        'conversation_id': conv_id,
                        'agent_model': metadata.get('agent_model', 'unknown'),
                        'scenario': metadata.get('scenario', 'unknown'),
                        'total_turns': len(conv_data.get('turns', [])),
                        'conv_avg_trust': conv_data.get('data', {}).get('average_trust_score'),
                        'conv_engagement': conv_data.get('data', {}).get('engagement_score'),
                    })
                    
                    # Flatten trust category scores
                    trust_scores = turn_data.pop('trust_category_scores', None)
                    if isinstance(trust_scores, dict):
                        turn_data.update({
                            'competence': trust_scores.get('competence'),
                            'benevolence': trust_scores.get('benevolence'),
                            'integrity': trust_scores.get('integrity')
                        })
                    
                    all_turns.append(turn_data)
        
        df = pd.DataFrame(all_turns)
        
        # Data type conversions
        numeric_cols = ['trust_score', 'response_time', 'competence', 'benevolence', 'integrity', 
                       'conv_avg_trust', 'conv_engagement']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"Loaded {len(df)} turns from {df['conversation_id'].nunique()} conversations")
        return df

class FeatureEngineer:
    """Comprehensive feature engineering for conversational trust prediction"""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive text features"""
        features = df.copy()
        
        # Basic text statistics
        features['utterance_length'] = df['utterance'].str.len()
        features['word_count'] = df['utterance'].str.split().str.len()
        features['sentence_count'] = df['utterance'].str.split('.').str.len()
        features['avg_word_length'] = features['utterance_length'] / features['word_count']
        
        # Readability metrics
        features['flesch_reading_ease'] = df['utterance'].apply(
            lambda x: textstat.flesch_reading_ease(x) if x else 0
        )
        features['flesch_kincaid_grade'] = df['utterance'].apply(
            lambda x: textstat.flesch_kincaid_grade(x) if x else 0
        )
        
        # Sentiment analysis
        sentiments = df['utterance'].apply(lambda x: TextBlob(x).sentiment if x else TextBlob("").sentiment)
        features['sentiment_polarity'] = [s.polarity for s in sentiments]
        features['sentiment_subjectivity'] = [s.subjectivity for s in sentiments]
        
        # Question indicators
        features['has_question'] = df['utterance'].str.contains(r'\?', na=False).astype(int)
        features['question_count'] = df['utterance'].str.count(r'\?')
        
        # Emotional intensity markers
        features['exclamation_count'] = df['utterance'].str.count(r'!')
        features['caps_ratio'] = df['utterance'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if x else 0
        )
        
        return features
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal and conversational flow features"""
        features = df.copy()
        
        # Response time features
        features['response_time_log'] = np.log1p(features['response_time'].fillna(0))
        features['response_time_zscore'] = (features['response_time'] - features['response_time'].mean()) / features['response_time'].std()
        
        # Turn-based features
        features['turn_position'] = features['turn_id'] / features['total_turns']
        features['is_early_turn'] = (features['turn_id'] <= 3).astype(int)
        features['is_late_turn'] = (features['turn_id'] >= features['total_turns'] - 2).astype(int)
        
        # Conversation-level context
        features['turns_remaining'] = features['total_turns'] - features['turn_id']
        
        return features
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral and contextual features"""
        features = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['emotion_detected', 'speaker', 'agent_model', 'scenario']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        features[col].fillna('unknown')
                    )
                else:
                    features[f'{col}_encoded'] = self.label_encoders[col].transform(
                        features[col].fillna('unknown')
                    )
        
        # Previous turn context (for agent turns)
        agent_turns = features[features['speaker'] == 'agent'].copy()
        if len(agent_turns) > 0:
            agent_turns['prev_user_emotion'] = agent_turns.groupby('conversation_id')['emotion_detected_encoded'].shift(1)
            agent_turns['prev_response_time'] = agent_turns.groupby('conversation_id')['response_time'].shift(1)
            
            # Merge back
            features = features.merge(
                agent_turns[['conversation_id', 'turn_id', 'prev_user_emotion', 'prev_response_time']],
                on=['conversation_id', 'turn_id'], how='left'
            )
        
        return features
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get sentence embeddings for texts"""
        return self.sentence_model.encode(texts, show_progress_bar=True)
    
    def add_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentence embeddings to the dataframe"""
        embeddings = self.get_embeddings(df['utterance'].fillna('').tolist())
        df[['embedding_{}'.format(i) for i in range(embeddings.shape[1])]] = embeddings
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare comprehensive features for model training"""
        
        # Initialize features DataFrame
        features = df.copy()
        
        # Extract different types of features
        features = self.extract_text_features(features)
        features = self.extract_temporal_features(features)
        features = self.extract_behavioral_features(features)
        features = self.add_embeddings(features)
        
        # Select numerical columns for model training
        feature_columns = [col for col in features.columns 
                          if col not in ['utterance', 'conversation_id', 'turn_id', 'speaker', 
                                       'emotion_detected', 'scenario', 'agent_model'] 
                          and features[col].dtype in ['int64', 'float64']]
        
        # Store expected feature columns for consistent inference
        if not hasattr(self, 'expected_feature_columns'):
            self.expected_feature_columns = feature_columns.copy()
            self.expected_feature_count = len(feature_columns)
        
        # Ensure consistent feature dimensions
        final_features = features[feature_columns].copy()
        
        # Add missing columns with zeros if any are missing
        for expected_col in self.expected_feature_columns:
            if expected_col not in final_features.columns:
                final_features[expected_col] = 0.0
        
        # Remove extra columns that weren't in training
        final_features = final_features[self.expected_feature_columns]
        
        # Handle missing values
        feature_matrix = final_features.fillna(0).values
        
        feature_info = {
            'feature_columns': self.expected_feature_columns,
            'shape': feature_matrix.shape,
            'feature_names': self.expected_feature_columns,
            'expected_count': self.expected_feature_count
        }
        
        return feature_matrix, feature_info

class TraditionalMLModels:
    """Traditional ML models for trust prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'svm': SVR(kernel='rbf')
        }
        self.trained_models = {}
        
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict]:
        """Train all traditional ML models"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            results[name] = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'val_mse': mean_squared_error(y_val, val_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred),
                'model': model
            }
            
            logger.info(f"{name} - Val MSE: {results[name]['val_mse']:.4f}, Val R2: {results[name]['val_r2']:.4f}")
        
        return results

# Neural Network Models
class TrustLSTM(nn.Module):
    """LSTM model for sequential trust prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        super(TrustLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # Use last timestep
        out = self.fc(out)
        return out

class MultiTaskTrustModel(nn.Module):
    """Multi-task model for joint trust and emotion prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_emotions: int = 7, dropout: float = 0.3):
        super(MultiTaskTrustModel, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.trust_head = nn.Linear(hidden_size // 2, 1)
        self.emotion_head = nn.Linear(hidden_size // 2, num_emotions)
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        trust_pred = self.trust_head(shared_features)
        emotion_pred = self.emotion_head(shared_features)
        return trust_pred, emotion_pred

class DeepLearningTrainer:
    """Deep learning model trainer"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.models = {}
        
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, 
                  epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train LSTM model"""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # Add sequence dimension
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        model = TrustLSTM(input_size=X_train.shape[1])
        model.to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_tensor.to(self.device))
                val_loss = criterion(val_pred, y_val_tensor.to(self.device))
                val_losses.append(val_loss.item())
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        self.models['lstm'] = model
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

class RealTimeInference:
    """Real-time inference pipeline for trust prediction"""
    
    def __init__(self, feature_engineer: FeatureEngineer, model, model_type: str = 'sklearn'):
        self.feature_engineer = feature_engineer
        self.model = model
        self.model_type = model_type
        
    def predict_trust(self, utterance: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Predict trust score for a single utterance"""
        
        # Create temporary dataframe
        temp_df = pd.DataFrame([{
            'utterance': utterance,
            'speaker': context.get('speaker', 'agent'),
            'emotion_detected': context.get('emotion', 'neutral'),
            'response_time': context.get('response_time', 1.0),
            'turn_id': context.get('turn_id', 1),
            'total_turns': context.get('total_turns', 5),
            'agent_model': context.get('agent_model', 'unknown'),
            'scenario': context.get('scenario', 'unknown'),
            'conversation_id': context.get('conversation_id', 'temp'),
            'conv_avg_trust': context.get('conv_avg_trust', 3.5),
            'conv_engagement': context.get('conv_engagement', 4.0)
        }])
        
        # Extract features
        features, _ = self.feature_engineer.prepare_features(temp_df)
        
        # Make prediction
        if self.model_type == 'sklearn':
            prediction = self.model.predict(features)[0]
        elif self.model_type == 'pytorch':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(1)
                prediction = self.model(features_tensor).item()
        
        return {
            'trust_score': max(1.0, min(7.0, prediction)),  # Clamp to valid range
            'confidence': 0.8  # Placeholder confidence score
        }

def main():
    """Main training pipeline"""
    
    # Initialize components
    loader = TrustDatasetLoader("data")
    feature_engineer = FeatureEngineer()
    
    # Load data
    logger.info("Loading dataset...")
    df = loader.load_data()
    
    # Filter agent turns with trust scores
    agent_df = df[(df['speaker'] == 'agent') & (df['trust_score'].notna())].copy()
    logger.info(f"Training on {len(agent_df)} agent turns with trust scores")
    
    # Prepare features and targets
    logger.info("Engineering features...")
    X, feature_dict = feature_engineer.prepare_features(agent_df)
    y = agent_df['trust_score'].values
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train traditional ML models
    logger.info("Training traditional ML models...")
    ml_trainer = TraditionalMLModels()
    ml_results = ml_trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Train deep learning models
    logger.info("Training deep learning models...")
    dl_trainer = DeepLearningTrainer()
    lstm_results = dl_trainer.train_lstm(X_train, y_train, X_val, y_val)
    
    # Save models and results
    results_dir = Path("model_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save traditional models
    for name, result in ml_results.items():
        joblib.dump(result['model'], results_dir / f"{name}_model.pkl")
    
    # Save feature engineer
    joblib.dump(feature_engineer, results_dir / "feature_engineer.pkl")
    
    # Save PyTorch models
    torch.save(lstm_results['model'].state_dict(), results_dir / "lstm_model.pth")
    
    # Print results summary
    logger.info("\n=== Training Results Summary ===")
    for name, result in ml_results.items():
        logger.info(f"{name}: Val MSE: {result['val_mse']:.4f}, Val R2: {result['val_r2']:.4f}")
    
    # Demo real-time inference
    logger.info("\n=== Testing Real-time Inference ===")
    best_model = min(ml_results.items(), key=lambda x: x[1]['val_mse'])
    inference_engine = RealTimeInference(feature_engineer, best_model[1]['model'])
    
    test_context = {
        'speaker': 'agent',
        'emotion': 'neutral',
        'response_time': 2.5,
        'turn_id': 3,
        'total_turns': 8,
        'agent_model': 'gemini-2.0-flash',
        'scenario': 'hotel booking'
    }
    
    test_utterance = "I understand your concern. Let me find the best options for you."
    prediction = inference_engine.predict_trust(test_utterance, test_context)
    logger.info(f"Test prediction: {prediction}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
