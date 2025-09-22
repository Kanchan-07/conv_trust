#!/usr/bin/env python3
"""
Quick script to retrain models with consistent feature dimensions
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

from train_trust_models import TrustDatasetLoader, FeatureEngineer, TraditionalMLModels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_and_retrain_models():
    """Retrain models with consistent feature engineering"""
    
    print("🔧 Fixing feature dimensions and retraining models...")
    
    # Load data
    loader = TrustDatasetLoader('data')
    dataset = loader.load_data(max_files=500)  # Use subset for quick fix
    agent_data = dataset[(dataset['speaker'] == 'agent') & (dataset['trust_score'].notna())].copy()
    
    print(f"Using {len(agent_data)} agent turns for training")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Extract features (this will set expected_feature_columns)
    X, feature_info = fe.prepare_features(agent_data)
    y = agent_data['trust_score'].values
    
    print(f"✓ Training features shape: {X.shape}")
    print(f"✓ Expected feature count: {getattr(fe, 'expected_feature_count', len(feature_info['feature_columns']))}")
    
    # Test inference consistency
    test_df = pd.DataFrame([{
        'utterance': 'Test message for consistency check',
        'conversation_id': 'test_001',
        'turn_id': 1,
        'speaker': 'agent',
        'emotion_detected': agent_data['emotion_detected'].iloc[0],
        'response_time': 2.0,
        'scenario': agent_data['scenario'].iloc[0],
        'agent_model': agent_data['agent_model'].iloc[0],
        'total_turns': 5
    }])
    
    X_test, _ = fe.prepare_features(test_df)
    print(f"✓ Test inference shape: {X_test.shape}")
    
    if X.shape[1] == X_test.shape[1]:
        print("✅ Feature dimensions are consistent!")
    else:
        print(f"❌ Still inconsistent: {X.shape[1]} vs {X_test.shape[1]}")
        return False
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test_split, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Train set: {X_train.shape}, Test set: {X_test_split.shape}")
    
    # Train models
    ml_trainer = TraditionalMLModels()
    
    models_dir = Path("pipeline_results/trained_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and save each model
    for name, model in ml_trainer.models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Test prediction
        pred = model.predict(X_test_split)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        print(f"  {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Save model
        model_path = models_dir / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")
    
    # Save feature engineer
    fe_path = models_dir / "feature_engineer.pkl"
    joblib.dump(fe, fe_path)
    print(f"✓ Saved feature engineer: {fe_path}")
    
    # Test final inference consistency
    print("\n🔬 Testing inference consistency...")
    fe_loaded = joblib.load(fe_path)
    X_final_test, _ = fe_loaded.prepare_features(test_df)
    
    if X_final_test.shape[1] == X.shape[1]:
        print("✅ Final consistency check passed!")
        return True
    else:
        print(f"❌ Final check failed: {X_final_test.shape[1]} vs {X.shape[1]}")
        return False

if __name__ == "__main__":
    success = fix_and_retrain_models()
    if success:
        print("\n🎉 Models fixed and retrained successfully!")
        print("You can now run: uv run trust_inference.py")
    else:
        print("\n❌ Fix failed. Check the code for issues.")
