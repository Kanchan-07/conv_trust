#!/usr/bin/env python3
"""
Standalone Trust Inference Script
Load pre-trained models and perform real-time trust prediction without retraining
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
from pathlib import Path
import joblib
import torch
from typing import Dict, Any, List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class TrustInferenceEngine:
    """Standalone trust inference engine using pre-trained models"""
    
    def __init__(self, models_dir: str = "pipeline_results/trained_models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_engineer = None
        self.available_scenarios = []
        self.available_models = []
        self.available_emotions = []
        
        # Load all components
        self.load_models()
        self.load_metadata()
    
    def load_models(self):
        """Load all pre-trained models and feature engineer"""
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")
        
        logger.info("Loading pre-trained models...")
        
        # Load feature engineer
        feature_engineer_path = self.models_dir / "feature_engineer.pkl"
        if feature_engineer_path.exists():
            self.feature_engineer = joblib.load(feature_engineer_path)
            logger.info("✓ Feature engineer loaded")
        else:
            raise FileNotFoundError("Feature engineer not found. Please train models first.")
        
        # Load traditional ML models
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'svm': 'svm_model.pkl',
            'ensemble': 'ensemble_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"✓ {model_name} model loaded")
        
        # Load LSTM model if available
        lstm_path = self.models_dir / "lstm_model.pth"
        if lstm_path.exists():
            try:
                from train_trust_models import DeepLearningTrainer
                # Create LSTM model architecture
                dl_trainer = DeepLearningTrainer()
                lstm_model = dl_trainer.create_lstm_model(409)  # Assuming 409 features
                lstm_model.load_state_dict(torch.load(lstm_path, map_location='cpu'))
                lstm_model.eval()
                self.models['lstm'] = lstm_model
                logger.info("✓ LSTM model loaded")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e}")
        
        if not self.models:
            raise RuntimeError("No models loaded successfully")
        
        logger.info(f"Loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def load_metadata(self):
        """Load training metadata to get valid categories"""
        try:
            # Try to load from training data if available
            from train_trust_models import TrustDatasetLoader
            
            if os.path.exists("data"):
                loader = TrustDatasetLoader("data")
                # Load a small sample to get valid categories
                sample_data = loader.load_data(max_files=100)  # Load sample to get categories
                
                self.available_scenarios = sample_data['scenario'].unique().tolist()
                self.available_models = sample_data['agent_model'].unique().tolist()
                self.available_emotions = sample_data['emotion_detected'].unique().tolist()
                
                logger.info(f"✓ Metadata loaded: {len(self.available_scenarios)} scenarios, "
                          f"{len(self.available_models)} models, {len(self.available_emotions)} emotions")
            else:
                # Use default values if data not available
                self.available_scenarios = ['general', 'customer_service', 'technical_support']
                self.available_models = ['default_model']
                self.available_emotions = ['neutral', 'positive', 'negative']
                logger.warning("Using default metadata values")
        
        except Exception as e:
            # Fallback to defaults
            self.available_scenarios = ['general', 'customer_service', 'technical_support']
            self.available_models = ['default_model']
            self.available_emotions = ['neutral', 'positive', 'negative']
            logger.warning(f"Could not load metadata, using defaults: {e}")
    
    def validate_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix context to use known categories"""
        validated_context = context.copy()
        
        # Validate scenario
        if 'scenario' in validated_context:
            if validated_context['scenario'] not in self.available_scenarios:
                validated_context['scenario'] = self.available_scenarios[0]
        else:
            validated_context['scenario'] = self.available_scenarios[0]
        
        # Validate agent model
        if 'agent_model' in validated_context:
            if validated_context['agent_model'] not in self.available_models:
                validated_context['agent_model'] = self.available_models[0]
        else:
            validated_context['agent_model'] = self.available_models[0]
        
        # Validate emotion
        if 'emotion' in validated_context:
            if validated_context['emotion'] not in self.available_emotions:
                validated_context['emotion'] = self.available_emotions[0]
        else:
            validated_context['emotion'] = self.available_emotions[0]
        
        # Set defaults for missing fields
        defaults = {
            'speaker': 'agent',
            'response_time': 2.0,
            'turn_id': 1,
            'total_turns': 5
        }
        
        for key, default_value in defaults.items():
            if key not in validated_context:
                validated_context[key] = default_value
        
        return validated_context
    
    def predict_trust(self, utterance: str, context: Optional[Dict[str, Any]] = None, 
                     model_name: str = 'ensemble') -> Dict[str, Any]:
        """
        Predict trust score for a given utterance and context
        
        Args:
            utterance: The text to analyze
            context: Context information (scenario, model, etc.)
            model_name: Which model to use for prediction
            
        Returns:
            Dict with trust_score, confidence, and model_used
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            logger.warning(f"Model {model_name} not available. Using {available_models[0]}")
            model_name = available_models[0]
        
        # Use default context if none provided
        if context is None:
            context = {
                'speaker': 'agent',
                'emotion': 'neutral',
                'response_time': 2.0,
                'turn_id': 1,
                'total_turns': 5,
                'scenario': self.available_scenarios[0],
                'agent_model': self.available_models[0]
            }
        
        # Validate context
        context = self.validate_context(context)
        
        # Create temporary DataFrame for feature extraction
        temp_df = pd.DataFrame([{
            'utterance': utterance,
            'conversation_id': 'inference_temp',
            'turn_id': context.get('turn_id', 1),
            'speaker': context.get('speaker', 'agent'),
            'emotion_detected': context.get('emotion', 'neutral'),
            'response_time': context.get('response_time', 2.0),
            'scenario': context.get('scenario', self.available_scenarios[0]),
            'agent_model': context.get('agent_model', self.available_models[0]),
            'total_turns': context.get('total_turns', 5)
        }])
        
        try:
            # Extract features
            features, feature_info = self.feature_engineer.prepare_features(temp_df)
            
            # Debug feature dimensions
            logger.debug(f"Features shape: {features.shape}, Expected: {getattr(self.feature_engineer, 'expected_feature_count', 'unknown')}")
            
            # Get model and predict
            model = self.models[model_name]
            
            if model_name == 'lstm':
                # Handle LSTM prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(features).unsqueeze(1)
                    prediction = model(X_tensor).squeeze().numpy()
                    if len(prediction.shape) == 0:
                        prediction = float(prediction)
                    else:
                        prediction = float(prediction[0])
            else:
                # Handle sklearn models
                prediction = model.predict(features)
                if hasattr(prediction, '__len__') and len(prediction) > 0:
                    prediction = float(prediction[0])
                else:
                    prediction = float(prediction)
            
            # Calculate confidence (simplified)
            confidence = min(1.0, 1.0 - abs(prediction - 4.0) / 3.0)  # Higher confidence near middle values
            
            return {
                'trust_score': round(prediction, 2),
                'confidence': round(confidence, 2),
                'model_used': model_name,
                'features_extracted': features.shape[1] if hasattr(features, 'shape') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'trust_score': 4.0,  # Default middle value
                'confidence': 0.1,   # Low confidence
                'model_used': model_name,
                'error': str(e)
            }
    
    def predict_batch(self, conversations: List[Dict[str, Any]], 
                     model_name: str = 'ensemble') -> List[Dict[str, Any]]:
        """Predict trust scores for multiple conversations"""
        results = []
        
        for conv in conversations:
            utterance = conv.get('utterance', '')
            context = conv.get('context', {})
            
            result = self.predict_trust(utterance, context, model_name)
            result['input'] = conv
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'available_models': list(self.models.keys()),
            'feature_dimensions': getattr(self.feature_engineer, 'feature_count', 'unknown'),
            'available_scenarios': self.available_scenarios,
            'available_agent_models': self.available_models,
            'available_emotions': self.available_emotions
        }
    
    def demo_predictions(self):
        """Run demonstration predictions"""
        print("\n" + "="*70)
        print("🤖 TRUST PREDICTION DEMONSTRATION")
        print("="*70)
        
        demo_conversations = [
            {
                'utterance': "I understand your concern and I'll help you find the best solution for you.",
                'context': {
                    'emotion': 'neutral',
                    'response_time': 2.1,
                    'scenario': self.available_scenarios[0] if self.available_scenarios else 'general'
                }
            },
            {
                'utterance': "I apologize for the confusion. Let me clarify that information right away.",
                'context': {
                    'emotion': 'neutral', 
                    'response_time': 1.8,
                    'scenario': self.available_scenarios[1] if len(self.available_scenarios) > 1 else self.available_scenarios[0]
                }
            },
            {
                'utterance': "I'm not sure about that. You might want to check with someone else.",
                'context': {
                    'emotion': 'neutral',
                    'response_time': 4.2,
                    'scenario': self.available_scenarios[2] if len(self.available_scenarios) > 2 else self.available_scenarios[0]
                }
            },
            {
                'utterance': "Thank you for your patience! I've found exactly what you need.",
                'context': {
                    'emotion': 'positive',
                    'response_time': 1.5,
                    'scenario': self.available_scenarios[0] if self.available_scenarios else 'general'
                }
            }
        ]
        
        for i, conv in enumerate(demo_conversations, 1):
            print(f"\n📝 Scenario {i}:")
            print(f"Agent says: \"{conv['utterance']}\"")
            print(f"Context: {conv['context']['scenario']} (Response time: {conv['context']['response_time']}s)")
            
            # Predict with different models
            for model_name in self.models.keys():
                result = self.predict_trust(conv['utterance'], conv['context'], model_name)
                print(f"  {model_name:12s}: Trust={result['trust_score']:.2f}/7.0 (confidence: {result['confidence']:.2f})")
            
            print("-" * 70)

def main():
    """Main function for standalone execution"""
    print("🤖 Trust Inference Engine")
    print("=" * 50)
    
    try:
        # Initialize inference engine
        engine = TrustInferenceEngine()
        
        # Show model info
        info = engine.get_model_info()
        print(f"✓ Loaded models: {', '.join(info['available_models'])}")
        print(f"✓ Feature dimensions: {info['feature_dimensions']}")
        print(f"✓ Available scenarios: {len(info['available_scenarios'])}")
        
        # Run demonstration
        engine.demo_predictions()
        
        print("\n" + "="*50)
        print("🎉 Inference engine ready!")
        print("You can now use this engine for real-time trust prediction!")
        print("="*50)
        
        return engine
        
    except Exception as e:
        print(f"❌ Failed to initialize inference engine: {e}")
        print("Make sure you have run the training pipeline first to generate models.")
        return None

# Interactive prediction function
def predict_interactive():
    """Interactive prediction mode"""
    try:
        engine = TrustInferenceEngine()
        
        print("\n🤖 Interactive Trust Prediction Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            utterance = input("\nEnter agent utterance: ").strip()
            if utterance.lower() in ['quit', 'exit', 'q']:
                break
            
            if not utterance:
                continue
            
            # Simple prediction with default context
            result = engine.predict_trust(utterance)
            
            print(f"Predicted Trust Score: {result['trust_score']:.2f}/7.0")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Model used: {result['model_used']}")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trust Inference Engine')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--demo', '-d', action='store_true', 
                       help='Run demonstration (default)')
    
    args = parser.parse_args()
    
    if args.interactive:
        predict_interactive()
    else:
        main()
