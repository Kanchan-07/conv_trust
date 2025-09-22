#!/usr/bin/env python3
"""
Complete ML/DL Pipeline Execution for Trust and Emotion Detection
Integrates all components: data loading, feature engineering, model training, evaluation, and real-time inference
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
from datetime import datetime

# Import our custom modules
from train_trust_models import (
    TrustDatasetLoader, FeatureEngineer, TraditionalMLModels, 
    DeepLearningTrainer, RealTimeInference
)
from advanced_bert_models import (
    BERTTrustRegressor, BERTTrainer, prepare_bert_data,
    ConversationalTrustDataset
)
from evaluation_framework import (
    TrustModelEvaluator, ModelEnsemble, create_evaluation_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CompletePipeline:
    """Complete pipeline for trust and emotion detection model training and evaluation"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "pipeline_results"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.loader = TrustDatasetLoader(data_dir)
        self.feature_engineer = FeatureEngineer()
        self.evaluator = TrustModelEvaluator(str(self.output_dir / "evaluation"))
        
        # Storage for results
        self.trained_models = {}
        self.model_results = {}
        self.dataset = None
        
    def load_and_prepare_data(self):
        """Load and prepare the complete dataset"""
        logger.info("Loading and preparing dataset...")
        
        # Load raw data
        self.dataset = self.loader.load_data()
        logger.info(f"Loaded {len(self.dataset)} total turns")
        
        # Filter agent turns with trust scores for training
        self.agent_data = self.dataset[
            (self.dataset['speaker'] == 'agent') & 
            (self.dataset['trust_score'].notna())
        ].copy()
        
        logger.info(f"Training on {len(self.agent_data)} agent turns with trust scores")
        
        # Dataset statistics
        stats = {
            'total_turns': len(self.dataset),
            'agent_turns_with_trust': len(self.agent_data),
            'conversations': self.dataset['conversation_id'].nunique(),
            'scenarios': self.dataset['scenario'].nunique(),
            'models': self.dataset['agent_model'].nunique(),
            'trust_score_range': (
                self.agent_data['trust_score'].min(), 
                self.agent_data['trust_score'].max()
            ),
            'avg_trust_score': self.agent_data['trust_score'].mean()
        }
        
        logger.info(f"Dataset statistics: {stats}")
        return stats
    
    def prepare_features(self):
        """Prepare features for training"""
        logger.info("Engineering features...")
        
        # Extract comprehensive features
        self.X, self.feature_dict = self.feature_engineer.prepare_features(self.agent_data)
        self.y = self.agent_data['trust_score'].values
        
        logger.info(f"Feature matrix shape: {self.X.shape}")
        logger.info(f"Target variable shape: {self.y.shape}")
        
        # Train-validation-test split
        from sklearn.model_selection import train_test_split
        
        # First split: 80% train+val, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Second split: 80% train, 20% val from the temp data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        logger.info(f"Train set: {self.X_train.shape}, Val set: {self.X_val.shape}, Test set: {self.X_test.shape}")
        
    def train_traditional_models(self):
        """Train traditional ML models"""
        logger.info("Training traditional ML models...")
        
        ml_trainer = TraditionalMLModels()
        ml_results = ml_trainer.train_models(
            self.X_train, self.y_train, self.X_val, self.y_val
        )
        
        # Evaluate on test set
        for model_name, result in ml_results.items():
            model = result['model']
            test_pred = model.predict(self.X_test)
            
            # Evaluate
            test_metrics = self.evaluator.evaluate_regression_model(
                self.y_test, test_pred, f"{model_name}_traditional"
            )
            
            # Store results
            self.trained_models[model_name] = model
            self.model_results[model_name] = test_metrics
            
            # Create visualizations
            self.evaluator.plot_regression_results(
                self.y_test, test_pred, f"{model_name}_traditional"
            )
            
            logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}, Test RMSE: {test_metrics['rmse']:.4f}")
    
    def train_deep_learning_models(self):
        """Train deep learning models"""
        logger.info("Training deep learning models...")
        
        # LSTM Model
        dl_trainer = DeepLearningTrainer()
        lstm_result = dl_trainer.train_lstm(
            self.X_train, self.y_train, self.X_val, self.y_val, epochs=50
        )
        
        # Evaluate LSTM on test set
        lstm_model = lstm_result['model']
        lstm_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).unsqueeze(1)
            lstm_test_pred = lstm_model(X_test_tensor).squeeze().numpy()
        
        lstm_metrics = self.evaluator.evaluate_regression_model(
            self.y_test, lstm_test_pred, "lstm_deep"
        )
        
        self.trained_models['lstm'] = lstm_model
        self.model_results['lstm'] = lstm_metrics
        
        self.evaluator.plot_regression_results(
            self.y_test, lstm_test_pred, "lstm_deep"
        )
        
        logger.info(f"LSTM - Test R²: {lstm_metrics['r2']:.4f}, Test RMSE: {lstm_metrics['rmse']:.4f}")
    
    def train_bert_model(self):
        """Train BERT-based model (lightweight version for demo)"""
        try:
            logger.info("Training BERT-based model...")
            
            from transformers import AutoTokenizer
            
            # Use lightweight DistilBERT for faster training
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            bert_model = BERTTrustRegressor('distilbert-base-uncased')
            
            # Prepare BERT data (subset for demo)
            bert_data = self.agent_data.sample(min(1000, len(self.agent_data))).copy()
            
            train_dataset, val_dataset = prepare_bert_data(bert_data, tokenizer)
            
            # Train BERT model
            bert_trainer = BERTTrainer(bert_model, tokenizer)
            bert_result = bert_trainer.train_trust_regressor(
                train_dataset, val_dataset, epochs=2, batch_size=8  # Reduced for demo
            )
            
            # Simple evaluation (using a subset due to computational constraints)
            sample_texts = self.agent_data['utterance'].head(100).tolist()
            bert_predictions = []
            
            for text in sample_texts:
                pred = bert_trainer.predict(text)
                bert_predictions.append(pred)
            
            bert_true = self.agent_data['trust_score'].head(100).values
            bert_metrics = self.evaluator.evaluate_regression_model(
                bert_true, np.array(bert_predictions), "bert_distil"
            )
            
            self.trained_models['bert'] = bert_model
            self.model_results['bert'] = bert_metrics
            
            logger.info(f"BERT - Test R²: {bert_metrics['r2']:.4f}, Test RMSE: {bert_metrics['rmse']:.4f}")
            
        except Exception as e:
            logger.warning(f"BERT training failed: {e}. Skipping BERT model.")
    
    def create_ensemble(self):
        """Create and evaluate model ensemble"""
        logger.info("Creating model ensemble...")
        
        # Use traditional models for ensemble (more reliable)
        ensemble_models = {
            name: model for name, model in self.trained_models.items() 
            if name in ['random_forest', 'xgboost', 'svm']
        }
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble. Skipping ensemble creation.")
            return
        
        # Create ensemble
        ensemble = ModelEnsemble(ensemble_models)
        
        # Optimize weights on validation set
        optimized_weights = ensemble.optimize_weights(self.X_val, self.y_val)
        
        # Evaluate ensemble on test set
        ensemble_pred = ensemble.predict(self.X_test)
        ensemble_metrics = self.evaluator.evaluate_regression_model(
            self.y_test, ensemble_pred, "ensemble"
        )
        
        self.trained_models['ensemble'] = ensemble
        self.model_results['ensemble'] = ensemble_metrics
        
        self.evaluator.plot_regression_results(
            self.y_test, ensemble_pred, "ensemble"
        )
        
        logger.info(f"Ensemble - Test R²: {ensemble_metrics['r2']:.4f}, Test RMSE: {ensemble_metrics['rmse']:.4f}")
        logger.info(f"Optimized weights: {optimized_weights}")
    
    def evaluate_and_compare_models(self):
        """Compare all models and create comprehensive evaluation"""
        logger.info("Creating comprehensive model comparison...")
        
        # Model comparison
        comparison_df = self.evaluator.compare_models(self.model_results)
        self.evaluator.plot_model_comparison(comparison_df)
        
        # Feature importance for tree-based models
        if 'random_forest' in self.trained_models:
            # Create feature names (simplified)
            feature_names = [f'feature_{i}' for i in range(self.X.shape[1])]
            self.evaluator.analyze_feature_importance(
                self.trained_models['random_forest'], feature_names, 'random_forest'
            )
        
        # Create comprehensive report
        dataset_info = {
            'name': 'Conversational Trust Dataset',
            'total_samples': len(self.y_test),
            'num_features': self.X.shape[1]
        }
        
        report = create_evaluation_report(self.evaluator, self.model_results, dataset_info)
        print("\n" + "="*80)
        print("FINAL EVALUATION REPORT")
        print("="*80)
        print(report)
        
        return comparison_df
    
    def demonstrate_real_time_inference(self):
        """Demonstrate real-time inference capabilities"""
        logger.info("Demonstrating real-time inference...")
        
        # Get best performing model
        best_model_name = min(self.model_results.items(), key=lambda x: x[1]['rmse'])[0]
        best_model = self.trained_models[best_model_name]
        
        logger.info(f"Using {best_model_name} for real-time demo")
        
        # Create real-time inference engine
        inference_engine = RealTimeInference(self.feature_engineer, best_model)
        
        # Get actual scenarios from training data to avoid unseen label errors
        available_scenarios = self.agent_data['scenario'].unique()[:3]
        available_models = self.agent_data['agent_model'].unique()[:1]
        
        # Demo scenarios using actual data values
        demo_scenarios = [
            {
                'utterance': "I understand your concern and I'll help you find the best solution.",
                'context': {
                    'speaker': 'agent',
                    'emotion': 'neutral',
                    'response_time': 2.1,
                    'turn_id': 3,
                    'total_turns': 8,
                    'agent_model': available_models[0],
                    'scenario': available_scenarios[0]
                }
            },
            {
                'utterance': "I apologize for the confusion. Let me clarify that for you immediately.",
                'context': {
                    'speaker': 'agent',
                    'emotion': 'neutral',
                    'response_time': 1.8,
                    'turn_id': 5,
                    'total_turns': 10,
                    'agent_model': available_models[0],
                    'scenario': available_scenarios[1] if len(available_scenarios) > 1 else available_scenarios[0]
                }
            },
            {
                'utterance': "I'm not sure about that. You might want to check with someone else.",
                'context': {
                    'speaker': 'agent',
                    'emotion': 'neutral',
                    'response_time': 4.2,
                    'turn_id': 2,
                    'total_turns': 6,
                    'agent_model': available_models[0],
                    'scenario': available_scenarios[2] if len(available_scenarios) > 2 else available_scenarios[0]
                }
            }
        ]
        
        print("\n" + "="*60)
        print("REAL-TIME INFERENCE DEMONSTRATION")
        print("="*60)
        
        for i, scenario in enumerate(demo_scenarios, 1):
            prediction = inference_engine.predict_trust(
                scenario['utterance'], scenario['context']
            )
            
            print(f"\nScenario {i}:")
            print(f"Agent Response: '{scenario['utterance']}'")
            print(f"Context: {scenario['context']['scenario']} (Turn {scenario['context']['turn_id']})")
            print(f"Predicted Trust Score: {prediction['trust_score']:.2f}/7.0")
            print(f"Confidence: {prediction['confidence']:.2f}")
            print("-" * 60)
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving trained models...")
        
        models_dir = self.output_dir / "trained_models"
        models_dir.mkdir(exist_ok=True)
        
        # Save traditional ML models
        for name, model in self.trained_models.items():
            if name in ['random_forest', 'xgboost', 'svm']:
                joblib.dump(model, models_dir / f"{name}_model.pkl")
            elif name == 'lstm':
                torch.save(model.state_dict(), models_dir / f"{name}_model.pth")
            elif name == 'ensemble':
                joblib.dump(model, models_dir / f"{name}_model.pkl")
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, models_dir / "feature_engineer.pkl")
        
        # Save model results
        results_df = pd.DataFrame(self.model_results).T
        results_df.to_csv(models_dir / "model_results_summary.csv")
        
        logger.info(f"Models saved to {models_dir}")
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline"""
        start_time = datetime.now()
        logger.info("="*80)
        logger.info("STARTING COMPLETE ML/DL PIPELINE FOR TRUST DETECTION")
        logger.info("="*80)
        
        try:
            # 1. Data Loading and Preparation
            self.load_and_prepare_data()
            
            # 2. Feature Engineering
            self.prepare_features()
            
            # 3. Train Traditional ML Models
            self.train_traditional_models()
            
            # 4. Train Deep Learning Models
            self.train_deep_learning_models()
            
            # 5. Train BERT Model (optional, may skip if too resource intensive)
            self.train_bert_model()
            
            # 6. Create Ensemble
            self.create_ensemble()
            
            # 7. Comprehensive Evaluation
            comparison_df = self.evaluate_and_compare_models()
            
            # 8. Real-time Inference Demo
            self.demonstrate_real_time_inference()
            
            # 9. Save Everything
            self.save_models()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total execution time: {duration}")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("="*80)
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    
    print("🤖 Trust & Emotion Detection ML/DL Pipeline")
    print("=" * 80)
    print("This pipeline will:")
    print("✓ Load and analyze your conversational trust dataset")
    print("✓ Engineer comprehensive features from text, emotions, and temporal data") 
    print("✓ Train multiple ML models (Random Forest, XGBoost, SVM)")
    print("✓ Train deep learning models (LSTM, BERT)")
    print("✓ Create optimized model ensemble")
    print("✓ Evaluate and compare all models")
    print("✓ Demonstrate real-time trust prediction")
    print("=" * 80)
    
    # Check if data directory exists
    if not os.path.exists("data"):
        print("❌ Error: 'data' directory not found!")
        print("Please ensure your dataset is in the 'data' directory with 'conversations' and 'metadata' subdirectories.")
        return
    
    # Create and run pipeline
    pipeline = CompletePipeline()
    results = pipeline.run_complete_pipeline()
    
    if results is not None:
        print("\n🎉 SUCCESS! Your trust detection models are ready!")
        print(f"📊 Check '{pipeline.output_dir}' for detailed results and visualizations")
        print("🚀 You can now use the trained models for real-time trust estimation in live conversations")
    else:
        print("❌ Pipeline execution failed. Check the logs for details.")

if __name__ == "__main__":
    main()
