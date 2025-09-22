#!/usr/bin/env python3
"""
Complete Training Pipeline for ConTrust Model

Novel conversational trust prediction with hierarchical attention,
trust evolution memory, and multi-task learning
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from datetime import datetime
import warnings
from typing import Dict, List, Tuple
import argparse

# Import our ConTrust components
from contrust_model import ConTrustModel, ConTrustTrainer, create_contrust_config
from contrust_data_processor import create_contrust_dataloaders, ConTrustDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('contrust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ConTrustExperiment:
    """
    Complete experimental pipeline for ConTrust model
    """
    
    def __init__(self, config: Dict, output_dir: str = "contrust_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize model and trainer
        self.model = None
        self.trainer = None
        self.processor = None
        
        # Training history
        self.history = None
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_model(self):
        """Initialize ConTrust model and trainer"""
        
        logger.info("Initializing ConTrust model...")
        
        self.model = ConTrustModel(self.config)
        self.trainer = ConTrustTrainer(self.model, self.config)
        
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("Model initialized")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Save model architecture
        with open(self.output_dir / 'model_architecture.txt', 'w') as f:
            f.write(str(self.model))
            f.write(f"\n\nTotal parameters: {total_params:,}")
            f.write(f"\nTrainable parameters: {trainable_params:,}")
    
    def prepare_data(self, data_dir: str):
        """Prepare data for training"""
        
        logger.info("Preparing conversational trust dataset...")
        
        batch_size = self.config.get('batch_size', 8)
        val_split = self.config.get('val_split', 0.2)
        
        self.train_loader, self.val_loader, self.processor = create_contrust_dataloaders(
            data_dir, self.config, batch_size, val_split
        )
        
        # Save processor
        processor_path = self.output_dir / 'contrust_processor.pkl'
        self.processor.save(str(processor_path))
        
        logger.info(f"Data prepared: {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val")
        
        # Dataset statistics
        self.analyze_dataset()
    
    def analyze_dataset(self):
        """Analyze dataset characteristics"""
        
        logger.info("Analyzing dataset characteristics...")
        
        # Sample a few batches to understand data distribution
        trust_scores = []
        conversation_lengths = []
        emotions = []
        
        for i, batch in enumerate(self.train_loader):
            if i >= 10:  # Sample first 10 batches
                break
            
            trust_scores.extend(batch['target_trust'].cpu().numpy().flatten())
            conversation_lengths.extend(batch['conversation_lengths'].cpu().numpy())
            emotions.extend(batch['target_emotion'].cpu().numpy().flatten())
        
        # Create analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trust score distribution
        axes[0, 0].hist(trust_scores, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Trust Score Distribution')
        axes[0, 0].set_xlabel('Trust Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Conversation length distribution
        axes[0, 1].hist(conversation_lengths, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Conversation Length Distribution')
        axes[0, 1].set_xlabel('Number of Turns')
        axes[0, 1].set_ylabel('Frequency')
        
        # Trust vs conversation length
        axes[1, 0].scatter(conversation_lengths, trust_scores, alpha=0.6, color='orange')
        axes[1, 0].set_title('Trust Score vs Conversation Length')
        axes[1, 0].set_xlabel('Conversation Length')
        axes[1, 0].set_ylabel('Trust Score')
        
        # Emotion distribution
        unique_emotions, emotion_counts = np.unique(emotions, return_counts=True)
        axes[1, 1].bar(unique_emotions, emotion_counts, color='plum')
        axes[1, 1].set_title('Emotion Distribution')
        axes[1, 1].set_xlabel('Emotion Class')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics
        stats = {
            'trust_score_mean': float(np.mean(trust_scores)),
            'trust_score_std': float(np.std(trust_scores)),
            'trust_score_min': float(np.min(trust_scores)),
            'trust_score_max': float(np.max(trust_scores)),
            'avg_conversation_length': float(np.mean(conversation_lengths)),
            'max_conversation_length': int(np.max(conversation_lengths)),
            'num_emotions': len(unique_emotions),
            'total_training_samples': len(self.train_loader.dataset),
            'total_validation_samples': len(self.val_loader.dataset)
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Dataset analysis complete")
        logger.info(f"  Trust scores: {stats['trust_score_mean']:.2f} ± {stats['trust_score_std']:.2f}")
        logger.info(f"  Avg conversation length: {stats['avg_conversation_length']:.1f}")
    
    def train_model(self, epochs: int = 50):
        """Train ConTrust model"""
        
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        if self.train_loader is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        logger.info(f"Training ConTrust model for {epochs} epochs...")
        
        start_time = datetime.now()
        
        # Train the model
        self.history = self.trainer.train(self.train_loader, self.val_loader, epochs)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Training completed in {training_duration}")
        
        # Save training history
        self.save_training_results()
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.history
    
    def save_training_results(self):
        """Save training results and model"""
        
        # Save training history
        history_df = pd.DataFrame({
            'epoch': range(len(self.history['train'])),
            'train_loss': [h['total'] for h in self.history['train']],
            'val_loss': [h['total'] for h in self.history['val']],
            'train_trust_loss': [h.get('trust', 0) for h in self.history['train']],
            'val_trust_loss': [h.get('trust', 0) for h in self.history['val']],
        })
        
        history_df.to_csv(self.output_dir / 'training_history.csv', index=False)
        
        # Save model
        model_path = self.output_dir / 'contrust_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.history
        }, model_path)
        
        # Save best model (if exists)
        best_model_path = 'contrust_best_model.pth'
        if os.path.exists(best_model_path):
            import shutil
            shutil.move(best_model_path, self.output_dir / 'contrust_best_model.pth')
        
        logger.info(f"✓ Model and results saved to {self.output_dir}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.history['train']))
        
        # Total loss
        train_losses = [h['total'] for h in self.history['train']]
        val_losses = [h['total'] for h in self.history['val']]
        
        axes[0, 0].plot(epochs, train_losses, label='Train', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Validation', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trust loss
        train_trust = [h.get('trust', 0) for h in self.history['train']]
        val_trust = [h.get('trust', 0) for h in self.history['val']]
        
        axes[0, 1].plot(epochs, train_trust, label='Train', color='blue')
        axes[0, 1].plot(epochs, val_trust, label='Validation', color='red')
        axes[0, 1].set_title('Trust Prediction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss weights evolution
        if 'weights' in self.history['train'][0]:
            weight_history = [h['weights'] for h in self.history['train'] if 'weights' in h]
            if weight_history:
                weight_array = torch.stack(weight_history).cpu().numpy()
                
                for i in range(weight_array.shape[1]):
                    axes[1, 0].plot(epochs[:len(weight_array)], weight_array[:, i], 
                                   label=f'Task {i+1}')
                
                axes[1, 0].set_title('Adaptive Loss Weights')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Weight')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate (if available)
        try:
            lrs = [self.trainer.scheduler.get_last_lr()[0] for _ in epochs]
            axes[1, 1].plot(epochs, lrs, color='green')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        except:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✓ Training curves saved")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        
        logger.info("📊 Evaluating ConTrust model...")
        
        self.model.eval()
        
        # Collect predictions and targets
        all_predictions = {'trust': [], 'emotion': [], 'engagement': []}
        all_targets = {'trust': [], 'emotion': [], 'engagement': []}
        all_attention_weights = []
        all_modality_weights = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Collect predictions
                all_predictions['trust'].extend(outputs['trust_score'].cpu().numpy().flatten())
                all_predictions['emotion'].extend(torch.argmax(outputs['emotion'], dim=1).cpu().numpy())
                all_predictions['engagement'].extend(outputs['engagement'].cpu().numpy().flatten())
                
                # Collect targets
                all_targets['trust'].extend(batch['target_trust'].cpu().numpy().flatten())
                all_targets['emotion'].extend(batch['target_emotion'].cpu().numpy().flatten())
                all_targets['engagement'].extend(batch['target_engagement'].cpu().numpy().flatten())
                
                # Collect attention weights (first head only for visualization)
                if outputs['attention_weights']:
                    attn_weights = outputs['attention_weights'][0][0, 0].cpu().numpy()  # First sample, first head
                    all_attention_weights.append(attn_weights)
                
                # Collect modality weights
                modality_weights = outputs['modality_weights'][0].cpu().numpy()  # First sample
                all_modality_weights.append(modality_weights)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
        
        # Trust prediction metrics
        trust_mse = mean_squared_error(all_targets['trust'], all_predictions['trust'])
        trust_mae = mean_absolute_error(all_targets['trust'], all_predictions['trust'])
        trust_r2 = r2_score(all_targets['trust'], all_predictions['trust'])
        trust_rmse = np.sqrt(trust_mse)
        
        # Emotion classification accuracy
        emotion_acc = accuracy_score(all_targets['emotion'], all_predictions['emotion'])
        
        # Engagement prediction metrics
        engagement_mse = mean_squared_error(all_targets['engagement'], all_predictions['engagement'])
        engagement_r2 = r2_score(all_targets['engagement'], all_predictions['engagement'])
        
        # Trust-specific metrics
        trust_errors = np.abs(np.array(all_targets['trust']) - np.array(all_predictions['trust']))
        within_05 = np.mean(trust_errors <= 0.5) * 100
        within_10 = np.mean(trust_errors <= 1.0) * 100
        
        metrics = {
            'trust_mse': trust_mse,
            'trust_rmse': trust_rmse,
            'trust_mae': trust_mae,
            'trust_r2': trust_r2,
            'trust_within_0.5': within_05,
            'trust_within_1.0': within_10,
            'emotion_accuracy': emotion_acc,
            'engagement_mse': engagement_mse,
            'engagement_r2': engagement_r2
        }
        
        # Save metrics
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create evaluation plots
        self.plot_evaluation_results(all_predictions, all_targets, all_attention_weights, all_modality_weights)
        
        logger.info("✓ Evaluation complete")
        logger.info(f"  Trust R²: {trust_r2:.4f}")
        logger.info(f"  Trust RMSE: {trust_rmse:.4f}")
        logger.info(f"  Within ±0.5: {within_05:.1f}%")
        logger.info(f"  Emotion Accuracy: {emotion_acc:.4f}")
        
        return metrics
    
    def plot_evaluation_results(self, predictions, targets, attention_weights, modality_weights):
        """Plot evaluation results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Trust prediction scatter plot
        axes[0, 0].scatter(targets['trust'], predictions['trust'], alpha=0.6, color='blue')
        axes[0, 0].plot([1, 7], [1, 7], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Trust Score')
        axes[0, 0].set_ylabel('Predicted Trust Score')
        axes[0, 0].set_title('Trust Score Prediction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trust prediction errors
        errors = np.array(targets['trust']) - np.array(predictions['trust'])
        axes[0, 1].hist(errors, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Trust Prediction Error Distribution')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        
        # Engagement prediction
        axes[0, 2].scatter(targets['engagement'], predictions['engagement'], alpha=0.6, color='orange')
        min_eng, max_eng = min(min(targets['engagement']), min(predictions['engagement'])), max(max(targets['engagement']), max(predictions['engagement']))
        axes[0, 2].plot([min_eng, max_eng], [min_eng, max_eng], 'r--', lw=2)
        axes[0, 2].set_xlabel('True Engagement')
        axes[0, 2].set_ylabel('Predicted Engagement')
        axes[0, 2].set_title('Engagement Prediction')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Attention weights heatmap (if available)
        if attention_weights:
            avg_attention = np.mean(attention_weights[:10], axis=0)  # Average first 10 samples
            im = axes[1, 0].imshow(avg_attention, cmap='Blues', aspect='auto')
            axes[1, 0].set_title('Average Attention Weights')
            axes[1, 0].set_xlabel('Key Position')
            axes[1, 0].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Modality weights distribution
        if modality_weights:
            modality_avg = np.mean(modality_weights, axis=0)
            modality_names = ['Text-Temporal', 'Text-Behavioral', 'Text-Only']
            axes[1, 1].bar(modality_names, modality_avg, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[1, 1].set_title('Average Modality Weights')
            axes[1, 1].set_ylabel('Weight')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Trust score distribution comparison
        axes[1, 2].hist(targets['trust'], bins=20, alpha=0.7, label='True', color='blue')
        axes[1, 2].hist(predictions['trust'], bins=20, alpha=0.7, label='Predicted', color='orange')
        axes[1, 2].set_xlabel('Trust Score')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Trust Score Distributions')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_complete_experiment(self, data_dir: str, epochs: int = 50):
        """Run complete ConTrust experiment"""
        
        start_time = datetime.now()
        
        logger.info("Starting ConTrust Experiment")
        logger.info("=" * 80)
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        logger.info("=" * 80)
        
        try:
            # Setup
            self.setup_model()
            self.prepare_data(data_dir)
            
            # Training
            self.train_model(epochs)
            
            # Evaluation
            metrics = self.evaluate_model()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Final report
            logger.info("=" * 80)
            logger.info("🎉 CONTRUST EXPERIMENT COMPLETED!")
            logger.info(f"Duration: {duration}")
            logger.info(f"Best Trust R²: {metrics['trust_r2']:.4f}")
            logger.info(f"Best Trust RMSE: {metrics['trust_rmse']:.4f}")
            logger.info(f"Predictions within ±0.5: {metrics['trust_within_0.5']:.1f}%")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 80)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function for ConTrust training"""
    
    parser = argparse.ArgumentParser(description='Train ConTrust Model')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='contrust_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_contrust_config()
    config.update({
        'batch_size': args.batch_size,
        'lr': args.lr,
    })
    
    # Print welcome message
    print("🚀 ConTrust: Novel Conversational Trust Prediction")
    print("=" * 60)
    print("Features:")
    print("✓ Hierarchical Attention Transformers")
    print("✓ Trust-Aware Self-Attention")
    print("✓ Conversation Memory for Trust Evolution")
    print("✓ Multi-Modal Fusion (Text + Temporal + Behavioral)")
    print("✓ Multi-Task Learning (Trust + Emotion + Engagement)")
    print("✓ Adaptive Loss Weighting")
    print("=" * 60)
    
    # Run experiment
    experiment = ConTrustExperiment(config, args.output_dir)
    results = experiment.run_complete_experiment(args.data_dir, args.epochs)
    
    if results:
        print("\n🎯 ConTrust training completed successfully!")
        print(f"📊 Results saved to: {args.output_dir}")
    else:
        print("\n❌ ConTrust training failed. Check logs for details.")

if __name__ == "__main__":
    main()
