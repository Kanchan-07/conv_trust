#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Trust and Emotion Detection Models
- Model performance metrics and visualization
- Cross-validation and statistical significance testing
- Error analysis and feature importance
- Model comparison and ensemble evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
import torch
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TrustModelEvaluator:
    """Comprehensive evaluation for trust prediction models"""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str) -> Dict[str, float]:
        """Comprehensive evaluation for regression models"""
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Trust-specific metrics (1-7 scale)
        metrics['within_0.5'] = np.mean(np.abs(y_true - y_pred) <= 0.5) * 100
        metrics['within_1.0'] = np.mean(np.abs(y_true - y_pred) <= 1.0) * 100
        metrics['correlation'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Save detailed results
        results_df = pd.DataFrame({
            'true': y_true,
            'predicted': y_pred,
            'error': y_true - y_pred,
            'abs_error': np.abs(y_true - y_pred)
        })
        results_df.to_csv(self.results_dir / f"{model_name}_predictions.csv", index=False)
        
        return metrics
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str, save: bool = True):
        """Create comprehensive plots for regression results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Trust Score')
        axes[0, 0].set_ylabel('Predicted Trust Score')
        axes[0, 0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² to the plot
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residuals plot
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Trust Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'{model_name}: Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'{model_name}: Error Distribution')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        
        # 4. Trust score distribution comparison
        axes[1, 1].hist(y_true, bins=20, alpha=0.7, label='True', color='blue')
        axes[1, 1].hist(y_pred, bins=20, alpha=0.7, label='Predicted', color='orange')
        axes[1, 1].set_xlabel('Trust Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'{model_name}: Score Distributions')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / f"{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation evaluation"""
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = {
            'r2': cross_val_score(model, X, y, cv=kfold, scoring='r2'),
            'neg_mse': cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'),
            'neg_mae': cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_absolute_error')
        }
        
        # Summary statistics
        results = {}
        for metric, scores in cv_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)
            results[f'{metric}_scores'] = scores
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict], 
                      metric: str = 'R²') -> pd.DataFrame:
        """Compare multiple models and create ranking"""
        
        comparison_data = []
        for model_name, results in model_results.items():
            comparison_data.append({
                'Model': model_name,
                'R²': results.get('r2', 0),
                'RMSE': results.get('rmse', float('inf')),
                'MAE': results.get('mae', float('inf')),
                'Within 0.5': results.get('within_0.5', 0),
                'Within 1.0': results.get('within_1.0', 0),
                'MAPE': results.get('mape', float('inf'))
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by specified metric (higher is better for R², within_X; lower is better for errors)
        if metric in ['R²', 'r2', 'Within 0.5', 'within_0.5', 'Within 1.0', 'within_1.0']:
            comparison_df = comparison_df.sort_values(metric, ascending=False)
        else:
            comparison_df = comparison_df.sort_values(metric, ascending=True)
        
        # Save comparison
        comparison_df.to_csv(self.results_dir / "model_comparison.csv", index=False)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame):
        """Visualize model comparison"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['R²', 'RMSE', 'MAE', 'Within 0.5', 'Within 1.0', 'MAPE']
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            row, col = i // 3, i % 3
            
            bars = axes[row, col].bar(comparison_df['Model'], comparison_df[metric], 
                                     color=color, alpha=0.8)
            axes[row, col].set_title(f'Model Comparison: {metric}')
            axes[row, col].set_xlabel('Model')
            axes[row, col].set_ylabel(metric)
            
            # Rotate x-axis labels for better readability
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}',
                                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison_plot.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, model, feature_names: List[str], 
                                 model_name: str, top_k: int = 20):
        """Analyze and visualize feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            logger.warning(f"Model {model_name} doesn't have feature importance")
            return None
        
        # Create feature importance DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = feature_imp_df.head(top_k)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'{model_name}: Top {top_k} Feature Importances')
        plt.gca().invert_yaxis()
        
        # Color bars by importance
        norm = plt.Normalize(top_features['importance'].min(), top_features['importance'].max())
        colors = plt.cm.viridis(norm(top_features['importance']))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"{model_name}_feature_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save feature importance data
        feature_imp_df.to_csv(self.results_dir / f"{model_name}_feature_importance.csv", index=False)
        
        return feature_imp_df
    
    def error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      features_df: pd.DataFrame, model_name: str):
        """Perform detailed error analysis"""
        
        errors = np.abs(y_true - y_pred)
        
        # Identify high-error samples
        high_error_threshold = np.percentile(errors, 90)
        high_error_mask = errors > high_error_threshold
        
        analysis_results = {
            'total_samples': len(y_true),
            'high_error_samples': np.sum(high_error_mask),
            'high_error_threshold': high_error_threshold,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors)
        }
        
        # Analyze error patterns by different features if available
        if features_df is not None and len(features_df) == len(y_true):
            error_df = features_df.copy()
            error_df['error'] = errors
            error_df['high_error'] = high_error_mask
            
            # Error by categorical features
            categorical_cols = error_df.select_dtypes(include=['object']).columns
            
            for col in categorical_cols:
                if col in error_df.columns:
                    error_by_category = error_df.groupby(col)['error'].agg(['mean', 'std', 'count'])
                    error_by_category.to_csv(self.results_dir / f"{model_name}_error_by_{col}.csv")
        
        # Save error analysis
        with open(self.results_dir / f"{model_name}_error_analysis.txt", 'w') as f:
            f.write("Error Analysis Results\n")
            f.write("=" * 50 + "\n")
            for key, value in analysis_results.items():
                f.write(f"{key}: {value}\n")
        
        return analysis_results

class ModelEnsemble:
    """Ensemble of multiple trust prediction models"""
    
    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0 for name in models.keys()}
        self.normalize_weights()
        
    def normalize_weights(self):
        """Normalize ensemble weights to sum to 1"""
        total_weight = sum(self.weights.values())
        self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif hasattr(model, 'forward'):  # PyTorch model
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy()
            else:
                continue
                
            predictions.append(pred * self.weights[name])
        
        return np.sum(predictions, axis=0)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                        method: str = 'mse') -> Dict[str, float]:
        """Optimize ensemble weights using validation data"""
        from scipy.optimize import minimize
        
        def objective(weights):
            self.weights = {name: w for name, w in zip(self.models.keys(), weights)}
            self.normalize_weights()
            pred = self.predict(X_val)
            
            if method == 'mse':
                return mean_squared_error(y_val, pred)
            elif method == 'mae':
                return mean_absolute_error(y_val, pred)
        
        # Initial weights
        initial_weights = list(self.weights.values())
        
        # Optimization constraints (weights must be non-negative and sum to 1)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in initial_weights]
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimized_weights = {name: w for name, w in zip(self.models.keys(), result.x)}
            self.weights = optimized_weights
            logger.info(f"Optimized ensemble weights: {optimized_weights}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
        
        return self.weights

def create_evaluation_report(evaluator: TrustModelEvaluator, 
                           model_results: Dict[str, Dict],
                           dataset_info: Dict[str, Any]) -> str:
    """Create comprehensive evaluation report"""
    
    report = []
    report.append("=" * 80)
    report.append("TRUST PREDICTION MODEL EVALUATION REPORT")
    report.append("=" * 80)
    report.append(f"Generated on: {pd.Timestamp.now()}")
    report.append(f"Dataset: {dataset_info.get('name', 'Trust Dataset')}")
    report.append(f"Total samples: {dataset_info.get('total_samples', 'N/A')}")
    report.append(f"Features used: {dataset_info.get('num_features', 'N/A')}")
    report.append("")
    
    # Model comparison
    comparison_df = evaluator.compare_models(model_results)
    report.append("MODEL PERFORMANCE COMPARISON")
    report.append("-" * 40)
    report.append(comparison_df.to_string(index=False))
    report.append("")
    
    # Best model
    best_model = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['R²']
    report.append(f"BEST PERFORMING MODEL: {best_model}")
    report.append(f"R² Score: {best_r2:.4f}")
    report.append("")
    
    # Detailed results for each model
    report.append("DETAILED MODEL RESULTS")
    report.append("-" * 40)
    for model_name, results in model_results.items():
        report.append(f"\n{model_name.upper()}:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                report.append(f"  {metric}: {value:.4f}")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(evaluator.results_dir / "evaluation_report.txt", 'w') as f:
        f.write(report_text)
    
    return report_text

if __name__ == "__main__":
    # Example usage
    evaluator = TrustModelEvaluator()
    
    # Mock data for demonstration
    np.random.seed(42)
    y_true = np.random.uniform(1, 7, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)
    
    # Evaluate mock model
    metrics = evaluator.evaluate_regression_model(y_true, y_pred, "MockModel")
    evaluator.plot_regression_results(y_true, y_pred, "MockModel")
    
    print("Evaluation framework ready!")
    print(f"Mock model metrics: {metrics}")
