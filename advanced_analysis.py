import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.ticker as ticker

# Set up visualization style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Create analysis results directory
RESULTS_DIR = "advanced_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_conversations(data_dir):
    """Load all conversation JSON files from the data directory and include model info from metadata."""
    conversations = []
    conv_dir = os.path.join(data_dir, "conversations")
    metadata_dir = os.path.join(data_dir, "metadata")
    
    # First, load all metadata into a dictionary for quick lookup
    metadata = {}
    if os.path.exists(metadata_dir):
        for meta_file in os.listdir(metadata_dir):
            if meta_file.endswith('.json'):
                try:
                    with open(os.path.join(metadata_dir, meta_file), 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        conv_id = meta.get('conversation_id')
                        if conv_id:
                            metadata[conv_id] = meta
                except Exception as e:
                    print(f"Error loading metadata {meta_file}: {e}")
    
    # Now load conversations and add model info from metadata
    for filename in tqdm(os.listdir(conv_dir), desc="Loading conversations"):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(conv_dir, filename), 'r', encoding='utf-8') as f:
                    conv = json.load(f)
                    base_name = os.path.splitext(filename)[0]  # e.g., 'conv_0001'
                    conv_id = base_name.replace('conv_', '')  # e.g., '0001'
                    conv['filename'] = filename
                    
                    # Get model name from metadata if available
                    if conv_id in metadata and 'agent_model' in metadata[conv_id]:
                        conv['model'] = metadata[conv_id]['agent_model']
                    # Try with full conversation ID (with 'conv_' prefix)
                    elif base_name in metadata and 'agent_model' in metadata[base_name]:
                        conv['model'] = metadata[base_name]['agent_model']
                    # Fallback to conversation data if not in metadata
                    elif 'data' in conv and 'model_name' in conv['data']:
                        conv['model'] = conv['data']['model_name']
                    else:
                        conv['model'] = 'unknown'
                        
                    conversations.append(conv)
            except json.JSONDecodeError:
                print(f"Error loading {filename}")
                
    return conversations

def analyze_trust_dynamics(conversations):
    """Analyze how trust evolves over conversation turns."""
    max_turns = max(len(conv.get('turns', [])) for conv in conversations)
    trust_by_turn = {}
    
    # Initialize data structure
    for turn_idx in range(max_turns):
        trust_by_turn[turn_idx + 1] = []
    
    # Collect trust scores by turn
    for conv in conversations:
        for turn in conv.get('turns', []):
            if turn['speaker'] == 'agent' and 'trust_score' in turn and turn['trust_score'] is not None:
                turn_num = turn['turn_id'] // 2  # Adjust for turn numbering
                if 0 < turn_num <= max_turns:  # Ensure turn number is within bounds
                    trust_by_turn[turn_num].append(turn['trust_score'])
    
    # Calculate mean and confidence intervals
    turn_numbers = []
    mean_trust = []
    ci_lower = []
    ci_upper = []
    
    for turn_num, scores in trust_by_turn.items():
        if scores:  # Only include turns with data
            turn_numbers.append(turn_num)
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            n = len(scores)
            
            # Calculate 95% confidence interval
            ci = 1.96 * (std / np.sqrt(n))
            mean_trust.append(mean)
            ci_lower.append(mean - ci)
            ci_upper.append(mean + ci)
    
    return {
        'turn_numbers': turn_numbers,
        'mean_trust': mean_trust,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def analyze_models_comparison(conversations):
    """Compare trust metrics across different agent models."""
    model_data = defaultdict(lambda: {
        'trust_scores': [],
        'competence_scores': [],
        'benevolence_scores': [],
        'integrity_scores': []
    })
    
    # Collect scores by model
    for conv in conversations:
        model = conv.get('model', 'unknown')
        for turn in conv.get('turns', []):
            if turn['speaker'] == 'agent':
                if 'trust_score' in turn and turn['trust_score'] is not None:
                    model_data[model]['trust_scores'].append(turn['trust_score'])
                if 'trust_category_scores' in turn and turn['trust_category_scores']:
                    model_data[model]['competence_scores'].append(turn['trust_category_scores'].get('competence', 0))
                    model_data[model]['benevolence_scores'].append(turn['trust_category_scores'].get('benevolence', 0))
                    model_data[model]['integrity_scores'].append(turn['trust_category_scores'].get('integrity', 0))
    
    # Calculate mean scores
    results = {}
    for model, scores in model_data.items():
        if scores['trust_scores']:  # Only include models with trust score data
            results[model] = {
                'mean_trust': np.mean(scores['trust_scores']),
                'mean_competence': np.mean(scores['competence_scores']) if scores['competence_scores'] else 0,
                'mean_benevolence': np.mean(scores['benevolence_scores']) if scores['benevolence_scores'] else 0,
                'mean_integrity': np.mean(scores['integrity_scores']) if scores['integrity_scores'] else 0,
                'n': len(scores['trust_scores'])
            }
    
    return results

def analyze_latency_impact(conversations):
    """Analyze the relationship between response time and trust."""
    response_times = []
    trust_scores = []
    
    for conv in conversations:
        for turn in conv.get('turns', []):
            if (turn['speaker'] == 'agent' and 
                'response_time' in turn and 
                'trust_score' in turn and 
                turn['response_time'] is not None and 
                turn['trust_score'] is not None):
                response_times.append(turn['response_time'])
                trust_scores.append(turn['trust_score'])
    
    return {
        'response_times': response_times,
        'trust_scores': trust_scores,
        'correlation': np.corrcoef(response_times, trust_scores)[0, 1] if len(response_times) > 1 else 0
    }

def analyze_sentiment_impact(conversations):
    """Analyze the relationship between sentiment/emotion and trust."""
    emotion_trust = defaultdict(list)
    
    for conv in conversations:
        for turn in conv.get('turns', []):
            if (turn['speaker'] == 'user' and 
                'emotion_detected' in turn and 
                'trust_score' in turn and 
                turn['emotion_detected'] and 
                turn['trust_score'] is not None):
                emotion_trust[turn['emotion_detected']].append(turn['trust_score'])
    
    # Calculate mean trust by emotion
    emotion_stats = {}
    for emotion, scores in emotion_trust.items():
        if scores:
            emotion_stats[emotion] = {
                'mean_trust': np.mean(scores),
                'std': np.std(scores, ddof=1),
                'n': len(scores)
            }
    
    return emotion_stats

def plot_trust_dynamics(trust_data):
    """Plot trust dynamics across conversation turns."""
    plt.figure(figsize=(12, 6))
    
    # Plot mean trust with confidence intervals
    plt.plot(trust_data['turn_numbers'], trust_data['mean_trust'], 
             marker='o', label='Mean Trust', color='#2E86C1')
    plt.fill_between(trust_data['turn_numbers'], 
                    trust_data['ci_lower'], 
                    trust_data['ci_upper'], 
                    color='#85C1E9', alpha=0.3, label='95% CI')
    
    # Add trend line
    z = np.polyfit(trust_data['turn_numbers'], trust_data['mean_trust'], 1)
    p = np.poly1d(z)
    plt.plot(trust_data['turn_numbers'], p(trust_data['turn_numbers']), 
             'r--', label='Trend Line')
    
    plt.title('Trust Dynamics Across Conversation Turns', pad=20)
    plt.xlabel('Turn Number')
    plt.ylabel('Mean Trust Score (1-7)')
    plt.xticks(trust_data['turn_numbers'])
    plt.ylim(1, 7)  # Assuming 7-point Likert scale
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'trust_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison(model_data):
    """Plot comparison of trust metrics across models."""
    if not model_data:
        print("No model comparison data available.")
        return
    
    # Prepare data for plotting
    models = list(model_data.keys())
    metrics = ['mean_trust', 'mean_competence', 'mean_benevolence', 'mean_integrity']
    metric_names = ['Overall Trust', 'Competence', 'Benevolence', 'Integrity']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, metric_names)):
        # Sort models by the current metric
        sorted_models = sorted(
            [(m, model_data[m][metric]) for m in models if metric in model_data[m]],
            key=lambda x: x[1],
            reverse=True
        )
        
        if not sorted_models:
            continue
            
        model_names = [m[0] for m in sorted_models]
        values = [m[1] for m in sorted_models]
        
        # Plot
        ax = axes[i]
        bars = ax.barh(model_names, values, color='#2E86C1')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}', ha='left', va='center')
        
        ax.set_title(f'{title} by Model')
        ax.set_xlabel('Mean Score (1-7)')
        ax.set_xlim(0, 7)  # Assuming 7-point Likert scale
        
        # Rotate model names if they're long
        if max(len(str(m)) for m in model_names) > 15:
            ax.tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_latency_impact(latency_data):
    """Plot the relationship between response time and trust."""
    if not latency_data['response_times']:
        print("No latency data available.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create hexbin plot for better visualization of dense data
    hb = plt.hexbin(latency_data['response_times'], 
                   latency_data['trust_scores'],
                   gridsize=30, 
                   cmap='Blues',
                   mincnt=1)
    
    # Add colorbar
    cb = plt.colorbar(hb)
    cb.set_label('Number of Observations')
    
    # Add correlation coefficient
    plt.text(0.95, 0.05, f'Correlation: {latency_data["correlation"]:.2f}',
             transform=plt.gca().transAxes, ha='right',
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Impact of Response Time on Trust')
    plt.xlabel('Response Time (seconds)')
    plt.ylabel('Trust Score (1-7)')
    plt.ylim(0.5, 7.5)  # Assuming 7-point Likert scale
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'latency_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sentiment_impact(emotion_data):
    """Plot the relationship between sentiment/emotion and trust."""
    if not emotion_data:
        print("No emotion data available.")
        return
    
    # Sort emotions by mean trust
    emotions = sorted(emotion_data.items(), key=lambda x: x[1]['mean_trust'], reverse=True)
    emotion_names = [e[0].capitalize() for e in emotions]
    means = [e[1]['mean_trust'] for e in emotions]
    stds = [e[1]['std'] for e in emotions]
    ns = [e[1]['n'] for e in emotions]
    
    # Calculate 95% CI
    ci = [1.96 * (s / np.sqrt(n)) for s, n in zip(stds, ns)]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with error bars
    x = np.arange(len(emotion_names))
    bars = plt.bar(x, means, yerr=ci, capsize=5, color='#2E86C1')
    
    # Add value labels
    for i, (mean, n) in enumerate(zip(means, ns)):
        plt.text(i, mean + 0.1, f'{mean:.2f}\n(n={n})', 
                ha='center', va='bottom', fontsize=9)
    
    plt.title('Impact of User Emotion on Trust')
    plt.xlabel('User Emotion')
    plt.ylabel('Mean Trust Score (1-7)')
    plt.xticks(x, emotion_names, rotation=45, ha='right')
    plt.ylim(0, 7)  # Assuming 7-point Likert scale
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'emotion_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_advanced_metrics(trust_data, model_data, latency_data, emotion_data):
    """Save advanced metrics to a text file."""
    with open(os.path.join(RESULTS_DIR, 'advanced_metrics_summary.txt'), 'w') as f:
        f.write("Advanced Conversation Metrics Summary\n")
        f.write("="*50 + "\n\n")
        
        # Trust dynamics
        f.write("TRUST DYNAMICS ACROSS TURNS\n")
        f.write("-"*50 + "\n")
        f.write("Turn\tMean Trust\t95% CI Lower\t95% CI Upper\n")
        for i in range(len(trust_data['turn_numbers'])):
            f.write(f"{trust_data['turn_numbers'][i]}\t"
                   f"{trust_data['mean_trust'][i]:.2f}\t\t"
                   f"{trust_data['ci_lower'][i]:.2f}\t\t"
                   f"{trust_data['ci_upper'][i]:.2f}\n")
        
        # Model comparison
        if model_data:
            f.write("\nMODEL COMPARISON\n")
            f.write("-"*50 + "\n")
            f.write("Model\tTrust\tCompetence\tBenevolence\tIntegrity\tN\n")
            for model, data in sorted(model_data.items(), key=lambda x: x[1].get('mean_trust', 0), reverse=True):
                f.write(f"{model}\t"
                       f"{data.get('mean_trust', 0):.2f}\t"
                       f"{data.get('mean_competence', 0):.2f}\t\t"
                       f"{data.get('mean_benevolence', 0):.2f}\t\t"
                       f"{data.get('mean_integrity', 0):.2f}\t"
                       f"{data.get('n', 0)}\n")
        
        # Latency impact
        if latency_data['correlation'] != 0:
            f.write("\nLATENCY IMPACT\n")
            f.write("-"*50 + "\n")
            f.write(f"Correlation between response time and trust: {latency_data['correlation']:.3f}\n")
        
        # Emotion impact
        if emotion_data:
            f.write("\nEMOTION IMPACT\n")
            f.write("-"*50 + "\n")
            f.write("Emotion\tMean Trust\tStd Dev\tN\n")
            for emotion, data in sorted(emotion_data.items(), key=lambda x: x[1]['mean_trust'], reverse=True):
                f.write(f"{emotion}\t{data['mean_trust']:.2f}\t\t{data['std']:.2f}\t{data['n']}\n")

def main():
    print("Starting advanced analysis of conversation data...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Load conversations
    conversations = load_conversations(data_dir)
    if not conversations:
        print("No conversations found. Exiting.")
        return
    
    print(f"Analyzing {len(conversations)} conversations...")
    
    # Perform analyses
    print("Analyzing trust dynamics...")
    trust_data = analyze_trust_dynamics(conversations)
    
    print("Comparing models...")
    model_data = analyze_models_comparison(conversations)
    
    print("Analyzing latency impact...")
    latency_data = analyze_latency_impact(conversations)
    
    print("Analyzing emotion impact...")
    emotion_data = analyze_sentiment_impact(conversations)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_trust_dynamics(trust_data)
    plot_model_comparison(model_data)
    plot_latency_impact(latency_data)
    plot_sentiment_impact(emotion_data)
    
    # Save metrics
    save_advanced_metrics(trust_data, model_data, latency_data, emotion_data)
    
    print(f"\nAdvanced analysis complete! Results saved to '{RESULTS_DIR}' directory.")
    print("Files created:")
    for f in os.listdir(RESULTS_DIR):
        print(f"- {os.path.join(RESULTS_DIR, f)}")

if __name__ == "__main__":
    main()
