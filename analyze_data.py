import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import textstat
from wordcloud import WordCloud
from datetime import datetime

# Set up visualization style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# Create analysis results directory
RESULTS_DIR = "analysis_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_conversations(data_dir):
    """Load all conversation JSON files from the data directory."""
    conversations = []
    conv_dir = os.path.join(data_dir, "conversations")
    
    for filename in tqdm(os.listdir(conv_dir), desc="Loading conversations"):
        if filename.endswith('.json'):
            with open(os.path.join(conv_dir, filename), 'r', encoding='utf-8') as f:
                try:
                    conv = json.load(f)
                    conv['filename'] = filename
                    conversations.append(conv)
                except json.JSONDecodeError:
                    print(f"Error loading {filename}")
    return conversations

def analyze_conversation_metrics(conversations):
    """Analyze basic conversation metrics."""
    metrics = {
        'conversation_lengths': [],
        'response_times': [],
        'trust_scores': [],
        'competence_scores': [],
        'benevolence_scores': [],
        'integrity_scores': [],
        'user_emotions': defaultdict(int),
        'agent_emotions': defaultdict(int),
        'word_counts': [],
        'readability_scores': []
    }
    
    for conv in conversations:
        turns = conv.get('turns', [])
        metrics['conversation_lengths'].append(len(turns))
        
        for turn in turns:
            if turn['speaker'] == 'agent':
                if turn.get('response_time'):
                    metrics['response_times'].append(turn['response_time'])
                
                if turn.get('trust_score'):
                    metrics['trust_scores'].append(turn['trust_score'])
                
                if turn.get('trust_category_scores'):
                    metrics['competence_scores'].append(turn['trust_category_scores'].get('competence', 0))
                    metrics['benevolence_scores'].append(turn['trust_category_scores'].get('benevolence', 0))
                    metrics['integrity_scores'].append(turn['trust_category_scores'].get('integrity', 0))
            
            # Emotion analysis
            if turn.get('emotion_detected'):
                if turn['speaker'] == 'user':
                    metrics['user_emotions'][turn['emotion_detected']] += 1
                else:
                    metrics['agent_emotions'][turn['emotion_detected']] += 1
            
            # Text analysis
            text = turn.get('utterance', '')
            if text:
                metrics['word_counts'].append(len(text.split()))
                try:
                    metrics['readability_scores'].append(textstat.flesch_reading_ease(text))
                except:
                    pass
    
    return metrics

def plot_conversation_metrics(metrics):
    """Generate various plots from conversation metrics."""
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    
    # Plot conversation length distribution
    sns.histplot(metrics['conversation_lengths'], bins=20, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Conversation Lengths (Turns)')
    axes[0, 0].set_xlabel('Number of Turns')
    axes[0, 0].set_ylabel('Frequency')
    
    # Plot response time distribution
    if metrics['response_times']:
        sns.histplot(metrics['response_times'], bins=20, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Agent Response Times (seconds)')
        axes[0, 1].set_xlabel('Response Time (s)')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot trust scores
    if metrics['trust_scores']:
        sns.boxplot(data=pd.DataFrame({
            'Trust Score': metrics['trust_scores'],
            'Competence': metrics['competence_scores'],
            'Benevolence': metrics['benevolence_scores'],
            'Integrity': metrics['integrity_scores']
        }), ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Trust Scores by Category')
        axes[1, 0].set_ylabel('Score (1-7)')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot emotion distribution
    if metrics['user_emotions'] and metrics['agent_emotions']:
        df_emotions = pd.DataFrame({
            'User': dict(metrics['user_emotions']),
            'Agent': dict(metrics['agent_emotions'])
        }).fillna(0)
        df_emotions.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Emotion Distribution by Speaker')
        axes[1, 1].set_xlabel('Emotion')
        axes[1, 1].set_ylabel('Count')
    
    # Plot word count vs readability
    if metrics['word_counts'] and metrics['readability_scores']:
        sns.scatterplot(x=metrics['word_counts'], y=metrics['readability_scores'], alpha=0.6, ax=axes[2, 0])
        axes[2, 0].set_title('Word Count vs Readability Score')
        axes[2, 0].set_xlabel('Word Count')
        axes[2, 0].set_ylabel('Readability Score (Flesch)')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'conversation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_word_cloud(conversations, speaker='agent'):
    """Generate word cloud for agent or user utterances."""
    text = ' '.join([turn['utterance'] for conv in conversations 
                    for turn in conv.get('turns', []) 
                    if turn.get('speaker') == speaker and 'utterance' in turn])
    
    if not text.strip():
        return
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {speaker.capitalize()} Utterances')
    plt.savefig(os.path.join(RESULTS_DIR, f'wordcloud_{speaker}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_metrics_to_file(metrics):
    """Save computed metrics to a text file."""
    with open(os.path.join(RESULTS_DIR, 'metrics_summary.txt'), 'w') as f:
        f.write("Conversation Metrics Summary\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total conversations analyzed: {len(metrics['conversation_lengths'])}\n")
        f.write(f"Average conversation length: {np.mean(metrics['conversation_lengths']):.2f} turns\n")
        f.write(f"Median conversation length: {np.median(metrics['conversation_lengths'])} turns\n")
        f.write(f"Min conversation length: {min(metrics['conversation_lengths'])} turns\n")
        f.write(f"Max conversation length: {max(metrics['conversation_lengths'])} turns\n\n")
        
        if metrics['response_times']:
            f.write(f"Average response time: {np.mean(metrics['response_times']):.2f} seconds\n")
            f.write(f"Median response time: {np.median(metrics['response_times']):.2f} seconds\n")
            f.write(f"Min response time: {min(metrics['response_times']):.2f} seconds\n")
            f.write(f"Max response time: {max(metrics['response_times']):.2f} seconds\n\n")
        
        if metrics['trust_scores']:
            f.write("Trust Score Statistics (1-7 scale):\n")
            f.write(f"  - Overall Trust: {np.mean(metrics['trust_scores']):.2f} ± {np.std(metrics['trust_scores']):.2f}\n")
            f.write(f"  - Competence: {np.mean(metrics['competence_scores']):.2f} ± {np.std(metrics['competence_scores']):.2f}\n")
            f.write(f"  - Benevolence: {np.mean(metrics['benevolence_scores']):.2f} ± {np.std(metrics['benevolence_scores']):.2f}\n")
            f.write(f"  - Integrity: {np.mean(metrics['integrity_scores']):.2f} ± {np.std(metrics['integrity_scores']):.2f}\n\n")
        
        f.write("Emotion Distribution:\n")
        f.write("User Emotions:\n")
        for emotion, count in sorted(metrics['user_emotions'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {emotion}: {count} ({count/sum(metrics['user_emotions'].values())*100:.1f}%)\n")
        
        f.write("\nAgent Emotions:\n")
        for emotion, count in sorted(metrics['agent_emotions'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {emotion}: {count} ({count/sum(metrics['agent_emotions'].values())*100:.1f}%)\n")

def main():
    print("Starting analysis of conversation data...")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Load conversations
    conversations = load_conversations(data_dir)
    if not conversations:
        print("No conversations found. Exiting.")
        return
    
    print(f"Analyzing {len(conversations)} conversations...")
    
    # Analyze metrics
    metrics = analyze_conversation_metrics(conversations)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_conversation_metrics(metrics)
    generate_word_cloud(conversations, 'user')
    generate_word_cloud(conversations, 'agent')
    
    # Save metrics to file
    save_metrics_to_file(metrics)
    
    print(f"\nAnalysis complete! Results saved to '{RESULTS_DIR}' directory.")
    print("Files created:")
    for f in os.listdir(RESULTS_DIR):
        print(f"- {os.path.join(RESULTS_DIR, f)}")

if __name__ == "__main__":
    main()
