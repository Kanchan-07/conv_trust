import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import textstat
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# --- Configuration ---
DATA_DIR = 'data'
RESULTS_DIR = 'final_analysis_results'

# --- Setup ---
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# --- Data Loading ---
def load_data(data_dir):
    """Loads and merges conversation and metadata files into a single DataFrame."""
    conv_dir = os.path.join(data_dir, "conversations")
    meta_dir = os.path.join(data_dir, "metadata")
    
    all_turns = []
    
    for filename in tqdm(os.listdir(conv_dir), desc="Loading Data"):
        if not filename.endswith('.json'):
            continue

        conv_id = os.path.splitext(filename)[0]
        
        # Load metadata
        model_name = 'unknown'
        scenario = 'unknown'
        meta_path = os.path.join(meta_dir, filename)
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f_meta:
                meta = json.load(f_meta)
                model_name = meta.get('agent_model', 'unknown')
                scenario = meta.get('scenario', 'unknown')

        # Load conversation turns
        conv_path = os.path.join(conv_dir, filename)
        with open(conv_path, 'r', encoding='utf-8') as f_conv:
            conv = json.load(f_conv)
            for turn in conv.get('turns', []):
                turn_data = turn.copy()

                # Flatten trust category scores
                trust_scores = turn_data.pop('trust_category_scores', None)
                if isinstance(trust_scores, dict):
                    turn_data['competence'] = trust_scores.get('competence')
                    turn_data['benevolence'] = trust_scores.get('benevolence')
                    turn_data['integrity'] = trust_scores.get('integrity')
                else: # Ensure columns exist even if no scores
                    turn_data['competence'] = None
                    turn_data['benevolence'] = None
                    turn_data['integrity'] = None
                
                turn_data['conversation_id'] = conv_id
                turn_data['model'] = model_name
                turn_data['scenario'] = scenario
                turn_data['total_turns'] = len(conv.get('turns', []))
                all_turns.append(turn_data)
                
    df = pd.DataFrame(all_turns)
    # Ensure numeric types
    for col in ['trust_score', 'response_time', 'competence', 'benevolence', 'integrity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- 5.2 Trust Dynamics Across Turns ---
def analyze_trust_dynamics(df, results_dir):
    """Analyzes and plots trust trajectory across turns."""
    agent_turns = df[df['speaker'] == 'agent'].copy()
    agent_turns['turn_num'] = agent_turns['turn_id'] // 2
    
    trust_by_turn = agent_turns.groupby('turn_num')['trust_score'].agg(['mean', 'sem']).dropna()
    trust_by_turn['ci95'] = trust_by_turn['sem'] * 1.96

    plt.figure(figsize=(10, 6))
    plt.plot(trust_by_turn.index, trust_by_turn['mean'], marker='o', linestyle='-', label='Mean Trust Score')
    plt.fill_between(trust_by_turn.index, 
                     trust_by_turn['mean'] - trust_by_turn['ci95'], 
                     trust_by_turn['mean'] + trust_by_turn['ci95'], 
                     alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Figure 6: Average Trust Trajectory Across Conversation Turns')
    plt.xlabel('Agent Turn Number')
    plt.ylabel('Mean Trust Score')
    plt.xlim(left=0)
    plt.ylim(bottom=1, top=7)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'figure_6_trust_dynamics.png'), dpi=300)
    plt.close()

# --- 5.3 Agent Model Comparisons ---
def analyze_model_comparisons(df, results_dir):
    """Analyzes and plots trust scores by agent model."""
    agent_turns = df[(df['speaker'] == 'agent') & (df['trust_score'].notna())]
    
    # 5.3.1 Overall Trust
    model_trust = agent_turns.groupby('model')['trust_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    model_trust.columns = ['Mean Trust', 'Std Dev', 'N']
    model_trust.to_csv(os.path.join(results_dir, 'table_5_model_trust_levels.csv'))

    plt.figure(figsize=(12, 8))
    sns.barplot(x=model_trust['Mean Trust'], y=model_trust.index, palette='viridis')
    plt.title('Figure 7: Mean Overall Trust Scores by Agent Model')
    plt.xlabel('Mean Trust Score (1-7)')
    plt.ylabel('Agent Model')
    plt.xlim(0, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_7_model_comparison_overall.png'), dpi=300)
    plt.close()

    # ANOVA
    model = ols('trust_score ~ C(model)', data=agent_turns).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    with open(os.path.join(results_dir, 'stats_summary.txt'), 'w') as f:
        f.write("--- Agent Model Comparison (ANOVA) ---\n")
        f.write(str(anova_table))
        f.write('\n')

    # 5.3.2 Dimension-wise Comparison
    dims = ['competence', 'benevolence', 'integrity']
    dim_df = agent_turns.groupby('model')[dims].mean().sort_values('competence', ascending=False)
    dim_df.plot(kind='bar', figsize=(14, 8), colormap='viridis')
    plt.title('Dimension-Wise Trust Scores by Model')
    plt.ylabel('Mean Score (1-7)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_model_comparison_dimensions.png'), dpi=300)
    plt.close()

# --- 5.4 Impact of Latency, Sentiment, and Emotion ---
def analyze_impacts(df, results_dir):
    """Analyzes impact of latency and emotion on trust."""
    agent_turns = df[(df['speaker'] == 'agent') & (df['trust_score'].notna())]

    # 5.4.1 Latency
    latency_df = agent_turns[agent_turns['response_time'].notna()]
    plt.figure(figsize=(10, 6))
    sns.regplot(x='response_time', y='trust_score', data=latency_df, 
                scatter_kws={'alpha':0.2, 's':10}, line_kws={'color':'red'},
                lowess=True)
    plt.title('Figure 8: Trust Score vs. Response Latency')
    plt.xlabel('Response Latency (seconds)')
    plt.ylabel('Trust Score')
    plt.xlim(0, latency_df['response_time'].quantile(0.99)) # Exclude outliers
    plt.savefig(os.path.join(results_dir, 'figure_8_latency_impact.png'), dpi=300)
    plt.close()

    # 5.4.2 & 5.4.3 Emotion/Sentiment
    user_turns = df[df['speaker'] == 'user'].copy()
    # Associate user emotion with the next agent's trust score
    user_turns['next_turn_id'] = user_turns['turn_id'] + 1
    merged = pd.merge(user_turns[['conversation_id', 'next_turn_id', 'emotion_detected']],
                      agent_turns[['conversation_id', 'turn_id', 'trust_score']],
                      left_on=['conversation_id', 'next_turn_id'],
                      right_on=['conversation_id', 'turn_id'])
    
    emotion_trust = merged.groupby('emotion_detected')['trust_score'].agg(['mean', 'sem']).dropna()
    emotion_trust['ci95'] = emotion_trust['sem'] * 1.96

    plt.figure(figsize=(12, 7))
    emotion_trust['mean'].sort_values().plot(kind='barh', xerr=emotion_trust['ci95'], color=sns.color_palette('coolwarm', len(emotion_trust)))
    plt.title('Figure 9: Mean Trust Score Following User Emotion')
    plt.xlabel('Mean Trust Score (1-7)')
    plt.ylabel('Preceding User Emotion')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'figure_9_emotion_impact.png'), dpi=300)
    plt.close()

# --- 5.5 Engagement Patterns ---
def analyze_engagement(df, results_dir):
    """Analyzes impact of engagement on trust."""
    conv_summary = df.groupby('conversation_id').agg(
        avg_trust=('trust_score', 'mean'),
        total_turns=('turn_id', 'max'),
        user_utterance_len=('utterance', lambda x: x[df['speaker'] == 'user'].str.len().mean())
    ).dropna()

    # Conversation Length
    conv_summary['length_group'] = pd.cut(conv_summary['total_turns'], bins=[0, 6, np.inf], labels=['<=6 turns', '>6 turns'])
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='length_group', y='avg_trust', data=conv_summary)
    plt.title('Trust by Conversation Length')
    plt.xlabel('Conversation Length')
    plt.ylabel('Average Trust Score')
    plt.savefig(os.path.join(results_dir, 'engagement_length_impact.png'), dpi=300)
    plt.close()

# --- 5.6 Qualitative Analysis ---
def analyze_qualitative(df, results_dir):
    """Extracts examples of high and low trust conversations."""
    conv_trust = df.groupby('conversation_id')['trust_score'].mean().dropna().sort_values()
    low_trust_ids = conv_trust.head(3).index
    high_trust_ids = conv_trust.tail(3).index

    with open(os.path.join(results_dir, 'table_6_qualitative_examples.txt'), 'w', encoding='utf-8') as f:
        f.write("--- Table 6: Qualitative Examples of Trust Failures and Successes ---\n\n")
        
        f.write("=== LOW TRUST EXAMPLES ===\n")
        for conv_id in low_trust_ids:
            f.write(f"\n--- Conversation ID: {conv_id} (Avg Trust: {conv_trust[conv_id]:.2f}) ---\n")
            conv_df = df[df['conversation_id'] == conv_id].sort_values('turn_id')
            for _, row in conv_df.iterrows():
                f.write(f"{row['speaker'].upper()}: {row['utterance']}\n")

        f.write("\n=== HIGH TRUST EXAMPLES ===\n")
        for conv_id in high_trust_ids:
            f.write(f"\n--- Conversation ID: {conv_id} (Avg Trust: {conv_trust[conv_id]:.2f}) ---\n")
            conv_df = df[df['conversation_id'] == conv_id].sort_values('turn_id')
            for _, row in conv_df.iterrows():
                f.write(f"{row['speaker'].upper()}: {row['utterance']}\n")

# --- Main Execution ---
def main():
    """Main function to run all analyses."""
    print("Starting comprehensive analysis...")
    df = load_data(DATA_DIR)
    
    print("Analyzing trust dynamics (5.2)...")
    analyze_trust_dynamics(df, RESULTS_DIR)
    
    print("Analyzing model comparisons (5.3)...")
    analyze_model_comparisons(df, RESULTS_DIR)
    
    print("Analyzing impacts of latency and emotion (5.4)...")
    analyze_impacts(df, RESULTS_DIR)
    
    print("Analyzing engagement patterns (5.5)...")
    analyze_engagement(df, RESULTS_DIR)
    
    print("Performing qualitative analysis (5.6)...")
    analyze_qualitative(df, RESULTS_DIR)
    
    print(f"\nAnalysis complete. All results saved in '{RESULTS_DIR}' directory.")

if __name__ == '__main__':
    main()
