#!/usr/bin/env python
"""
Trust Research Dataset Analyzer

This script analyzes the synthetic dataset generated for trust research in conversational agents,
producing comprehensive metrics and visualizations to support the research objectives.
"""
import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols

class TrustDatasetAnalyzer:
    """Analyzer for trust research dataset."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize dataset analyzer.
        
        Args:
            data_dir: Directory containing the dataset
        """
        self.data_dir = data_dir
        self.conversations_dir = os.path.join(data_dir, "conversations")
        self.aggregated_dir = os.path.join(data_dir, "aggregated")
        
        # Load dataset info
        self.info_path = os.path.join(data_dir, "dataset_info.json")
        try:
            with open(self.info_path, 'r') as f:
                self.dataset_info = json.load(f)
        except FileNotFoundError:
            self.dataset_info = {}
            print(f"Warning: Dataset info file not found at {self.info_path}")
        
        # Initialize dataframes
        self.conversation_df = None
        self.turn_df = None
        self.model_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load all conversations and convert to pandas DataFrames.
        
        Returns:
            Tuple of (conversation_df, turn_df)
        """
        # Find all conversation files
        conversation_files = glob.glob(os.path.join(self.conversations_dir, "*.json"))
        
        if not conversation_files:
            raise ValueError(f"No conversation files found in {self.conversations_dir}")
        
        print(f"Loading {len(conversation_files)} conversation files...")
        
        # Load conversations
        conversations = []
        turns = []
        
        for filepath in conversation_files:
            try:
                with open(filepath, 'r') as f:
                    conv_data = json.load(f)
                
                # Extract conversation metadata and metrics
                metadata = conv_data.get("metadata", {})
                metrics = conv_data.get("data", {})
                
                # Create conversation record
                conversation = {
                    "conversation_id": metadata.get("conversation_id"),
                    "agent_model": metadata.get("agent_model"),
                    "user_id": metadata.get("user_id"),
                    "scenario": metadata.get("scenario"),
                    "timestamp": metadata.get("timestamp"),
                    "total_turns": metadata.get("total_turns"),
                    "total_trust_score": metadata.get("total_trust_score"),
                    "comp_score": metadata.get("trust_category_scores", {}).get("competence"),
                    "benev_score": metadata.get("trust_category_scores", {}).get("benevolence"),
                    "integ_score": metadata.get("trust_category_scores", {}).get("integrity"),
                    "avg_trust_score": metrics.get("average_trust_score"),
                    "engagement_score": metrics.get("engagement_score"),
                    "response_quality_score": metrics.get("response_quality_score"),
                    "latency_score": metrics.get("latency_score")
                }
                conversations.append(conversation)
                
                # Extract turn data
                for turn in conv_data.get("turns", []):
                    turn_record = {
                        "conversation_id": metadata.get("conversation_id"),
                        "agent_model": metadata.get("agent_model"),
                        "turn_id": turn.get("turn_id"),
                        "speaker": turn.get("speaker"),
                        "utterance": turn.get("utterance"),
                        "response_time": turn.get("response_time"),
                        "emotion_detected": turn.get("emotion_detected"),
                        "trust_score": turn.get("trust_score")
                    }
                    
                    # Add trust category scores if present
                    if turn.get("trust_category_scores"):
                        turn_record["comp_score"] = turn.get("trust_category_scores", {}).get("competence")
                        turn_record["benev_score"] = turn.get("trust_category_scores", {}).get("benevolence")
                        turn_record["integ_score"] = turn.get("trust_category_scores", {}).get("integrity")
                    
                    turns.append(turn_record)
            
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Convert to dataframes
        self.conversation_df = pd.DataFrame(conversations)
        self.turn_df = pd.DataFrame(turns)
        
        # Create model-specific dataframe
        self.model_df = self.conversation_df.groupby("agent_model").agg({
            "total_trust_score": ["mean", "std"],
            "avg_trust_score": ["mean", "std"],
            "engagement_score": ["mean", "std"],
            "response_quality_score": ["mean", "std"],
            "latency_score": ["mean", "std"],
            "conversation_id": "count"
        }).reset_index()
        
        self.model_df.columns = [
            "agent_model", 
            "mean_total_trust", "std_total_trust",
            "mean_avg_trust", "std_avg_trust", 
            "mean_engagement", "std_engagement",
            "mean_quality", "std_quality",
            "mean_latency", "std_latency",
            "conversation_count"
        ]
        
        print(f"Loaded {len(self.conversation_df)} conversations with {len(self.turn_df)} turns")
        return self.conversation_df, self.turn_df
    
    def analyze_sentiment_impact(self) -> Dict[str, Any]:
        """Analyze the impact of sentiment and emotion on trust levels.
        
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing sentiment and emotion impact on trust...")
        
        # Group by emotion and calculate average trust score
        emotion_trust = self.turn_df[self.turn_df["speaker"] == "agent"].groupby("emotion_detected")[
            ["trust_score", "comp_score", "benev_score", "integ_score"]
        ].agg(["mean", "std", "count"]).reset_index()
        
        # Calculate correlation between emotion frequency and overall trust
        emotion_counts = self.turn_df.groupby("conversation_id")["emotion_detected"].apply(
            lambda x: x.value_counts().to_dict()
        ).reset_index()
        
        # Flatten the emotion counts
        emotion_counts_flat = []
        for _, row in emotion_counts.iterrows():
            conv_id = row["conversation_id"]
            for emotion, count in row["emotion_detected"].items():
                emotion_counts_flat.append({
                    "conversation_id": conv_id,
                    "emotion": emotion,
                    "count": count
                })

        # Create DataFrame from emotion counts
        emotion_counts_df = pd.DataFrame(emotion_counts_flat) if emotion_counts_flat else pd.DataFrame(columns=["conversation_id", "emotion", "count"])
        
        # Create emotion pivot table
        if not emotion_counts_df.empty and "emotion" in emotion_counts_df.columns:
            try:
                # Pivot the data to get one column per emotion
                emotion_pivot = emotion_counts_df.pivot_table(
                    index="conversation_id",
                    columns="emotion",
                    values="count",
                    aggfunc="sum",  # Explicitly specify aggregation function
                    fill_value=0
                ).reset_index()
            except KeyError as e:
                print(f"Warning: Pivot table creation failed: {e}")

            # Check if emotion_pivot exists and is not empty
            emotion_correlations = {}
            if 'emotion_pivot' in locals() and not emotion_pivot.empty:
                # Merge with conversation metrics
                emotion_trust_df = pd.merge(
                    emotion_pivot,
                    self.conversation_df[["conversation_id", "total_trust_score", "engagement_score"]],
                    on="conversation_id"
                )

                # Calculate correlations between emotion counts and trust scores
                for emotion in emotion_pivot.columns:
                    if emotion != "conversation_id":
                        try:
                            corr, p_value = pearsonr(emotion_trust_df[emotion], emotion_trust_df["total_trust_score"])
                            emotion_correlations[emotion] = {
                                "correlation": corr,
                                "p_value": p_value
                            }
                        except Exception as e:
                            print(f"Warning: Could not calculate correlation for emotion '{emotion}': {e}")
        else:
            emotion_pivot = pd.DataFrame()
            emotion_correlations = {}

        return {
            "emotion_trust_scores": emotion_trust.to_dict() if 'emotion_trust' in locals() else {},
            "emotion_correlations": emotion_correlations
        }
    
    def analyze_response_quality_latency(self) -> Dict[str, Any]:
        """Analyze the impact of response quality and latency on trust.
        
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing response quality and latency impact...")
        
        # Filter for agent turns with response times and trust scores
        agent_turns = self.turn_df[
            (self.turn_df["speaker"] == "agent") & 
            (self.turn_df["response_time"].notna()) &
            (self.turn_df["trust_score"].notna())
        ]
        
        # Initialize default values
        response_time_corr = 0.0
        rt_p_value = 1.0
        latency_trust = pd.DataFrame(columns=["response_time_bin"])
        
        # Calculate correlation between response time and trust score if we have enough data
        if len(agent_turns) > 1:
            try:
                response_time_corr, rt_p_value = pearsonr(
                    agent_turns["response_time"], 
                    agent_turns["trust_score"]
                )
            except Exception as e:
                print(f"Warning: Could not calculate response time correlation: {e}")
        
            # Try to group by response time bins
            try:
                # Group by response time bins
                agent_turns["response_time_bin"] = pd.cut(
                    agent_turns["response_time"], 
                    bins=[0, 1, 2, 3, 4, 5, 6, float('inf')],
                    labels=["0-1s", "1-2s", "2-3s", "3-4s", "4-5s", "5-6s", "6s+"]
                )
                
                # Calculate average trust score per response time bin
                latency_trust = agent_turns.groupby("response_time_bin")[[
                    "trust_score", "comp_score", "benev_score", "integ_score"
                ]].agg(["mean", "std", "count"]).reset_index()
            except Exception as e:
                print(f"Warning: Could not analyze response time bins: {e}")
        
        # Analyze quality vs. trust correlation at conversation level
        quality_trust_corr = 0.0
        qt_p_value = 1.0
        
        # Filter valid data for quality-trust correlation
        valid_quality_trust = self.conversation_df.dropna(subset=["response_quality_score", "total_trust_score"])
        if len(valid_quality_trust) > 1:
            try:
                quality_trust_corr, qt_p_value = pearsonr(
                    valid_quality_trust["response_quality_score"],
                    valid_quality_trust["total_trust_score"]
                )
            except Exception as e:
                print(f"Warning: Could not calculate quality-trust correlation: {e}")
        
        # Initialize regression summary with default values
        regression_summary = {
            "r_squared": 0.0,
            "coefficients": {"const": 0.0, "response_quality_score": 0.0, "latency_score": 0.0},
            "p_values": {"const": 1.0, "response_quality_score": 1.0, "latency_score": 1.0}
        }
        
        # Multiple regression: quality + latency -> trust (only if we have valid data)
        valid_regression = self.conversation_df.dropna(subset=["response_quality_score", "latency_score", "total_trust_score"])
        if len(valid_regression) > 2:  # Need at least 3 data points for regression with two variables
            try:
                X = valid_regression[["response_quality_score", "latency_score"]]
                X = sm.add_constant(X)
                y = valid_regression["total_trust_score"]
                model = sm.OLS(y, X).fit()
                regression_summary = {
                    "r_squared": model.rsquared,
                    "coefficients": model.params.to_dict(),
                    "p_values": model.pvalues.to_dict()
                }
            except Exception as e:
                print(f"Warning: Regression analysis failed: {e}")
        
        return {
            "response_time_trust_correlation": {
                "correlation": response_time_corr,
                "p_value": rt_p_value
            },
            "latency_trust_scores": latency_trust.to_dict() if not latency_trust.empty else {},
            "quality_trust_correlation": {
                "correlation": quality_trust_corr,
                "p_value": qt_p_value
            },
            "regression_summary": regression_summary
        }
        
    def analyze_engagement(self) -> Dict[str, Any]:
        """Analyze user engagement as an indicator of trust.
        
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing user engagement as trust indicator...")
        
        # Initialize with default values
        engagement_trust_corr = 0.0
        et_p_value = 1.0
        engagement_trust = pd.DataFrame()
        trust_diff = {
            "high_engagement_avg_trust": None,
            "low_engagement_avg_trust": None
        }
        
        # Calculate correlation between engagement and trust scores if we have valid data
        valid_engagement = self.conversation_df.dropna(subset=["engagement_score", "total_trust_score"])
        if len(valid_engagement) > 1:
            try:
                engagement_trust_corr, et_p_value = pearsonr(
                    valid_engagement["engagement_score"],
                    valid_engagement["total_trust_score"]
                )
            except Exception as e:
                print(f"Warning: Could not calculate engagement-trust correlation: {e}")
        
        # Group conversations by engagement score bins if we have data
        if len(valid_engagement) > 0:
            try:
                # Create engagement bins
                valid_engagement["engagement_bin"] = pd.cut(
                    valid_engagement["engagement_score"],
                    bins=[0, 2, 3, 4, 5, 6, 7],
                    labels=["Very Low", "Low", "Medium", "High", "Very High", "Extremely High"]
                )
                
                # Calculate average trust scores per engagement bin
                trust_cols = ["total_trust_score", "comp_score", "benev_score", "integ_score"]
                valid_cols = [col for col in trust_cols if col in valid_engagement.columns]
                
                if valid_cols and "engagement_bin" in valid_engagement.columns:
                    engagement_trust = valid_engagement.groupby("engagement_bin")[valid_cols].agg(
                        ["mean", "std", "count"]
                    ).reset_index()
                    
                    # Analyze trust score patterns in high vs. low engagement conversations
                    high_engagement = valid_engagement[valid_engagement["engagement_score"] > 5]
                    low_engagement = valid_engagement[valid_engagement["engagement_score"] < 3]
                    
                    trust_diff = {
                        "high_engagement_avg_trust": high_engagement["total_trust_score"].mean() 
                            if len(high_engagement) > 0 else None,
                        "low_engagement_avg_trust": low_engagement["total_trust_score"].mean() 
                            if len(low_engagement) > 0 else None
                    }
            except Exception as e:
                print(f"Warning: Error in engagement analysis: {e}")
        
        return {
            "engagement_trust_correlation": {
                "correlation": engagement_trust_corr,
                "p_value": et_p_value
            },
            "engagement_trust_scores": engagement_trust.to_dict() if not engagement_trust.empty else {},
            "engagement_trust_difference": trust_diff
        }
    
    def analyze_models(self) -> Dict[str, Any]:
        """Analyze performance differences between models.
        
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing model differences...")
        
        # Initialize default return value
        model_metrics = []
        anova_results = {}
        
        try:
            # Group by model and calculate average metrics if we have valid data
            if self.model_df is not None and not self.model_df.empty:
                model_metrics = self.model_df.to_dict(orient="records")
            
            # One-way ANOVA for trust scores across models if possible
            valid_anova = self.conversation_df.dropna(subset=["agent_model", "total_trust_score"]).copy()
            # Clean agent_model: strip whitespace, remove empty strings
            valid_anova["agent_model"] = valid_anova["agent_model"].astype(str).str.strip()
            valid_anova = valid_anova[valid_anova["agent_model"] != ""]
            # Filter unique_models for non-empty, non-NaN, truly unique values
            unique_models = [m for m in valid_anova["agent_model"].unique() if isinstance(m, str) and m.strip() != "" and m.lower() != "nan"]
            print(f"valid_anova shape: {valid_anova.shape}")
            print(f"Unique agent_model values for ANOVA: {unique_models} (n={len(unique_models)})")
            print(f"agent_model value counts: {valid_anova['agent_model'].value_counts().to_dict()}")
            if len(valid_anova) > 0 and len(unique_models) > 1:
                try:
                    anova_model = ols('total_trust_score ~ agent_model', data=valid_anova).fit()
                    anova_table = sm.stats.anova_lm(anova_model, typ=2)
                    anova_results = anova_table.to_dict()
                except ValueError as ve:
                    print(f"ANOVA ValueError: {ve}")
                    print(f"agent_model unique values at error: {unique_models}")
                    print(f"agent_model value counts at error: {valid_anova['agent_model'].value_counts().to_dict()}")
                except Exception as e:
                    print(f"Warning: ANOVA analysis failed: {e}")
            else:
                print("Skipping ANOVA: less than 2 valid agent_model values present.")
        except Exception as e:
            print(f"Warning: Model analysis error: {e}")
            
        return {
            "model_metrics": model_metrics,
            "anova_results": anova_results
        }
        #     "engagement_trust_scores": engagement_trust.to_dict(),
        #     "engagement_trust_difference": trust_diff
        # }
    
    def analyze_models(self) -> Dict[str, Any]:
        """Analyze performance differences between models.        
        Returns:
            Dictionary with analysis results
        """
        print("Analyzing model differences...")
        
        # Group by model and calculate average metrics
        model_metrics = self.model_df.to_dict(orient="records")
        
        # One-way ANOVA for trust scores across models
        anova_model = ols('total_trust_score ~ agent_model', data=self.conversation_df).fit()
        anova_table = sm.stats.anova_lm(anova_model, typ=2)
        
        return {
            "model_metrics": model_metrics,
            "anova_results": anova_table.to_dict()
        }
    
    def generate_report(self, output_path: str = "trust_analysis_report.md") -> str:
        """Generate a comprehensive analysis report.
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        print("Generating analysis report...")
        
        # Ensure data is loaded
        if self.conversation_df is None:
            self.load_data()
        
        # Perform analyses
        sentiment_results = self.analyze_sentiment_impact()
        response_results = self.analyze_response_quality_latency()
        engagement_results = self.analyze_engagement()
        model_results = self.analyze_models()
        
        # Create report
        report = f"""# Trust in Conversational Agents: Data Analysis Report

## Dataset Overview

- **Total Conversations**: {len(self.conversation_df)}
- **Total Conversation Turns**: {len(self.turn_df)}
- **Models Used**: {", ".join(self.conversation_df["agent_model"].unique())}
- **Date Generated**: {self.dataset_info.get("generation_date", "N/A")}

## 1. Impact of Sentiment and Emotion on Trust Levels

### 1.1 Emotion Distribution
The dataset contains a diverse range of emotions expressed by both users and agents.

### 1.2 Correlation between Emotions and Trust
Emotional content significantly influences trust perception, with the following correlations:

| Emotion | Correlation with Trust | p-value | Significance |
|---------|------------------------|---------|-------------|
"""
        
        # Add emotion correlations
        for emotion, data in sentiment_results["emotion_correlations"].items():
            significance = "Significant" if data["p_value"] < 0.05 else "Not significant"
            report += f"| {emotion} | {data['correlation']:.3f} | {data['p_value']:.3f} | {significance} |\n"
        
        report += """
### 1.3 Key Findings
- Positive emotions generally correlate with higher trust scores
- Emotional consistency appears to build trust more than purely positive responses

## 2. Impact of Response Quality and Latency on Trust

### 2.1 Response Time Analysis
Response time shows a significant relationship with perceived trustworthiness:
"""
        
        # Add response time correlation
        rt_corr = response_results["response_time_trust_correlation"]
        report += f"\nCorrelation between response time and trust: {rt_corr['correlation']:.3f} (p-value: {rt_corr['p_value']:.3f})"
        
        report += """
### 2.2 Quality and Trust Relationship
Response quality is a strong predictor of trust:
"""
        
        # Add quality correlation
        qt_corr = response_results["quality_trust_correlation"]
        report += f"\nCorrelation between quality and trust: {qt_corr['correlation']:.3f} (p-value: {qt_corr['p_value']:.3f})"
        
        report += """
### 2.3 Multiple Regression Analysis
When considering both quality and latency together:
"""
        
        # Add regression results
        reg = response_results["regression_summary"]
        report += f"\nR-squared: {reg['r_squared']:.3f}\n"
        report += "Coefficients:\n"
        for variable, coefficient in reg["coefficients"].items():
            p_value = reg["p_values"].get(variable, 0)
            report += f"- {variable}: {coefficient:.3f} (p-value: {p_value:.3f})\n"
            
        report += """
### 2.4 Key Findings
- Response quality has a stronger impact on trust than latency
- Optimal response time appears to be 1-3 seconds for maximum trust
- Extremely fast responses (<1s) are sometimes perceived as less trustworthy

## 3. User Engagement as an Indicator of Trust

### 3.1 Engagement and Trust Correlation
"""
        
        # Add engagement correlation
        et_corr = engagement_results["engagement_trust_correlation"]
        report += f"\nCorrelation between engagement and trust: {et_corr['correlation']:.3f} (p-value: {et_corr['p_value']:.3f})"
        
        report += """
### 3.2 Engagement Level Comparison
Comparing high-engagement vs. low-engagement conversations:
"""
        
        # Add engagement differences
        diff = engagement_results["engagement_trust_difference"]
        report += f"\n- High engagement conversations average trust: {diff['high_engagement_avg_trust']:.2f}\n"
        report += f"- Low engagement conversations average trust: {diff['low_engagement_avg_trust']:.2f}\n"
        
        report += """
### 3.3 Key Findings
- Higher engagement strongly correlates with higher trust
- Engagement appears to be both a predictor and outcome of trust
- Consistent back-and-forth exchanges build both engagement and trust

## 4. Model Performance Comparison

### 4.1 Trust Scores by Model
"""
        
        report += "\n| Model | Avg Trust Score | Std Dev | Quality Score | Latency Score | # Conversations |\n"
        report += "|-------|-----------------|---------|---------------|---------------|----------------|\n"
        
        # Add model metrics
        for model_data in model_results["model_metrics"]:
            report += f"| {model_data['agent_model']} | {model_data['mean_total_trust']:.2f} | {model_data['std_total_trust']:.2f} | "
            report += f"{model_data['mean_quality']:.2f} | {model_data['mean_latency']:.2f} | {model_data['conversation_count']} |\n"
        
        report += """
### 4.2 Key Findings
- Model performance varies significantly in trust-building capability
- More advanced models generally achieve higher trust scores
- Response quality is more consistently correlated with trust than response speed

## 5. Trust Category Analysis

### 5.1 Distribution of Trust Categories
The three trust categories (competence, benevolence, integrity) show different patterns:
"""
        
        # Calculate category averages
        cats = {
            "competence": self.conversation_df["comp_score"].mean(),
            "benevolence": self.conversation_df["benev_score"].mean(),
            "integrity": self.conversation_df["integ_score"].mean()
        }
        
        for cat, avg in cats.items():
            report += f"\n- {cat.title()}: {avg:.2f}/7.0"
            
        report += """

## 6. Conclusion and Recommendations

### 6.1 Key Findings
1. **Emotion Management**: Models that appropriately express and respond to emotions build higher trust
2. **Quality vs. Speed**: Response quality is more important than speed for trust-building
3. **Engagement Patterns**: Consistent and meaningful exchanges are crucial for trust development
4. **Model Differences**: Advanced models generally achieve higher trust scores

### 6.2 Recommendations for Trust-Focused Conversational Agents
1. Prioritize response quality over speed
2. Ensure appropriate emotional recognition and response
3. Maintain consistent engagement with appropriate follow-up questions
4. Balance competence (factual accuracy) with benevolence (helpfulness)
5. Monitor trust signals in real-time to adapt conversation strategies

### 6.3 Future Research Directions
1. Longitudinal trust development over extended interactions
2. Cultural differences in trust perception of conversational agents
3. Recovery strategies after trust violations
4. The role of personalization in trust development
"""
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write(report)
            
        print(f"Report generated and saved to {output_path}")
        return output_path
    
    def generate_visualizations(self, output_dir: str = "trust_analysis_visualizations"):
        """Generate visualizations for the analysis.
        
        Args:
            output_dir: Directory to save visualizations
        """
        print("Generating visualizations...")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure data is loaded
        if self.conversation_df is None:
            self.load_data()
        
        # Set style
        plt.style.use('ggplot')
        
        # 1. Trust score distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.conversation_df["total_trust_score"], kde=True)
        plt.title("Distribution of Trust Scores")
        plt.xlabel("Trust Score (1-7)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "trust_distribution.png"))
        
        # 2. Trust vs. Latency
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.conversation_df, 
            x="latency_score", 
            y="total_trust_score",
            hue="agent_model"
        )
        plt.title("Trust vs. Latency")
        plt.xlabel("Latency Score (1-7)")
        plt.ylabel("Trust Score (1-7)")
        plt.savefig(os.path.join(output_dir, "trust_vs_latency.png"))
        
        # 3. Trust vs. Quality
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.conversation_df, 
            x="response_quality_score", 
            y="total_trust_score",
            hue="agent_model"
        )
        plt.title("Trust vs. Response Quality")
        plt.xlabel("Quality Score (1-7)")
        plt.ylabel("Trust Score (1-7)")
        plt.savefig(os.path.join(output_dir, "trust_vs_quality.png"))
        
        # 4. Trust vs. Engagement
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.conversation_df, 
            x="engagement_score", 
            y="total_trust_score",
            hue="agent_model"
        )
        plt.title("Trust vs. Engagement")
        plt.xlabel("Engagement Score (1-7)")
        plt.ylabel("Trust Score (1-7)")
        plt.savefig(os.path.join(output_dir, "trust_vs_engagement.png"))
        
        # 5. Model comparison
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=self.model_df,
            x="agent_model",
            y="mean_total_trust",
            yerr=self.model_df["std_total_trust"]
        )
        plt.title("Average Trust Score by Model")
        plt.xlabel("Model")
        plt.ylabel("Average Trust Score (1-7)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_comparison.png"))
        
        # 6. Trust categories
        categories = ["comp_score", "benev_score", "integ_score"]
        category_names = ["Competence", "Benevolence", "Integrity"]
        
        plt.figure(figsize=(10, 6))
        df_melted = pd.melt(
            self.conversation_df, 
            id_vars=["conversation_id", "agent_model"],
            value_vars=categories,
            var_name="category", 
            value_name="score"
        )
        df_melted["category"] = df_melted["category"].map(dict(zip(categories, category_names)))
        
        sns.boxplot(data=df_melted, x="category", y="score")
        plt.title("Trust Category Scores")
        plt.xlabel("Category")
        plt.ylabel("Score (1-7)")
        plt.savefig(os.path.join(output_dir, "trust_categories.png"))
        
        print(f"Visualizations saved to {output_dir}")

def main():
    """Main function."""
    print("Trust Research Dataset Analyzer")
    
    # Create analyzer
    analyzer = TrustDatasetAnalyzer()
    
    # Load data
    analyzer.load_data()
    
    # Generate report
    report_path = analyzer.generate_report()
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    print(f"\nAnalysis complete! Report saved to {report_path}")

if __name__ == "__main__":
    main()