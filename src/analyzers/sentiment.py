"""
Sentiment and emotion analysis for conversation trust research.
"""
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from transformers import pipeline

# Load Hugging Face pipelines
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=3  # Get top 3 emotions for more nuanced analysis
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

quality_pipeline = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

class SentimentAnalyzer:
    """Analyzer for sentiment, emotion, and trust scores."""
    
    @staticmethod
    def detect_emotion(text: str) -> Dict[str, Any]:
        """Detect emotions in text, flattening nested pipeline output."""
        
        try:
            raw = emotion_pipeline(text)
            # Handle nested lists from pipeline
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                results = raw[0]
            else:
                results = raw
            # Ensure results is a list of dicts
            if isinstance(results, dict):
                results = [results]
            # Extract top emotions
            emotions = {item["label"]: item["score"] for item in results if isinstance(item, dict)}
            
            # Find the dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            return {
                "dominant_emotion": dominant_emotion[0],
                "dominant_score": dominant_emotion[1],
                "emotions": emotions
            }
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            return {"dominant_emotion": "neutral", "dominant_score": 1.0, "emotions": {"neutral": 1.0}}
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis
        """
        try:
            result = sentiment_pipeline(text)[0]
            
            # Extract the star rating (1-5) from the label
            label = result["label"]
            stars = int(label.split()[0])
            
            # Map 1-5 stars to 1-7 trust scale
            trust_score = round((stars - 1) * (6/4) + 1, 2)
            
            return {
                "label": label,
                "score": result["score"],
                "stars": stars,
                "trust_score": trust_score
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                "label": "3 stars",
                "score": 1.0,
                "stars": 3,
                "trust_score": 4.0
            }

    @staticmethod
    def generate_trust_category_scores(base_score: float, variation: float = 0.5) -> Dict[str, float]:
        """Generate trust category scores based on a base score.
        
        Args:
            base_score: Base trust score (1-7)
            variation: Amount of variation between categories
            
        Returns:
            Dictionary with trust category scores
        """
        # Ensure the base score is within 1-7 range
        base_score = max(1.0, min(7.0, base_score))
        
        # Random variations for each category
        competence_var = random.uniform(-variation, variation)
        benevolence_var = random.uniform(-variation, variation)
        integrity_var = random.uniform(-variation, variation)
        
        # Calculate scores with variation, ensuring they stay in the 1-7 range
        competence = max(1.0, min(7.0, base_score + competence_var))
        benevolence = max(1.0, min(7.0, base_score + benevolence_var))
        integrity = max(1.0, min(7.0, base_score + integrity_var))
        
        return {
            "competence": round(competence, 2),
            "benevolence": round(benevolence, 2),
            "integrity": round(integrity, 2)
        }
    
    @staticmethod
    def generate_response_time(quality_score: float, 
                              min_time: float = 0.5, 
                              max_time: float = 6.0) -> float:
        """Generate realistic response time based on response quality.
        
        Higher quality responses tend to take a bit longer, with some randomness.
        
        Args:
            quality_score: Quality score (1-7)
            min_time: Minimum response time in seconds
            max_time: Maximum response time in seconds
            
        Returns:
            Response time in seconds
        """
        # Normalize quality score to 0-1 range
        norm_quality = (quality_score - 1) / 6
        
        # Base response time increases with quality
        base_time = min_time + norm_quality * (max_time - min_time)
        
        # Add randomness (±30%)
        randomness = random.uniform(-0.3, 0.3) * base_time
        
        # Final time with limits
        final_time = max(min_time, min(max_time, base_time + randomness))
        
        return round(final_time, 2)
    
    @staticmethod
    def calculate_engagement_score(conversation: List[Dict[str, Any]]) -> float:
        """Calculate engagement score based on conversation patterns.
        
        Factors:
        - Conversation length
        - User response lengths
        - User question frequency
        - Emotional variation
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Engagement score (1-7)
        """
        if not conversation:
            return 4.0  # Default middle score
        
        # Count turns
        num_turns = len(conversation)
        turns_score = min(7, 1 + (num_turns / 3))  # More turns = higher engagement
        
        # Average user message length
        user_msgs = [turn for turn in conversation if turn.get("speaker") == "user"]
        if user_msgs:
            avg_user_length = sum(len(msg.get("utterance", "")) for msg in user_msgs) / len(user_msgs)
            length_score = min(7, 1 + (avg_user_length / 20))  # Longer messages = higher engagement
        else:
            length_score = 4.0
        
        # Question frequency (simple heuristic)
        question_count = sum(1 for msg in user_msgs if "?" in msg.get("utterance", ""))
        if user_msgs:
            question_ratio = question_count / len(user_msgs)
            question_score = 1 + (question_ratio * 6)  # More questions = higher engagement
        else:
            question_score = 4.0
        
        # Emotion variation
        emotions = [turn.get("emotion_detected", "neutral") for turn in conversation]
        unique_emotions = len(set(emotions))
        emotion_score = min(7, 1 + unique_emotions)  # More emotion variety = higher engagement
        
        # Combine scores
        final_score = (turns_score + length_score + question_score + emotion_score) / 4
        
        return round(final_score, 2)
    
    @staticmethod
    def calculate_aggregated_trust_metrics(conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated trust metrics for a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Extract agent turns with trust scores
        agent_turns = [turn for turn in conversation 
                     if turn.get("speaker") == "agent" and turn.get("trust_score") is not None]
        
        if not agent_turns:
            return {
                "average_trust_score": None,
                "trust_category_averages": {
                    "competence": None,
                    "benevolence": None,
                    "integrity": None
                },
                "response_quality_score": None,
                "latency_score": None
            }
        
        # Calculate average trust score
        trust_scores = [turn.get("trust_score", 0) for turn in agent_turns]
        avg_trust = sum(trust_scores) / len(trust_scores)
        
        # Calculate category averages
        competence_scores = []
        benevolence_scores = []
        integrity_scores = []
        
        for turn in agent_turns:
            categories = turn.get("trust_category_scores", {})
            if categories:
                if categories.get("competence") is not None:
                    competence_scores.append(categories.get("competence"))
                if categories.get("benevolence") is not None:
                    benevolence_scores.append(categories.get("benevolence"))
                if categories.get("integrity") is not None:
                    integrity_scores.append(categories.get("integrity"))
        
        # Calculate averages for each category
        avg_competence = sum(competence_scores) / len(competence_scores) if competence_scores else None
        avg_benevolence = sum(benevolence_scores) / len(benevolence_scores) if benevolence_scores else None
        avg_integrity = sum(integrity_scores) / len(integrity_scores) if integrity_scores else None
        
        # Response times
        response_times = [turn.get("response_time", 0) for turn in agent_turns if turn.get("response_time") is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        # Map response time to latency score (faster = higher score)
        if avg_response_time is not None:
            # Convert avg time (expected 0.5-6s) to 1-7 scale (inverted: faster = higher score)
            latency_score = max(1, min(7, 7 - (avg_response_time - 0.5)))
        else:
            latency_score = None
        
        # Use trust score as proxy for quality score with slight variation
        if avg_trust is not None:
            quality_variation = random.uniform(-0.5, 0.5)
            quality_score = max(1, min(7, avg_trust + quality_variation))
        else:
            quality_score = None
        
        return {
            "average_trust_score": round(avg_trust, 2) if avg_trust is not None else None,
            "trust_category_averages": {
                "competence": round(avg_competence, 2) if avg_competence is not None else None,
                "benevolence": round(avg_benevolence, 2) if avg_benevolence is not None else None,
                "integrity": round(avg_integrity, 2) if avg_integrity is not None else None
            },
            "response_quality_score": round(quality_score, 2) if quality_score is not None else None,
            "latency_score": round(latency_score, 2) if latency_score is not None else None
        }
    
    @staticmethod
    def count_emotions(conversation: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count occurrences of each emotion in the conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with emotion counts
        """
        emotion_counts = {}
        
        for turn in conversation:
            emotion = turn.get("emotion_detected")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return emotion_counts
