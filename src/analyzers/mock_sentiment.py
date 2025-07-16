"""
Mock sentiment analyzer for trust research.
"""
import random
from typing import Dict, List, Any, Optional

class MockSentimentAnalyzer:
    """Mock analyzer for sentiment, emotion, and trust metrics."""
    
    def __init__(self):
        """Initialize the mock sentiment analyzer."""
        self.emotion_labels = [
            "joy", "sadness", "anger", "fear", "disgust", "surprise", 
            "neutral", "frustrated", "confused", "satisfied", "curious", "anxious"
        ]
    
    def detect_emotion(self, text: str) -> Dict[str, Any]:
        """Detect emotion in text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with emotion detection results
        """
        # Choose a random emotion with higher probability for neutral and satisfied
        weights = [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3, 0.1, 0.05, 0.2, 0.1, 0.05]
        emotion = random.choices(self.emotion_labels, weights=weights, k=1)[0]
        
        # Generate confidence score between 0.6 and 0.95
        confidence = random.uniform(0.6, 0.95)
        
        return {
            "dominant_emotion": emotion,
            "confidence": confidence,
            "all_emotions": {e: random.uniform(0.1, 0.3) for e in self.emotion_labels if e != emotion}
        }
    
    def calculate_trust_score(self, text: str) -> Dict[str, Any]:
        """Calculate trust score for text.
        
        Args:
            text: Input text
            
        Returns:
            Dict with trust scores
        """
        # Generate trust score between 3 and 7 (on 1-7 scale)
        trust_score = round(random.uniform(3, 7), 1)
        
        # Generate category scores between 3 and 7
        category_scores = {
            "competence": round(random.uniform(3, 7), 1),
            "benevolence": round(random.uniform(3, 7), 1),
            "integrity": round(random.uniform(3, 7), 1)
        }
        
        return {
            "trust_score": trust_score,
            "trust_category_scores": category_scores
        }
    
    def calculate_response_quality(self, text: str) -> float:
        """Calculate response quality score.
        
        Args:
            text: Input text
            
        Returns:
            Quality score (0-1)
        """
        # Generate quality score between 0.5 and 0.95
        return round(random.uniform(0.5, 0.95), 2)
    
    def calculate_engagement_score(self, turns: List[Dict[str, Any]]) -> float:
        """Calculate engagement score from conversation turns.
        
        Args:
            turns: List of conversation turns
            
        Returns:
            Engagement score (1-5)
        """
        # Generate engagement score between 3 and 5
        return round(random.uniform(3, 5), 1)
    
    def calculate_aggregated_trust_metrics(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated trust metrics from conversation turns.
        
        Args:
            turns: List of conversation turns
            
        Returns:
            Dict with aggregated metrics
        """
        # Generate average trust score between 3 and 7
        avg_trust = round(random.uniform(3, 7), 1)
        
        # Generate average category scores
        avg_categories = {
            "competence": round(random.uniform(3, 7), 1),
            "benevolence": round(random.uniform(3, 7), 1),
            "integrity": round(random.uniform(3, 7), 1)
        }
        
        # Generate quality and latency scores
        quality_score = round(random.uniform(0.5, 0.95), 2)
        latency_score = round(random.uniform(0.6, 0.9), 2)
        
        return {
            "average_trust_score": avg_trust,
            "trust_category_averages": avg_categories,
            "response_quality_score": quality_score,
            "latency_score": latency_score
        }
    
    def count_emotions(self, turns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count emotions in conversation turns.
        
        Args:
            turns: List of conversation turns
            
        Returns:
            Dict with emotion counts
        """
        # Generate random counts for emotions
        emotion_counts = {}
        available_emotions = random.sample(self.emotion_labels, k=random.randint(3, 6))
        
        for emotion in available_emotions:
            emotion_counts[emotion] = random.randint(1, 3)
            
        return emotion_counts
