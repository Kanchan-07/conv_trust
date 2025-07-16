"""
Conversation generator for trust dataset.
"""
import os
import json
import random
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    # Try to import the real LLM implementation first
    from src.models.llm import LLMModelInterface
except ImportError as e:
    # If that fails, use the mock implementation
    print(f"Using mock LLM model due to import error: {e}")
    from src.models.mock_llm import MockLLMModelInterface as LLMModelInterface
try:
    # Try to import the real sentiment analyzer first
    from src.analyzers.sentiment import SentimentAnalyzer
except ImportError as e:
    # If that fails, use the mock implementation
    print(f"Using mock sentiment analyzer due to import error: {e}")
    from src.analyzers.mock_sentiment import MockSentimentAnalyzer as SentimentAnalyzer
from src.utils.schema import (
    Conversation, ConversationMetadata, ConversationData, 
    Turn, TrustCategoryScores, EmotionDistribution
)

class ConversationGenerator:
    """Generator for synthetic conversations."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize conversation generator.
        
        Args:
            model_name: LLM model to use for generation
        """
        # Choose local or remote LLM based on prefix
        if model_name.startswith("local:"):
            from src.models.local import LocalLLMModelInterface
            local_name = model_name.split("local:", 1)[1]
            self.model = LocalLLMModelInterface(local_name)
        else:
            self.model = LLMModelInterface(model_name)
        self.model_name = model_name
        self.analyzer = SentimentAnalyzer()
    
    def generate_conversation_prompt(self, scenario: str, num_turns: int = 6) -> str:
        """Generate prompt for conversation generation.
        
        Args:
            scenario: Conversation scenario
            num_turns: Number of conversation turns
            
        Returns:
            Prompt for LLM
        """
        return f"""
        Create a realistic conversation between a user and an AI assistant for the following scenario: {scenario}.
        
        The conversation should have {num_turns} turns total (a turn is one user message and one assistant response).
        Make the conversation natural and realistic.
        
        - The user should occasionally express emotions like frustration, satisfaction, confusion, etc.
        - The assistant should be helpful and professional but occasionally have varying response quality
        - Include at least one instance where the user asks a challenging question

        Structure the conversation as a JSON object with the following format:
        {{
            "turns": [
                {{
                    "turn_id": 1,
                    "speaker": "user",
                    "utterance": "user message here"
                }},
                {{
                    "turn_id": 2,
                    "speaker": "agent",
                    "utterance": "agent response here"
                }},
                ...and so on
            ]
        }}
        
        Only provide the valid JSON response, nothing else.
        """
    
    def generate_raw_conversation(self, scenario: str, num_turns: int = 6) -> Dict[str, Any]:
        """Generate raw conversation data using LLM.
        
        Args:
            scenario: Conversation scenario
            num_turns: Number of conversation turns
            
        Returns:
            Raw conversation data
        """
        prompt = self.generate_conversation_prompt(scenario, num_turns)
        
        # Set system instruction to encourage proper JSON formatting
        system_instruction = "You are a conversation generator that produces realistic dialog. Always output valid JSON."
        
        # Generate raw conversation
        default_value = {"turns": []}
        raw_conversation = self.model.generate_json(prompt, system_instruction, default_value)
        
        return raw_conversation
    
    def annotate_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Annotate a conversation turn with trust metrics.
        
        Args:
            turn: Conversation turn data
            
        Returns:
            Annotated turn
        """
        utterance = turn.get("utterance", "")
        speaker = turn.get("speaker", "user")
        
        # Analyze emotion
        emotion_data = self.analyzer.detect_emotion(utterance)
        turn["emotion_detected"] = emotion_data["dominant_emotion"]
        
        # For agent turns, add trust-related metrics
        if speaker == "agent":
            # Sentiment analysis maps to trust score
            sentiment = self.analyzer.analyze_sentiment(utterance)
            turn["trust_score"] = sentiment["trust_score"]
            
            # Generate trust category scores
            turn["trust_category_scores"] = self.analyzer.generate_trust_category_scores(
                sentiment["trust_score"]
            )
            
            # Generate response time
            turn["response_time"] = self.analyzer.generate_response_time(sentiment["trust_score"])
        
        return turn
    
    def generate_conversation(self, scenario: str, user_id: str = None, num_turns: int = 6) -> Conversation:
        """Generate complete annotated conversation.
        
        Args:
            scenario: Conversation scenario
            user_id: User identifier (generated if None)
            num_turns: Number of conversation turns
            
        Returns:
            Complete annotated Conversation object
        """
        # Generate user ID if not provided
        if user_id is None:
            user_id = f"user_{str(uuid.uuid4())[:8]}"
        
        # Generate conversation ID
        conversation_id = f"conv_{str(uuid.uuid4())[:8]}"
        
        # Create metadata
        metadata = ConversationMetadata(
            conversation_id=conversation_id,
            agent_model=self.model_name,
            user_id=user_id,
            scenario=scenario,
            timestamp=datetime.datetime.now().isoformat(),
            total_turns=num_turns
        )
        
        # Generate raw conversation
        raw_conversation = self.generate_raw_conversation(scenario, num_turns)
        
        # Convert to Turn objects with annotations
        turns = []
        for turn_data in raw_conversation.get("turns", []):
            # Annotate the turn
            annotated_turn = self.annotate_turn(turn_data)
            
            # Convert to Turn object
            turn = Turn(
                turn_id=annotated_turn["turn_id"],
                speaker=annotated_turn["speaker"],
                utterance=annotated_turn["utterance"],
                response_time=annotated_turn.get("response_time"),
                emotion_detected=annotated_turn.get("emotion_detected"),
                trust_score=annotated_turn.get("trust_score")
            )
            
            # Add trust category scores if present
            if "trust_category_scores" in annotated_turn:
                scores = annotated_turn["trust_category_scores"]
                turn.trust_category_scores = TrustCategoryScores(
                    competence=scores.get("competence"),
                    benevolence=scores.get("benevolence"),
                    integrity=scores.get("integrity")
                )
            
            turns.append(turn)
        
        # Create Conversation object
        conversation = Conversation(metadata=metadata, turns=turns)
        
        # Calculate conversation-level metrics
        self.add_conversation_metrics(conversation)
        
        return conversation
    
    def add_conversation_metrics(self, conversation: Conversation) -> None:
        """Add conversation-level metrics to a Conversation object.
        
        Args:
            conversation: Conversation to analyze
            
        Side effects:
            Updates conversation.data and conversation.metadata
        """
        # Convert turns to dictionaries for analysis
        turn_dicts = [turn.to_dict() for turn in conversation.turns]
        
        # Calculate aggregated metrics
        metrics = self.analyzer.calculate_aggregated_trust_metrics(turn_dicts)
        
        # Calculate engagement score
        engagement_score = self.analyzer.calculate_engagement_score(turn_dicts)
        
        # Count emotions
        emotion_counts = self.analyzer.count_emotions(turn_dicts)
        
        # Create ConversationData object
        data = ConversationData(
            conversation_id=conversation.metadata.conversation_id,
            average_trust_score=metrics["average_trust_score"],
            engagement_score=engagement_score,
            response_quality_score=metrics["response_quality_score"],
            latency_score=metrics["latency_score"]
        )
        
        # Add trust category averages if available
        if metrics["trust_category_averages"]:
            cat_scores = metrics["trust_category_averages"]
            data.trust_category_averages = TrustCategoryScores(
                competence=cat_scores.get("competence"),
                benevolence=cat_scores.get("benevolence"),
                integrity=cat_scores.get("integrity")
            )
        
        # Add emotion distribution
        data.emotion_distribution = EmotionDistribution(counts=emotion_counts)
        
        # Update conversation data
        conversation.data = data
        
        # Update metadata with overall trust score
        conversation.metadata.total_trust_score = data.average_trust_score
        
        # Update metadata with trust category scores
        if data.trust_category_averages:
            conversation.metadata.trust_category_scores = data.trust_category_averages
