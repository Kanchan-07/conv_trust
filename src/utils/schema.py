"""
Dataset schema definitions for trust research.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import datetime

@dataclass
class TrustCategoryScores:
    """Trust scores broken down by category."""
    competence: Optional[float] = None
    benevolence: Optional[float] = None
    integrity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "competence": self.competence,
            "benevolence": self.benevolence,
            "integrity": self.integrity
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrustCategoryScores':
        """Create instance from dictionary."""
        return cls(
            competence=data.get("competence"),
            benevolence=data.get("benevolence"),
            integrity=data.get("integrity")
        )

@dataclass
class Turn:
    """A single turn in a conversation."""
    turn_id: int
    speaker: str  # "user" or "agent"
    utterance: str
    response_time: Optional[float] = None  # in seconds, for agent turns
    emotion_detected: Optional[str] = None
    trust_score: Optional[float] = None  # 1-7 scale, only for agent turns
    trust_category_scores: Optional[TrustCategoryScores] = None  # only for agent turns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "utterance": self.utterance,
            "response_time": self.response_time,
            "emotion_detected": self.emotion_detected,
            "trust_score": self.trust_score,
            "trust_category_scores": self.trust_category_scores.to_dict() if self.trust_category_scores else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        """Create instance from dictionary."""
        turn = cls(
            turn_id=data["turn_id"],
            speaker=data["speaker"],
            utterance=data["utterance"],
            response_time=data.get("response_time"),
            emotion_detected=data.get("emotion_detected"),
            trust_score=data.get("trust_score")
        )
        if data.get("trust_category_scores"):
            turn.trust_category_scores = TrustCategoryScores.from_dict(data["trust_category_scores"])
        return turn

@dataclass
class EmotionDistribution:
    """Distribution of emotions in a conversation."""
    counts: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return self.counts
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'EmotionDistribution':
        """Create instance from dictionary."""
        return cls(counts=data)

@dataclass
class ConversationMetadata:
    """Metadata for a conversation."""
    conversation_id: str
    agent_model: str
    user_id: str
    scenario: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    total_turns: int = 0
    total_trust_score: Optional[float] = None
    trust_category_scores: Optional[TrustCategoryScores] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "agent_model": self.agent_model,
            "user_id": self.user_id,
            "scenario": self.scenario,
            "timestamp": self.timestamp,
            "total_turns": self.total_turns,
            "total_trust_score": self.total_trust_score,
            "trust_category_scores": self.trust_category_scores.to_dict() if self.trust_category_scores else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMetadata':
        """Create instance from dictionary."""
        meta = cls(
            conversation_id=data["conversation_id"],
            agent_model=data["agent_model"],
            user_id=data["user_id"],
            scenario=data["scenario"],
            timestamp=data.get("timestamp", datetime.datetime.now().isoformat()),
            total_turns=data.get("total_turns", 0),
            total_trust_score=data.get("total_trust_score")
        )
        if data.get("trust_category_scores"):
            meta.trust_category_scores = TrustCategoryScores.from_dict(data["trust_category_scores"])
        return meta

@dataclass
class ConversationData:
    """Data for a conversation."""
    conversation_id: str
    average_trust_score: Optional[float] = None
    trust_category_averages: Optional[TrustCategoryScores] = None
    engagement_score: Optional[float] = None
    emotion_distribution: Optional[EmotionDistribution] = None
    response_quality_score: Optional[float] = None
    latency_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "average_trust_score": self.average_trust_score,
            "trust_category_averages": self.trust_category_averages.to_dict() if self.trust_category_averages else None,
            "engagement_score": self.engagement_score,
            "emotion_distribution": self.emotion_distribution.to_dict() if self.emotion_distribution else None,
            "response_quality_score": self.response_quality_score,
            "latency_score": self.latency_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationData':
        """Create instance from dictionary."""
        conv_data = cls(
            conversation_id=data["conversation_id"],
            average_trust_score=data.get("average_trust_score"),
            engagement_score=data.get("engagement_score"),
            response_quality_score=data.get("response_quality_score"),
            latency_score=data.get("latency_score")
        )
        if data.get("trust_category_averages"):
            conv_data.trust_category_averages = TrustCategoryScores.from_dict(data["trust_category_averages"])
        if data.get("emotion_distribution"):
            conv_data.emotion_distribution = EmotionDistribution.from_dict(data["emotion_distribution"])
        return conv_data

@dataclass
class Conversation:
    """A complete conversation with metadata, turns, and aggregated data."""
    metadata: ConversationMetadata
    turns: List[Turn] = field(default_factory=list)
    data: Optional[ConversationData] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
            "data": self.data.to_dict() if self.data else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Create instance from dictionary."""
        conv = cls(
            metadata=ConversationMetadata.from_dict(data["metadata"])
        )
        if "turns" in data:
            conv.turns = [Turn.from_dict(turn_data) for turn_data in data["turns"]]
        if "data" in data and data["data"]:
            conv.data = ConversationData.from_dict(data["data"])
        return conv
    
    def save(self, filepath: str):
        """Save conversation to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Conversation':
        """Load conversation from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
