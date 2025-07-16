"""
Configuration utilities for the trust dataset generator.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str
    rate_limit_rpm: int
    daily_limit: int
    supports_system_instruction: bool = True
    
@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    output_dir: str = "data"
    num_conversations: int = 100
    conversation_turns_range: tuple = (4, 10)
    trust_category_names: List[str] = field(default_factory=lambda: ["competence", "benevolence", "integrity"])
    emotion_labels: List[str] = field(default_factory=lambda: [
        "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral", 
        "frustrated", "confused", "satisfied", "curious", "anxious"
    ])
    scenarios: List[str] = field(default_factory=lambda: [
        "Customer service interaction about a delayed package",
        "Technical support for a malfunctioning device",
        "Health advice consultation",
        "Financial planning guidance",
        "Travel booking assistance",
        "Food ordering and recommendations",
        "Educational tutoring session",
        "Career counseling",
        "Legal advice consultation",
        "Home improvement guidance",
        "Customer support interaction about a billing issue",
        "Healthcare appointment scheduling",
        "Medical appointment scheduling",
        "Bank account management",
        "Insurance claim processing",
        "Ar denial processing",
        "Medical billing processing",
        "flight booking assistance",
        "hotel booking assistance",
        "restaurant booking assistance",
        "Fashion recommendations based on budget",
        "Grocery recommendations based on budget",
        "Music recommendations based on genre",
        "Book recommendations based on genre",
        "Movie recommendations based on genre",
        "Game recommendations based on genre",
        "Hackathon participation",
        "Job application assistance",
        "Resume writing assistance",
        "Interview preparation assistance",
        "E-commerce product recommendations",
        "E-commerce product recommendations based on budget",
        "E-commerce product recommendations based on category",
        "E-commerce Help Center",
        "Shoe recommendations based on style",
    ])

# Available Gemini model configurations
GEMINI_MODELS = {
    "gemini-2.0-flash": ModelConfig(
        name="gemini-2.0-flash",
        rate_limit_rpm=15,
        daily_limit=1500,
        supports_system_instruction=True
    ),
    "gemini-2.0-flash-lite": ModelConfig(
        name="gemini-2.0-flash-lite",
        rate_limit_rpm=30,
        daily_limit=1500,
        supports_system_instruction=True
    ),
    "gemini-2.5-flash-preview-04-17": ModelConfig(
        name="gemini-2.5-flash-preview-04-17",
        rate_limit_rpm=10,
        daily_limit=500,
        supports_system_instruction=True
    ),
    "gemini-2.5-pro-preview-03-25": ModelConfig(
        name="gemini-2.5-pro-preview-03-25",
        rate_limit_rpm=5,
        daily_limit=25,
        supports_system_instruction=True
    ),
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        rate_limit_rpm=15,
        daily_limit=1500,
        supports_system_instruction=True
    ),
    "gemini-1.5-flash-8b": ModelConfig(
        name="gemini-1.5-flash-8b",
        rate_limit_rpm=15,
        daily_limit=1500,
        supports_system_instruction=True
    ),
    "gemma-3-27b-it": ModelConfig(
        name="gemma-3-27b-it",
        rate_limit_rpm=5,
        daily_limit=500,
        supports_system_instruction=False
    )
}

# Default configuration
DEFAULT_CONFIG = DatasetConfig()

def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in GEMINI_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return GEMINI_MODELS[model_name]

def ensure_output_dirs(config: DatasetConfig = DEFAULT_CONFIG):
    """Ensure all required output directories exist."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "metadata"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "conversations"), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, "aggregated"), exist_ok=True)
