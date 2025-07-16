"""
Dataset generator for trust research.
"""
import os
import json
import random
import time
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

from src.utils.config import ensure_output_dirs, DatasetConfig, GEMINI_MODELS
from src.generators.conversation import ConversationGenerator

# Synthetic model names for metadata labeling
FAKE_MODEL_NAMES = [
    "Claude 3.7 Sonnet",
    "Claude 3.7 Sonnet-Thinking",
    "Llama 4 scout",
    "Llama 3.3 70B",
    "mistral 3 24b",
    "llama 3.1 27b",
    "Claude 3.5 Sonnet",
    "gpt 4",
    "gpt 4.5",
    "o3",
    "o3 mini",
    "o4",
    "o4 mini high",
    "Grok-3",
    "Qwen 2.5-Max",
    "Qwen2.5-VL-32B-Instruct",
    "DeepSeek R2",
    "DeepSeek V4",
    "Velvet 14B",
    "Falcon 7b"
]

class DatasetGenerator:
    """Generator for complete trust dataset."""
    
    def __init__(self, config: DatasetConfig = None):
        """Initialize dataset generator.
        
        Args:
            config: Dataset configuration
        """
        self.config = config or DatasetConfig()
        ensure_output_dirs(self.config)
        
    def generate_conversation(self, model_name: str, scenario: str, user_id: str, conv_index: int) -> str:
        """Generate a single conversation using specified model.
        
        Args:
            model_name: LLM model to use
            scenario: Conversation scenario
            user_id: User identifier
            conv_index: Conversation index for ID generation
            
        Returns:
            Path to saved conversation file
        """
        # Create generator
        generator = ConversationGenerator(model_name)
        
        # Generate turns count in the specified range
        min_turns, max_turns = self.config.conversation_turns_range
        num_turns = random.randint(min_turns, max_turns)
        
        # Generate conversation
        conversation = generator.generate_conversation(
            scenario=scenario,
            user_id=user_id,
            num_turns=num_turns
        )
        
        # Create ID with index for uniqueness
        conversation.metadata.conversation_id = f"conv_{conv_index:04d}"
        # Override model name in metadata for synthetic labeling
        conversation.metadata.agent_model = random.choice(FAKE_MODEL_NAMES)
        
        # Save metadata and conversation content separately
        metadata_dir = os.path.join(self.config.output_dir, "metadata")
        conv_dir = os.path.join(self.config.output_dir, "conversations")
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(conv_dir, exist_ok=True)
        metadata_path = os.path.join(metadata_dir, f"{conversation.metadata.conversation_id}.json")
        conv_path = os.path.join(conv_dir, f"{conversation.metadata.conversation_id}.json")
        # Write metadata file
        with open(metadata_path, 'w', encoding='utf-8') as mf:
            json.dump(conversation.metadata.to_dict(), mf, indent=2, ensure_ascii=False)
        # Write conversation content (turns and data)
        conv_dict = conversation.to_dict()
        conv_content = {"turns": conv_dict.get("turns"), "data": conv_dict.get("data")}
        with open(conv_path, 'w', encoding='utf-8') as cf:
            json.dump(conv_content, cf, indent=2, ensure_ascii=False)
        print(f"Generated conversation {conversation.metadata.conversation_id} using actual model {model_name}")
        return conv_path
    
    def generate_dataset(self, model_names: List[str] = None) -> Dict[str, Any]:
        """Generate complete dataset.
        
        Args:
            model_names: List of models to use (uses all available if None)
            
        Returns:
            Dataset statistics
        """
        try:
            # Default to all models if none specified
            if not model_names:
                model_names = list(GEMINI_MODELS.keys())
            
            print(f"Using models: {model_names}")
            
            # Validate models
            valid_models = [m for m in model_names if m in GEMINI_MODELS]
            if not valid_models:
                raise ValueError(f"No valid models specified. Available models: {list(GEMINI_MODELS.keys())}")
            
            print(f"Valid models: {valid_models}")
            
            # Statistics for tracking
            stats = {
                "total_conversations": self.config.num_conversations,
                "models_used": valid_models,
                "conversations_per_model": {},
                "conversations_per_scenario": {},
                "filepaths": []
            }
            
            # Distribute conversations across models
            conversations_per_model = self.config.num_conversations // len(valid_models)
            remainder = self.config.num_conversations % len(valid_models)
            
            print(f"Conversations per model: {conversations_per_model}, remainder: {remainder}")
            
            model_counts = {}
            for i, model in enumerate(valid_models):
                count = conversations_per_model
                if i < remainder:
                    count += 1
                model_counts[model] = count
                stats["conversations_per_model"][model] = count
            
            print(f"Model counts: {model_counts}")
            
            # Determine starting conversation index based on existing files
            conv_dir = os.path.join(self.config.output_dir, "conversations")
            os.makedirs(conv_dir, exist_ok=True)
            existing_files = os.listdir(conv_dir)
            max_idx = -1
            for fname in existing_files:
                if fname.startswith("conv_") and fname.endswith(".json"):
                    idx_str = fname[len("conv_"):-5]
                    try:
                        idx = int(idx_str)
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        continue
            start_index = max_idx + 1
            conv_index = start_index
            
            print(f"Starting from conversation index: {conv_index}")
            
            # Create user IDs aligned with conversation indexes
            user_ids = {}
            for i in range(self.config.num_conversations):
                user_ids[conv_index + i] = f"user_{conv_index + i:04d}"
            
            # Distribute scenarios and map them to conversation indexes
            scenario_assignments = {}
            if not self.config.scenarios:
                raise ValueError("No scenarios defined in configuration")
                
            print(f"Available scenarios: {len(self.config.scenarios)}")
            
            for i in range(self.config.num_conversations):
                scenario = random.choice(self.config.scenarios)
                scenario_assignments[conv_index + i] = scenario
                stats["conversations_per_scenario"][scenario] = stats["conversations_per_scenario"].get(scenario, 0) + 1
            
            # Generate conversations
            current_idx = conv_index
            total_generated = 0
            
            for model, count in model_counts.items():
                print(f"Generating {count} conversations with model {model}")
                model_generated = 0
                
                while model_generated < count and total_generated < self.config.num_conversations:
                    # Make sure we have the current index in our scenario and user mappings
                    if current_idx not in scenario_assignments or current_idx not in user_ids:
                        print(f"Warning: Missing scenario or user ID for index {current_idx}")
                        current_idx += 1
                        continue
                    
                    filepath = self.generate_conversation(
                        model_name=model,
                        scenario=scenario_assignments[current_idx],
                        user_id=user_ids[current_idx],
                        conv_index=current_idx
                    )
                    stats["filepaths"].append(filepath)
                    model_generated += 1
                    total_generated += 1
                    current_idx += 1
                    
                    # Add a small delay to avoid hitting rate limits
                    time.sleep(0.5)
                    
                print(f"Generated {model_generated} conversations with model {model}")
                
            print(f"Total conversations generated: {total_generated}")
            
            # If we didn't generate any conversations, return stats early
            if not stats["filepaths"]:
                return stats
            
            # Generate aggregated dataset information
            self._generate_dataset_info(stats)
            
            return stats
            
        except Exception as e:
            print(f"Error in generate_dataset: {e}")
            traceback.print_exc()
            raise
    
    def _generate_dataset_info(self, stats: Dict[str, Any]) -> None:
        """Generate aggregated dataset information.
        
        Args:
            stats: Dataset statistics
        """
        # Load all conversations
        conversations = []
        for filepath in stats["filepaths"]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conversations.append(json.load(f))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Create dataset info
        dataset_info = {
            "dataset_name": "Trust Research Synthetic Dataset",
            "creation_date": time.strftime("%Y-%m-%d"),
            "num_conversations": len(conversations),
            "models_used": stats["models_used"],
            "scenarios": list(stats["conversations_per_scenario"].keys()),
            "statistics": {
                "conversations_per_model": stats["conversations_per_model"],
                "conversations_per_scenario": stats["conversations_per_scenario"]
            },
            "schema_version": "1.0"
        }
        
        # Save dataset info
        info_path = os.path.join(self.config.output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
