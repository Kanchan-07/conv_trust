# Trust Research in Conversational Agents - Dataset Generator

## Abstract

Assessing Trust in Conversational Agents through Sentiment, 
Emotion, Response Quality, Latency, and Engagement - A Mixed 
Synthetic and Human-Intervened Dataset Study 
ABSTRACT 
In this agentic era, trust in conversational agents plays a critical role in their adoption 
and effectiveness across diverse applications. This research focuses on assessing trust 
in conversational agents by analyzing the impact of sentiment, emotion, response 
quality, latency, and user engagement. Synthetic and human-intervened datasets were 
generated using a range of proprietary and open-source large language models 
(LLMs), including GPT-4, Claude, Llama, Gemini, and others. A total of 3,909 
conversations with 26,329 turns were created, simulating diverse conversational 
scenarios. Trust evaluations were conducted by trained human annotators (n=15) using 
a standardized framework, scoring perceived competence, benevolence, integrity, and 
overall trustworthiness on a 1–7 scale. Inter-annotator agreement was established with 
Cohen’s Kappa coefficients (target κ > 0.75). Sentiment analysis and emotion 
detection were employed to derive trust metrics, while contextual factors influencing 
trust judgments were documented qualitatively. The findings reveal that sentiment and 
emotion significantly influence trust perceptions, while response latency impacts user 
satisfaction. Engagement patterns, such as conversation length and emotional 
variation, were found to correlate with trust levels. The research also developed a 
modular codebase to support reproducible experimentation and extensibility, enabling 
integration of new models and metrics. This study provides insights into designing 
trustworthy conversational agents and contributes to the broader field of trust 
engineering by offering methodologies for quantifying and optimizing trust in AI 
systems.    


## Overview

This project generates a synthetic dataset for studying trust in conversational agents, based on the research objectives:

1. **Assessing the Impact of Sentiment and Emotion on Trust Levels**
2. **Analyzing Response Quality and Latency in Trust Building**
3. **Investigating User Engagement as Trust Indicator**
4. **Real-Time Trust Estimation in Live Conversations**

The dataset includes conversations with detailed trust metrics at both turn and conversation levels.

## Project Structure

```
conv_trust/
├── data/                      # Generated dataset
│   ├── conversations/         # Individual conversation files
│   ├── aggregated/            # Aggregated data
│   └── dataset_info.json      # Dataset metadata
├── src/                       # Source code
│   ├── analyzers/             # Trust metrics analyzers
│   ├── generators/            # Conversation generation
│   ├── models/                # LLM model interfaces
│   └── utils/                 # Utility functions
├── .env                       # Environment variables
├── generate_dataset.py        # Main script
└── requirements.txt           # Dependencies
```

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### Generate the full dataset

```bash
python generate_dataset.py --num-conversations 50 --min-turns 4 --max-turns 10 --models gemini-2.0-flash gemini-1.5-flash
```

Parameters:
- `--output-dir`: Directory to save the generated dataset (default: `data`)
- `--num-conversations`: Number of conversations to generate (default: 20)
- `--min-turns`: Minimum turns per conversation (default: 4)
- `--max-turns`: Maximum turns per conversation (default: 10)
- `--models`: Models to use for generation (default: gemini-2.0-flash)

Available models:
- `gemini-2.0-flash` (15 RPM, 1500 daily)
- `gemini-2.0-flash-lite` (30 RPM, 1500 daily)
- `gemini-2.5-flash-preview-04-17` (10 RPM, 500 daily)
- `gemini-2.5-pro-preview-03-25` (5 RPM, 25 daily)
- `gemini-1.5-flash` (15 RPM, 1500 daily)
- `gemini-1.5-flash-8b` (15 RPM, 1500 daily)
- `gemma-3-27b-it` (5 RPM, 500 daily)

### Generate a single example

Running without arguments will generate a default dataset:

```bash
python generate_dataset.py
```

## Dataset Schema

### Conversation Metadata
- `conversation_id`: Unique identifier
- `agent_model`: Name of the conversational agent model
- `user_id`: User identifier
- `scenario`: Conversation context
- `timestamp`: Date and time
- `total_turns`: Total number of turns
- `total_trust_score`: Overall trust score
- `trust_category_scores`: Scores by category (competence, benevolence, integrity)

### Turn-Level Data
- `turn_id`: Unique identifier for the turn
- `speaker`: User or agent
- `utterance`: Text content
- `response_time`: Time taken by agent to respond (in seconds)
- `emotion_detected`: Emotion in the utterance
- `trust_score`: Trust score for the turn (1-7 scale)
- `trust_category_scores`: Trust scores by category

### Conversation-Level Data
- `conversation_id`: Unique identifier
- `average_trust_score`: Average trust score across turns
- `trust_category_averages`: Average trust scores by category
- `engagement_score`: Engagement score (1-7 scale)
- `emotion_distribution`: Distribution of emotions
- `response_quality_score`: Response quality score (1-7 scale)
- `latency_score`: Latency score (1-7 scale)

## Example Usage for Analysis

After generating the dataset, you can analyze it with:

```python
import json
import os
import pandas as pd

# Load a single conversation
with open('data/conversations/conv_0001.json', 'r') as f:
    conversation = json.load(f)

# Extract turn-level data
turns = pd.json_normalize(conversation['turns'])

# Extract conversation-level metrics
metrics = pd.json_normalize(conversation['data'])
```

## Extending the Dataset

To add more scenarios or models:
1. Edit `src/utils/config.py` to add new scenarios or model configurations
2. Use the command-line interface to generate data with new settings

## Future Improvements

- Add more sophisticated emotion detection models
- Implement more nuanced trust scoring algorithms
- Add support for additional LLM providers
- Create visualization tools for trust metrics
