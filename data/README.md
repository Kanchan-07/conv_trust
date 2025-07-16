# Trust Research Dataset (Conversational AI)

## 📌 Dataset Overview

This dataset contains synthetic conversations between humans and AI assistants, designed to facilitate research in trust dynamics within human-AI interactions. The dataset includes multiple conversation scenarios, each with detailed metadata and annotations.

## 📊 Dataset Statistics

- **Total Conversations**: 5
- **Total Language Models**: 22

### Model Categories

#### 1. Claude Series
- Claude 3.5 Sonnet
- Claude 3.7 Sonnet
- Claude 3.7 Sonnet-Thinking

#### 2. DeepSeek
- DeepSeek R2
- DeepSeek V4

#### 3. Falcon
- Falcon 7b (local)

#### 4. Grok
- Grok-3

#### 5. LLaMA
- LLaMA 3.3 70B
- LLaMA 4 Scout
- LLaMA 3.1 27B (local)

#### 6. Qwen
- Qwen 2.5 Max
- Qwen2.5-VL-32B-Instruct

#### 7. Velvet
- Velvet 14B (local)

#### 8. Gemini
- Gemini 2.0 Flash
- Gemini 2.0 Flash Lite
- Gemini 2.5 Flash Preview 04-17

#### 9. GPT
- GPT-4
- GPT-4.5

#### 10. Mistral
- Mistral 3 24B (local)

#### 11. OpenAI O-Series Reasoning Models
- O3
- O3 Mini
- O4
- O4 Mini High

### Model Selection Rationale

Models were selected to represent a diverse range of capabilities and architectures:

- **Base Models** (local deployment):
  - Falcon 7B
  - Velvet 14B
  - Mistral 3 24B
  - LLaMA 3.1 27B

- **Instruction-Tuned Models**:
  - GPT-4/4.5
  - Gemini series
  - Claude Sonnet variants
  - Qwen VL models

- **Multimodal Capabilities**:
  - Gemini 2.5 Flash
  - Qwen2.5-VL-32B-Instruct

- **Reasoning-Focused**:
  - OpenAI O-Series (O3, O3 Mini, O4, O4 Mini High)

- **Scenarios Covered`:
  - Bank account management
  - E-commerce Help Center
  - Flight booking assistance
  - Hackathon participation
  - Job application assistance

## 🗂 Dataset Structure

```
data/
├── conversations/         # Individual conversation files (JSON format)
├── metadata/             # Metadata for each conversation
└── dataset_info.json     # Dataset metadata and statistics
```

## 🔍 Annotation Protocol

### 4.1 Annotation Process
Conversations were annotated by a team of 15 trained human raters. Each agent turn received comprehensive annotations for multiple trust-related dimensions.

### 4.2 Annotation Dimensions
Each agent turn was labeled for the following attributes:

#### Trust Metrics
- **Overall Trust Score**: 7-point Likert scale (1 = Not at all trustworthy, 7 = Extremely trustworthy)
- **Trust Category Scores**:
  - **Competence**: Perceived capability and knowledge
  - **Benevolence**: Perceived goodwill and positive intent
  - **Integrity**: Perceived honesty and ethical behavior

#### Additional Annotations
- **Emotion Label**: Categorized from a predefined taxonomy
- **Sentiment Score**: Numerically mapped sentiment value
- **Response Latency**: Time taken for agent response

### 4.3 Quality Control
- **Inter-annotator Agreement**: Measured using Cohen's Kappa (κ)
- **Agreement Threshold**: Maintained at κ > 0.75 for all annotations
- **Continuous Monitoring**: Regular checks to ensure annotation consistency
- **Rater Training**: Comprehensive training and calibration sessions for all annotators

## 📄 Data Format

### Conversation Files
Each conversation is stored as a JSON file with the following structure:

```json
{
  "conversation_id": "unique_id",
  "model": "model_name",
  "scenario": "scenario_name",
  "turns": [
    {
      "speaker": "user|assistant",
      "text": "message content",
      "timestamp": "ISO-8601 timestamp"
    }
  ],
  "metadata": {
    "language": "en",
    "created_at": "ISO-8601 timestamp",
    "version": "1.0"
  }
}
```

### Metadata
Each conversation has associated metadata stored in the `metadata` directory, containing additional annotations and analysis results.

## 🔍 Dataset Information

The `dataset_info.json` file contains high-level information about the dataset:

```json
{
  "dataset_name": "Trust Research Synthetic Dataset",
  "creation_date": "2025-05-08",
  "num_conversations": 5,
  "models_used": ["gemini-2.0-flash-lite"],
  "scenarios": [
    "Bank account management",
    "E-commerce Help Center",
    "flight booking assistance",
    "Hackathon participation",
    "Job application assistance"
  ]
}
```

## 📝 Usage Examples

### Loading and Analyzing the Dataset

You can use the provided analysis tools to explore the dataset:

```python
from dataset_description import TrustDatasetAnalyzer

# Initialize the analyzer
analyzer = TrustDatasetAnalyzer("data")

# Load the data
conversation_df, turn_df = analyzer.load_data()

# Get summary statistics
summary_stats = analyzer.get_summary_statistics()
print(summary_stats)
```

## 📜 License

This dataset is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This means you are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

for any purpose, even commercially, under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For the full legal text, see the [LICENSE](LICENSE) file in this repository.

## 🤝 Contributing

Contributions to improve the dataset or analysis tools are welcome. Please submit a Pull Request with your proposed changes.

## 📧 Contact

For questions or feedback about this dataset, please contact [kanchan7489@gmail.com]

## 🔗 Citation

If you use this dataset in your research, please cite it as:

```
[Your Citation Here]
```
