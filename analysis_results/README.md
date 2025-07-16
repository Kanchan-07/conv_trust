# Conversation Data Analysis

This directory contains the results of analyzing the conversation dataset. The analysis includes various metrics and visualizations to understand the patterns and characteristics of the conversations.

## Files Generated

1. `conversation_metrics.png` - Visualizations showing:
   - Distribution of conversation lengths
   - Agent response times
   - Trust scores by category
   - Emotion distribution by speaker
   - Word count vs. readability

2. `wordcloud_user.png` - Word cloud of user utterances
3. `wordcloud_agent.png` - Word cloud of agent utterances
4. `metrics_summary.txt` - Summary statistics of the analysis

## Key Metrics

Key metrics include:
- Average conversation length
- Response time statistics
- Trust score distributions (overall, competence, benevolence, integrity)
- Emotion distribution for both users and agents
- Text analysis metrics (word count, readability)

## How to Run the Analysis

1. Ensure you have Python 3.8+ installed
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the analysis script:
   ```
   python analyze_data.py
   ```

The results will be saved in the `analysis_results` directory.

## Notes

- The analysis assumes the conversation data is in the `data/conversations` directory
- The script handles JSON-formatted conversation files
- Visualizations are saved as high-resolution PNG files for publication quality
