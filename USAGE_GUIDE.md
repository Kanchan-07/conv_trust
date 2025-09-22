# Trust Detection System - Usage Guide

## 🚀 Quick Start

### Step 1: Train Models (Run Once)
```bash
# Train all models and save them
uv run run_complete_pipeline.py
```
This will:
- Load your conversational dataset 
- Train multiple ML/DL models (Random Forest, XGBoost, SVM, LSTM, Ensemble)
- Save trained models to `pipeline_results/trained_models/`
- Generate evaluation reports and visualizations

### Step 2: Use Inference (Anytime)
```bash
# Run standalone inference with saved models
uv run trust_inference.py

# Interactive mode
uv run trust_inference.py --interactive
```

## 🎯 **Standalone Inference Features**

### **1. Load Pre-trained Models**
```python
from trust_inference import TrustInferenceEngine

# Initialize engine (loads all saved models)
engine = TrustInferenceEngine()

# Get model information
info = engine.get_model_info()
print(f"Available models: {info['available_models']}")
```

### **2. Single Prediction**
```python
# Predict trust for a single utterance
result = engine.predict_trust(
    utterance="I'll help you resolve this issue right away.",
    context={
        'emotion': 'neutral',
        'response_time': 1.5,
        'scenario': 'customer_service'
    },
    model_name='ensemble'  # or 'random_forest', 'xgboost', etc.
)

print(f"Trust Score: {result['trust_score']:.2f}/7.0")
print(f"Confidence: {result['confidence']:.2f}")
```

### **3. Batch Prediction**
```python
# Predict multiple conversations at once
conversations = [
    {
        'utterance': "I understand your concern...",
        'context': {'emotion': 'neutral', 'scenario': 'support'}
    },
    {
        'utterance': "I'm not sure about that...", 
        'context': {'emotion': 'uncertain', 'scenario': 'technical'}
    }
]

results = engine.predict_batch(conversations, model_name='ensemble')
for result in results:
    print(f"Trust: {result['trust_score']:.2f}, Confidence: {result['confidence']:.2f}")
```

## 📊 **Model Performance Summary**

Based on your dataset of 12,569 agent turns:

| Model | R² Score | RMSE | Accuracy (±0.5) |
|-------|----------|------|-----------------|
| **Ensemble** | **94.6%** | **0.401** | **82.3%** |
| Random Forest | 94.4% | 0.410 | 82.2% |
| XGBoost | 93.8% | 0.429 | 81.5% |
| LSTM | 91.8% | 0.494 | 74.8% |
| SVM | 92.0% | 0.489 | 73.1% |

## 🔧 **Integration Examples**

### **Web API Integration**
```python
from flask import Flask, request, jsonify
from trust_inference import TrustInferenceEngine

app = Flask(__name__)
engine = TrustInferenceEngine()

@app.route('/predict_trust', methods=['POST'])
def predict_trust():
    data = request.json
    result = engine.predict_trust(
        utterance=data['utterance'],
        context=data.get('context', {}),
        model_name=data.get('model', 'ensemble')
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### **Real-time Chat Integration**
```python
# Example: Monitor live chat and predict trust
def process_agent_message(message, conversation_context):
    result = engine.predict_trust(message, conversation_context)
    
    if result['trust_score'] < 3.0:
        alert_supervisor(f"Low trust detected: {result['trust_score']:.2f}")
    
    return result
```

### **Batch Processing**
```python
# Process conversation logs
import pandas as pd

def analyze_conversation_log(csv_file):
    df = pd.read_csv(csv_file)
    
    results = []
    for _, row in df.iterrows():
        if row['speaker'] == 'agent':
            result = engine.predict_trust(
                utterance=row['message'],
                context={
                    'scenario': row['scenario'],
                    'emotion': row.get('emotion', 'neutral'),
                    'response_time': row.get('response_time', 2.0)
                }
            )
            results.append(result)
    
    return results
```

## 📁 **File Structure After Training**
```
conv_trust/
├── pipeline_results/
│   ├── trained_models/
│   │   ├── feature_engineer.pkl      # Feature extraction pipeline
│   │   ├── random_forest_model.pkl   # Random Forest model
│   │   ├── xgboost_model.pkl         # XGBoost model  
│   │   ├── svm_model.pkl             # SVM model
│   │   ├── ensemble_model.pkl        # Optimized ensemble
│   │   ├── lstm_model.pth            # LSTM model (PyTorch)
│   │   └── model_results_summary.csv # Performance metrics
│   └── evaluation/
│       ├── model_comparison.csv      # Detailed comparison
│       ├── *_evaluation.png          # Performance plots
│       └── evaluation_report.txt     # Full report
├── trust_inference.py                # Standalone inference script
├── run_complete_pipeline.py          # Training pipeline
└── USAGE_GUIDE.md                    # This guide
```

## ⚡ **Performance Tips**

1. **Model Selection**: Use `ensemble` for best accuracy, `random_forest` for speed
2. **Batch Processing**: Process multiple predictions together for efficiency
3. **Caching**: Cache the inference engine initialization for repeated use
4. **Memory**: LSTM model uses more memory; use traditional ML for resource-constrained environments

## 🔍 **Troubleshooting**

**Error: "Models directory not found"**
- Solution: Run training pipeline first: `uv run run_complete_pipeline.py`

**Error: "Previously unseen labels"**
- Solution: The inference engine auto-validates and fixes unknown categories

**Low confidence scores**
- Check if input text is similar to training data
- Ensure context values are reasonable (e.g., response_time > 0)

## 🎯 **Next Steps**

1. **Deploy as Web Service**: Use Flask/FastAPI to create REST API
2. **Real-time Integration**: Connect to live chat systems  
3. **Monitoring Dashboard**: Build UI to visualize trust trends
4. **Model Updates**: Retrain periodically with new conversation data
5. **A/B Testing**: Compare different models in production

---

Your trust detection system is now ready for production use! 🚀
