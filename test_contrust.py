#!/usr/bin/env python3
"""
ConTrust Model Testing Script
Verify that the novel architecture works correctly
"""

import torch
import torch.nn.functional as F
import numpy as np
from contrust_model import ConTrustModel, create_contrust_config
from contrust_data_processor import ConTrustDataProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_architecture():
    """Test ConTrust model architecture"""
    
    print("🧪 Testing ConTrust Model Architecture")
    print("=" * 50)
    
    # Create configuration
    config = create_contrust_config()
    config['d_model'] = 256  # Smaller for testing
    config['n_layers'] = 2
    
    # Initialize model
    model = ConTrustModel(config)
    model.eval()
    
    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test batch
    batch_size, seq_len, max_tokens = 2, 3, 32
    
    test_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len, max_tokens)),
        'attention_mask': torch.ones((batch_size, seq_len, max_tokens)),
        'temporal_features': torch.randn(batch_size, seq_len, 64),
        'behavioral_features': torch.randn(batch_size, 128),
        'trust_scores': torch.randn(batch_size, seq_len),
        'conversation_lengths': torch.tensor([3, 2])
    }
    
    print(f"✓ Test batch created")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(test_batch)
        
        print(f"✓ Forward pass successful!")
        print(f"  Trust prediction shape: {outputs['trust_score'].shape}")
        print(f"  Trust categories shape: {outputs['trust_categories'].shape}")
        print(f"  Emotion prediction shape: {outputs['emotion'].shape}")
        print(f"  Engagement prediction shape: {outputs['engagement'].shape}")
        
        # Test loss computation
        targets = {
            'trust_score': torch.randn(batch_size),
            'trust_categories': torch.randn(batch_size, 3),
            'emotion': torch.randint(0, 7, (batch_size,)),
            'engagement': torch.randn(batch_size),
            'trust_evolution': torch.randn(batch_size, seq_len)
        }
        
        losses = model.compute_loss(outputs, targets)
        print(f"✓ Loss computation successful!")
        print(f"  Total loss: {losses['total'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processor():
    """Test data processor"""
    
    print("\n🔄 Testing ConTrust Data Processor")
    print("=" * 50)
    
    config = create_contrust_config()
    processor = ConTrustDataProcessor(config)
    
    print(f"✓ Processor initialized")
    
    # Create mock conversation data
    mock_conversation = {
        'texts': [
            '[USER] I need help with my order',
            '[AGENT] I\'ll be happy to help you with that',
            '[USER] Thank you, that would be great',
            '[AGENT] I found your order and can process the refund'
        ],
        'speakers': ['user', 'agent', 'user', 'agent'],
        'emotions': ['neutral', 'positive', 'positive', 'neutral'],
        'trust_scores': [None, 5.0, None, 6.0],
        'trust_categories': [[0,0,0], [5,5,5], [0,0,0], [6,6,6]],
        'response_times': [0.0, 2.1, 0.0, 1.8],
        'turn_positions': [1, 2, 3, 4],
        'behavioral_features': [4, 0.5, 1.95, 2, 0.5, 2, 'customer_service', 'gpt-4'],
        'conversation_length': 4,
        'trust_evolution': [5.0, 6.0]
    }
    
    print(f"✓ Mock conversation created with {len(mock_conversation['texts'])} turns")
    
    # Mock fit encoders with sample data
    try:
        sample_data = {
            'conv1': mock_conversation,
            'conv2': {
                **mock_conversation,
                'emotions': ['negative', 'neutral', 'positive', 'positive'],
                'behavioral_features': [3, 0.3, 2.5, 3, 0.2, 1, 'tech_support', 'claude'],
            }
        }
        
        processor.fit_encoders(sample_data)
        print(f"✓ Encoders fitted successfully")
        
        # Test processing
        processed = processor.process_conversation(mock_conversation)
        
        print(f"✓ Conversation processed successfully")
        print(f"  Input IDs shape: {processed['input_ids'].shape}")
        print(f"  Temporal features shape: {processed['temporal_features'].shape}")
        print(f"  Behavioral features shape: {processed['behavioral_features'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trust_aware_attention():
    """Test trust-aware attention mechanism"""
    
    print("\n🎯 Testing Trust-Aware Attention")
    print("=" * 50)
    
    from contrust_model import TrustAwareSelfAttention
    
    d_model = 128
    seq_len = 5
    batch_size = 2
    
    attention_layer = TrustAwareSelfAttention(d_model, n_heads=4)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    trust_scores = torch.randn(batch_size, seq_len)  # Trust scores for each turn
    mask = torch.ones(batch_size, seq_len)
    
    try:
        output, attention_weights = attention_layer(x, trust_scores, mask)
        
        print(f"✓ Trust-aware attention successful!")
        print(f"  Output shape: {output.shape}")
        print(f"  Attention weights shape: {attention_weights.shape}")
        
        # Test without trust scores
        output_no_trust, _ = attention_layer(x, None, mask)
        print(f"✓ Standard attention (no trust) successful!")
        
        # Verify trust modulation
        diff = torch.mean(torch.abs(output - output_no_trust)).item()
        print(f"  Trust modulation effect: {diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trust-aware attention failed: {e}")
        return False

def test_conversation_memory():
    """Test conversation memory mechanism"""
    
    print("\n🧠 Testing Conversation Memory")
    print("=" * 50)
    
    from contrust_model import ConversationMemory
    
    input_dim = 128
    batch_size = 2
    max_len = 6
    
    memory_module = ConversationMemory(input_dim, hidden_dim=64, num_layers=1)
    
    # Test input with variable lengths
    x = torch.randn(batch_size, max_len, input_dim)
    conversation_lengths = torch.tensor([6, 4])  # Different conversation lengths
    
    try:
        memory_state, trust_evolution = memory_module(x, conversation_lengths)
        
        print(f"✓ Conversation memory successful!")
        print(f"  Memory state shape: {memory_state.shape}")
        print(f"  Trust evolution shape: {trust_evolution.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversation memory failed: {e}")
        return False

def test_multimodal_fusion():
    """Test multi-modal fusion encoder"""
    
    print("\n🔗 Testing Multi-Modal Fusion")
    print("=" * 50)
    
    from contrust_model import MultiModalFusionEncoder
    
    batch_size = 2
    seq_len = 4
    text_dim = 256
    
    fusion_encoder = MultiModalFusionEncoder(
        text_dim=text_dim,
        temporal_dim=64,
        behavioral_dim=128,
        fusion_dim=256
    )
    
    # Test inputs
    text_features = torch.randn(batch_size, seq_len, text_dim)
    temporal_features = torch.randn(batch_size, seq_len, 64)
    behavioral_features = torch.randn(batch_size, 128)
    
    try:
        fused_features, modality_weights = fusion_encoder(
            text_features, temporal_features, behavioral_features
        )
        
        print(f"✓ Multi-modal fusion successful!")
        print(f"  Fused features shape: {fused_features.shape}")
        print(f"  Modality weights shape: {modality_weights.shape}")
        print(f"  Average weights: {torch.mean(modality_weights, dim=0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-modal fusion failed: {e}")
        return False

def run_all_tests():
    """Run all ConTrust tests"""
    
    print("🚀 ConTrust Model Validation Suite")
    print("=" * 60)
    print("Testing novel conversational trust prediction architecture...")
    print("=" * 60)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Data Processor", test_data_processor),
        ("Trust-Aware Attention", test_trust_aware_attention),
        ("Conversation Memory", test_conversation_memory),
        ("Multi-Modal Fusion", test_multimodal_fusion)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! ConTrust is ready for training!")
        print("Run: uv run train_contrust.py --data_dir data --epochs 50")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
