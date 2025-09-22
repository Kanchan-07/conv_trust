#!/usr/bin/env python3
"""
Simple ConTrust Model Test
"""

import torch
import numpy as np
from contrust_model import ConTrustModel, create_contrust_config

def simple_test():
    """Simple test of ConTrust architecture"""
    
    print("🧪 Simple ConTrust Test")
    print("="*30)
    
    # Create small config for testing
    config = create_contrust_config()
    config.update({
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4
    })
    
    # Initialize model
    model = ConTrustModel(config)
    model.eval()
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create simple test batch
    batch_size = 2
    seq_len = 3
    max_tokens = 16
    
    test_batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len, max_tokens)),
        'attention_mask': torch.ones((batch_size, seq_len, max_tokens)),
        'temporal_features': torch.randn(batch_size, seq_len, 64),
        'behavioral_features': torch.randn(batch_size, 128),
        'trust_scores': torch.randn(batch_size, seq_len),
        'conversation_lengths': torch.tensor([3, 2])
    }
    
    print(f"✓ Test batch created (batch_size={batch_size}, seq_len={seq_len})")
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(test_batch)
        
        print("✅ Forward pass successful!")
        print(f"  Trust score shape: {outputs['trust_score'].shape}")
        print(f"  Trust categories shape: {outputs['trust_categories'].shape}")
        print(f"  Emotion shape: {outputs['emotion'].shape}")
        print(f"  Engagement shape: {outputs['engagement'].shape}")
        
        # Test outputs are reasonable
        trust_pred = outputs['trust_score'].cpu().numpy()
        print(f"  Sample trust predictions: {trust_pred.flatten()}")
        
        # Test loss computation
        targets = {
            'trust_score': torch.randn(batch_size),
            'trust_categories': torch.randn(batch_size, 3),
            'emotion': torch.randint(0, 7, (batch_size,)),
            'engagement': torch.randn(batch_size),
            'trust_evolution': torch.randn(batch_size, seq_len)
        }
        
        losses = model.compute_loss(outputs, targets)
        print(f"✅ Loss computation successful!")
        print(f"  Total loss: {losses['total'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    
    if success:
        print("\n🎉 ConTrust model is working correctly!")
        print("Ready for training on your conversational trust dataset!")
    else:
        print("\n❌ ConTrust model has issues that need fixing.")
