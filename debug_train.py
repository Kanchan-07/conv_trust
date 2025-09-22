#!/usr/bin/env python3
"""
Debug version of ConTrust training to identify bottlenecks
"""

import torch
import time
from contrust_model import ConTrustModel, create_contrust_config
from contrust_data_processor import create_contrust_dataloaders

def debug_training():
    """Debug ConTrust training step by step"""
    
    print("🔍 ConTrust Debug Training")
    print("=" * 50)
    
    # Setup
    config = create_contrust_config()
    config['batch_size'] = 2
    
    print("1. Creating model...")
    model = ConTrustModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("2. Loading data...")
    start_time = time.time()
    train_loader, val_loader, processor = create_contrust_dataloaders(
        'data', config, batch_size=2, val_split=0.2
    )
    print(f"✓ Data loaded in {time.time() - start_time:.1f}s")
    
    print("3. Getting first batch...")
    try:
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded with keys: {list(batch.keys())}")
        
        # Print batch shapes for debugging
        for key, value in batch.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
            else:
                print(f"  {key}: {type(value)}")
        
    except Exception as e:
        print(f"❌ Error loading batch: {e}")
        return
    
    print("4. Testing forward pass...")
    model.eval()
    try:
        start_time = time.time()
        with torch.no_grad():
            predictions = model(batch)
        forward_time = time.time() - start_time
        print(f"✓ Forward pass completed in {forward_time:.2f}s")
        
        # Print prediction shapes
        for key, value in predictions.items():
            if torch.is_tensor(value):
                print(f"  pred_{key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("5. Testing loss computation...")
    try:
        targets = {
            'trust_score': batch.get('target_trust_score'),
            'trust_categories': batch.get('target_trust_categories'),
            'emotion': batch.get('target_emotion'),
            'engagement': batch.get('target_engagement'),
            'trust_evolution': batch.get('target_evolution')
        }
        
        start_time = time.time()
        losses = model.compute_loss(predictions, targets)
        loss_time = time.time() - start_time
        print(f"✓ Loss computation completed in {loss_time:.2f}s")
        
        # Print losses
        for key, value in losses.items():
            if torch.is_tensor(value):
                print(f"  {key}_loss: {value.item():.4f}")
        
    except Exception as e:
        print(f"❌ Error in loss computation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("6. Testing backward pass...")
    try:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        start_time = time.time()
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        backward_time = time.time() - start_time
        print(f"✓ Backward pass completed in {backward_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 All components working! Training should proceed normally.")
    print("If training still hangs, the issue might be:")
    print("- Very slow computation per batch")
    print("- Memory issues with large sequences")
    print("- Infinite loops in attention mechanisms")

if __name__ == "__main__":
    debug_training()
