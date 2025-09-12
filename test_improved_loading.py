#!/usr/bin/env python3
"""
Test script for the improved weight loading.
Run this in your MatchAnything environment to test the improved loader.
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "Convertion_Tensorrt_new"))
sys.path.insert(0, str(Path(__file__).parent / "Convertion_Tensorrt"))

def test_improved_loading():
    """Test the improved weight loading."""
    checkpoint_path = "/home/mathias/MatchAnything/imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt"
    
    try:
        # Import required modules
        from Convertion_Tensorrt_new.accurate_matchanything_trt_dyn import AccurateMatchAnythingTRT
        from Convertion_Tensorrt_new.improved_weight_loader import apply_improved_weight_loading
        
        print("=== TESTING IMPROVED WEIGHT LOADING ===")
        
        # Create model
        print("\n1. Creating model...")
        model = AccurateMatchAnythingTRT(amp=False)
        model_state_dict = model.state_dict()
        print(f"   Model has {len(model_state_dict)} parameters")
        
        # Test improved loading
        print(f"\n2. Testing improved weight loading...")
        print(f"   Checkpoint: {checkpoint_path}")
        
        loadable = apply_improved_weight_loading(
            checkpoint_path=checkpoint_path,
            model_state_dict=model_state_dict,
            load_dinov2_components=True,
            verbose=True
        )
        
        # Load the weights
        print(f"\n3. Loading weights into model...")
        missing, unexpected = model.load_state_dict(loadable, strict=False)
        
        loaded_pct = (len(loadable) / len(model_state_dict)) * 100
        print(f"\n=== FINAL RESULTS ===")
        print(f"Loaded: {len(loadable)}/{len(model_state_dict)} ({loaded_pct:.1f}%)")
        print(f"Missing: {len(missing)}")
        print(f"Unexpected: {len(unexpected)}")
        
        if loaded_pct >= 80:
            print("✅ SUCCESS: High loading percentage achieved!")
        elif loaded_pct >= 50:
            print("⚠️  PARTIAL: Moderate loading percentage")
        else:
            print("❌ FAILED: Low loading percentage")
        
        # Test forward pass
        print(f"\n4. Testing forward pass...")
        model.eval()
        with torch.no_grad():
            x0 = torch.rand(1, 3, 840, 840)
            x1 = torch.rand(1, 3, 840, 840)
            try:
                outputs = model(x0, x1)
                print("✅ Forward pass successful!")
                print("Output keys:", list(outputs.keys()))
                for key, tensor in outputs.items():
                    print(f"  {key}: {tuple(tensor.shape)}")
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
        
        return loaded_pct >= 80
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_improved_loading()
    sys.exit(0 if success else 1)