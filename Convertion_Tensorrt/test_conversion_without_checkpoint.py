#!/usr/bin/env python3
"""
Test TensorRT conversion without requiring a checkpoint.
This helps debug the conversion pipeline independently of weight loading issues.
"""
import torch
import os

def test_conversion():
    print("="*60)
    print("TESTING TENSORRT CONVERSION WITHOUT CHECKPOINT")
    print("="*60)
    
    # Test with dimensions that are multiples of 14
    test_sizes = [
        (420, 420),  # 14 × 30
        (560, 560),  # 14 × 40  
        (840, 840),  # 14 × 60
    ]
    
    for H, W in test_sizes:
        print(f"\nTesting with size {H}x{W} (divisible by 14: {H%14==0 and W%14==0})")
        
        try:
            # Import our TensorRT model
            from accurate_matchanything_trt import AccurateMatchAnythingTRT
            
            # Create model without checkpoint
            model = AccurateMatchAnythingTRT(
                model_name="matchanything_roma",
                img_resize=H,
                match_threshold=0.1,
                amp=False
            )
            model.eval()
            
            # Test forward pass
            x1 = torch.rand(1, 3, H, W)
            x2 = torch.rand(1, 3, H, W)
            
            with torch.no_grad():
                result = model(x1, x2)
                
            print(f"  ✅ Forward pass successful!")
            print(f"  📊 Results:")
            print(f"    - Keypoints0: {result['keypoints0'].shape}")
            print(f"    - Keypoints1: {result['keypoints1'].shape}")
            print(f"    - Confidences: {result['mconf'].shape}")
            print(f"    - Match count: {len(result['mconf'])}")
            
            if len(result['mconf']) > 0:
                print(f"    - Confidence range: {result['mconf'].min():.3f} - {result['mconf'].max():.3f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

def test_onnx_export():
    print(f"\n{'='*60}")
    print("TESTING ONNX EXPORT WITHOUT CHECKPOINT")
    print("="*60)
    
    try:
        from accurate_matchanything_trt import export_accurate_matchanything_onnx
        
        # Test ONNX export without checkpoint
        onnx_path = "out/test_without_checkpoint.onnx"
        os.makedirs("out", exist_ok=True)
        
        export_accurate_matchanything_onnx(
            onnx_path=onnx_path,
            model_name="matchanything_roma",
            H=840,
            W=840,
            match_threshold=0.1,
            ckpt=None  # No checkpoint
        )
        
        print("✅ ONNX export successful without checkpoint!")
        print(f"📁 Saved to: {onnx_path}")
        
        # Check file size
        if os.path.exists(onnx_path):
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"📏 File size: {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🧪 Testing TensorRT conversion pipeline...")
    
    # Test 1: Basic forward pass
    test_conversion()
    
    # Test 2: ONNX export
    test_onnx_export()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print("If the tests above passed, the conversion pipeline works correctly.")
    print("The remaining issue is likely checkpoint format compatibility.")
    print("\nNext steps:")
    print("1. ✅ Conversion pipeline works")
    print("2. 🔧 Need to fix checkpoint loading for your specific checkpoint format")
    print("3. 🚀 Then you can build the full TensorRT engine")

if __name__ == "__main__":
    main()