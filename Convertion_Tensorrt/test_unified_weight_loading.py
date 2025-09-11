#!/usr/bin/env python3
"""
Comprehensive test script for the unified weight loading system.

This script tests the new unified weight loader against the MatchAnything checkpoint
and validates that the DINOv2 weights are properly loaded.
"""

import os
import sys


def test_unified_loader_standalone():
    """Test the unified weight loader in isolation"""
    print("=" * 60)
    print("TESTING UNIFIED WEIGHT LOADER (STANDALONE)")
    print("=" * 60)

    try:
        from unified_weight_loader import apply_unified_weight_loading
        from accurate_matchanything_trt import AccurateMatchAnythingTRT

        # Create a model to get the expected state dict
        print("[TEST] Creating model to get expected state dict...")
        model = AccurateMatchAnythingTRT(
            model_name="matchanything_roma",
            img_resize=832,
            match_threshold=0.1,
            amp=False,
        )
        model_state_dict = model.state_dict()

        print(f"[TEST] Model expects {len(model_state_dict)} parameters")

        # Analyze model structure
        dino_keys = [
            k for k in model_state_dict.keys() if k.startswith("encoder.dino.")
        ]
        encoder_keys = [
            k
            for k in model_state_dict.keys()
            if k.startswith("encoder.") and not k.startswith("encoder.dino.")
        ]
        matcher_keys = [k for k in model_state_dict.keys() if k.startswith("matcher.")]

        print("[TEST] Model structure:")
        print(f"  - DINOv2 parameters: {len(dino_keys)}")
        print(f"  - CNN encoder parameters: {len(encoder_keys)}")
        print(f"  - Matcher parameters: {len(matcher_keys)}")
        print(
            f"  - Other parameters: {len(model_state_dict) - len(dino_keys) - len(encoder_keys) - len(matcher_keys)}"
        )

        # Try to download checkpoint if it doesn't exist
        checkpoint_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt",
            )
        )

        if not os.path.exists(checkpoint_path):
            print(f"[TEST] Checkpoint not found at {checkpoint_path}")
            print("[TEST] Attempting to download...")
            try:
                from download_weights import download_matchanything_weights

                checkpoint_path = download_matchanything_weights(
                    output_dir=os.path.dirname(checkpoint_path), force_download=False
                )
            except Exception as e:
                print(f"[TEST] Failed to download checkpoint: {e}")
                print("[TEST] Skipping checkpoint-based tests...")
                return False

        if os.path.exists(checkpoint_path):
            print(f"[TEST] Using checkpoint: {checkpoint_path}")

            # Test the unified loader
            print("\n[TEST] Testing unified weight loader...")
            loadable = apply_unified_weight_loading(
                checkpoint_path, model_state_dict, load_dinov2_components=True
            )

            # Analyze results
            loaded_dino = sum(
                1 for k in loadable.keys() if k.startswith("encoder.dino.")
            )
            loaded_encoder = sum(
                1
                for k in loadable.keys()
                if k.startswith("encoder.") and not k.startswith("encoder.dino.")
            )
            loaded_matcher = sum(1 for k in loadable.keys() if k.startswith("matcher."))

            print("\n[TEST] Loading results:")
            print(
                f"  - Total loaded: {len(loadable)} / {len(model_state_dict)} ({100.0 * len(loadable) / len(model_state_dict):.1f}%)"
            )
            print(
                f"  - DINOv2 loaded: {loaded_dino} / {len(dino_keys)} ({100.0 * loaded_dino / max(len(dino_keys), 1):.1f}%)"
            )
            print(
                f"  - CNN encoder loaded: {loaded_encoder} / {len(encoder_keys)} ({100.0 * loaded_encoder / max(len(encoder_keys), 1):.1f}%)"
            )
            print(
                f"  - Matcher loaded: {loaded_matcher} / {len(matcher_keys)} ({100.0 * loaded_matcher / max(len(matcher_keys), 1):.1f}%)"
            )

            # Check for critical components
            critical_components = [
                "encoder.dino.pos_embed",
                "encoder.dino.cls_token",
                "encoder.dino.patch_embed.proj.weight",
                "encoder.dino.blocks.0.0.norm1.weight",  # First transformer block
            ]

            print("\n[TEST] Critical component check:")
            for component in critical_components:
                if component in loadable:
                    shape = loadable[component].shape
                    print(f"  ✅ {component}: {shape}")
                else:
                    print(f"  ❌ {component}: MISSING")

            # Success criteria
            success = len(loadable) >= 0.8 * len(model_state_dict)
            dino_success = (
                loaded_dino >= 0.8 * len(dino_keys) if len(dino_keys) > 0 else True
            )

            print("\n[TEST] Success criteria:")
            print(f"  - Overall loading (>80%): {'✅' if success else '❌'}")
            print(f"  - DINOv2 loading (>80%): {'✅' if dino_success else '❌'}")

            return success and dino_success

        else:
            print("[TEST] No checkpoint available for testing")
            return False

    except Exception as e:
        print(f"[TEST] Error in standalone test: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests for the unified weight loading system"""
    print("🧪 TESTING UNIFIED WEIGHT LOADING SYSTEM")
    print("=" * 80)

    results = {}

    # Test 1: Standalone loader test
    print("\n" + "=" * 20 + " TEST 1 " + "=" * 20)
    results["standalone"] = test_unified_loader_standalone()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.upper():15} : {status}")

    all_passed = all(results.values())

    print(
        f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_passed:
        print("\n🎉 SUCCESS: The unified weight loading system is working correctly!")
        print("   - DINOv2 weights should be properly loaded")
        print("   - ONNX export should include all model weights")
        print(
            "   - TensorRT engine built from this ONNX should be much larger and more accurate"
        )
    else:
        print("\n⚠️  Issues detected with the unified weight loading system.")
        print("   Please check the error messages above for details.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
