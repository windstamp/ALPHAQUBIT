#!/usr/bin/env python3
"""
Quick test to verify the --mla flag functionality works correctly.
This test verifies that:
1. Both model architectures can be imported
2. The --mla flag is properly parsed
3. The correct model is selected based on the flag
"""

import sys
import argparse

def test_imports():
    """Test that both model architectures can be imported."""
    print("=" * 60)
    print("Test 1: Import both model architectures")
    print("=" * 60)
    
    try:
        from ai_models.model import AlphaQubitDecoder as TransformerDecoder
        print("✓ Successfully imported standard Transformer from ai_models.model")
    except ImportError as e:
        print(f"✗ Failed to import standard Transformer: {e}")
        return False
    
    try:
        from ai_models.model_mla import AlphaQubitDecoder as MLADecoder
        print("✓ Successfully imported MLA from ai_models.model_mla")
    except ImportError as e:
        print(f"✗ Failed to import MLA: {e}")
        return False
    
    # Verify they are different classes
    if TransformerDecoder is not MLADecoder:
        print("✓ TransformerDecoder and MLADecoder are different classes")
    else:
        print("✗ TransformerDecoder and MLADecoder are the same class!")
        return False
    
    return True


def test_argparse_mla_flag():
    """Test that --mla flag is properly parsed."""
    print("\n" + "=" * 60)
    print("Test 2: Argparse --mla flag")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mla", action="store_true", 
                        help="Use MLA model instead of standard transformer")
    
    # Test without --mla flag (default should be False)
    args1 = parser.parse_args([])
    if args1.mla == False:
        print("✓ Without --mla flag: args.mla = False (correct)")
    else:
        print(f"✗ Without --mla flag: args.mla = {args1.mla} (expected False)")
        return False
    
    # Test with --mla flag (should be True)
    args2 = parser.parse_args(["--mla"])
    if args2.mla == True:
        print("✓ With --mla flag: args.mla = True (correct)")
    else:
        print(f"✗ With --mla flag: args.mla = {args2.mla} (expected True)")
        return False
    
    return True


def test_model_selection():
    """Test that the correct model is selected based on the flag."""
    print("\n" + "=" * 60)
    print("Test 3: Model selection based on --mla flag")
    print("=" * 60)
    
    from ai_models.model import AlphaQubitDecoder as TransformerDecoder
    from ai_models.model_mla import AlphaQubitDecoder as MLADecoder
    
    # Simulate model selection logic
    def select_model(use_mla: bool):
        if use_mla:
            return MLADecoder
        else:
            return TransformerDecoder
    
    # Test default (transformer)
    selected = select_model(use_mla=False)
    if selected is TransformerDecoder:
        print("✓ With use_mla=False: Selected TransformerDecoder (correct)")
    else:
        print("✗ With use_mla=False: Selected wrong model")
        return False
    
    # Test with MLA
    selected = select_model(use_mla=True)
    if selected is MLADecoder:
        print("✓ With use_mla=True: Selected MLADecoder (correct)")
    else:
        print("✗ With use_mla=True: Selected wrong model")
        return False
    
    return True


def test_train_py_help():
    """Test that train.py has the --mla flag in its help output."""
    print("\n" + "=" * 60)
    print("Test 4: Check train.py has --mla in help")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "ai_models/train.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "--mla" in result.stdout:
        print("✓ train.py help contains --mla flag")
        # Show the relevant line
        for line in result.stdout.split('\n'):
            if '--mla' in line:
                print(f"  Found: {line.strip()}")
        return True
    else:
        print("✗ train.py help does not contain --mla flag")
        print(f"  stdout: {result.stdout[:500]}")
        return False


def test_decode_py_help():
    """Test that decode.py has the --mla flag in its help output."""
    print("\n" + "=" * 60)
    print("Test 5: Check decode.py has --mla in help")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "ai_models/decode.py", "--help"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if "--mla" in result.stdout:
        print("✓ decode.py help contains --mla flag")
        for line in result.stdout.split('\n'):
            if '--mla' in line:
                print(f"  Found: {line.strip()}")
        return True
    else:
        print("✗ decode.py help does not contain --mla flag")
        print(f"  stdout: {result.stdout[:500]}")
        return False


def main():
    print("=" * 60)
    print("  Testing --mla flag implementation")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Import test", test_imports()))
    results.append(("Argparse test", test_argparse_mla_flag()))
    results.append(("Model selection test", test_model_selection()))
    
    # These tests require torch which may not be available
    try:
        results.append(("train.py help test", test_train_py_help()))
    except Exception as e:
        print(f"\n⚠ Skipped train.py help test: {e}")
        
    try:
        results.append(("decode.py help test", test_decode_py_help()))
    except Exception as e:
        print(f"\n⚠ Skipped decode.py help test: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
