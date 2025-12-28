# SPDX-License-Identifier: Apache-2.0
"""
Unit test for LSQ syndrome encode/decode logic.

Verifies that modulo arithmetic handles negative numbers correctly.
"""

import math

# LSQ Quantization Constants (from lsq_math.cuh)
ANCHOR_RANGE_MIN = -40.0
ANCHOR_RANGE_MAX = 40.0
ANCHOR_LEVELS = 63  # 6 bits -> 0..62
ANCHOR_STEP = (ANCHOR_RANGE_MAX - ANCHOR_RANGE_MIN) / (ANCHOR_LEVELS - 1)
# ANCHOR_STEP = 80.0 / 62 = 1.2903...

LATTICE_INTERVAL = 2.0 * ANCHOR_STEP  # ~2.58
SYNDROME_LEVELS = 31  # 5 bits -> 0..30
SYNDROME_STEP = LATTICE_INTERVAL / SYNDROME_LEVELS

print(f"ANCHOR_STEP = {ANCHOR_STEP:.6f}")
print(f"LATTICE_INTERVAL = {LATTICE_INTERVAL:.6f}")
print(f"SYNDROME_STEP = {SYNDROME_STEP:.6f}")


def syndrome_encode(actual: float, predictor: float) -> int:
    """
    Python implementation of syndrome_encode from lsq_math.cuh.

    Encode syndrome using predictor value.
    Computes residual = actual - predictor, then wraps to [-L/2, L/2]
    and quantizes to 5 bits (0..30)
    """
    residual = actual - predictor

    half_interval = LATTICE_INTERVAL / 2.0

    # Wrap to [-LATTICE_INTERVAL/2, LATTICE_INTERVAL/2]
    # Use fmod with offset to handle negative values correctly
    # CUDA code: fmodf(residual + half_interval + LATTICE_INTERVAL * 1000.0f, LATTICE_INTERVAL)
    wrapped = math.fmod(residual + half_interval + LATTICE_INTERVAL * 1000.0, LATTICE_INTERVAL)
    wrapped -= half_interval

    # Quantize wrapped residual to [0, 30]
    # Map [-L/2, L/2] to [0, 30]
    q = (wrapped + half_interval) / SYNDROME_STEP
    iq = round(q)
    return max(0, min(SYNDROME_LEVELS - 1, iq))


def syndrome_decode(syn: int, predictor: float) -> float:
    """
    Python implementation of syndrome_decode from lsq_math.cuh.

    Decode syndrome using predictor to unwrap.
    """
    half_interval = LATTICE_INTERVAL / 2.0

    # Dequantize syndrome to residual in [-L/2, L/2]
    residual = (float(syn) * SYNDROME_STEP) - half_interval

    # Unwrap: actual = predictor + residual
    actual = predictor + residual

    return actual


def test_syndrome_roundtrip(actual: float, predictor: float, expected: float = None):
    """Test syndrome encode/decode roundtrip."""
    if expected is None:
        expected = actual

    syn = syndrome_encode(actual, predictor)
    decoded = syndrome_decode(syn, predictor)

    error = abs(decoded - expected)

    print(f"\nTest: actual={actual:.4f}, predictor={predictor:.4f}")
    print(f"  encoded syndrome: {syn}")
    print(f"  decoded: {decoded:.4f}")
    print(f"  expected: {expected:.4f}")
    print(f"  error: {error:.6f}")

    # Check that we're within quantization tolerance
    max_error = SYNDROME_STEP / 2 + 0.01  # Half step + small epsilon

    if error > max_error:
        print(f"  ❌ FAIL: Error {error:.6f} > max allowed {max_error:.6f}")
        return False
    else:
        print(f"  ✅ PASS")
        return True


def test_user_cases():
    """Test the specific cases the user requested."""
    print("\n" + "=" * 60)
    print("USER TEST CASES")
    print("=" * 60)

    results = []

    # Test Case 1: Predictor 320.0, Actual 319.5
    # Expected: should decode to 319.5, not -320.5
    # The residual is 319.5 - 320.0 = -0.5
    # This is within [-L/2, L/2] where L/2 ≈ 1.29, so no wrapping needed
    print("\n--- Test Case 1: Predictor=320.0, Actual=319.5 ---")
    results.append(test_syndrome_roundtrip(319.5, 320.0))

    # Test Case 2: Predictor 0.0, Actual 1.2
    # The residual is 1.2 - 0.0 = 1.2
    # This is within [-L/2, L/2] where L/2 ≈ 1.29, so no wrapping needed
    print("\n--- Test Case 2: Predictor=0.0, Actual=1.2 ---")
    results.append(test_syndrome_roundtrip(1.2, 0.0))

    return all(results)


def test_edge_cases():
    """
    Test edge cases for the modulo arithmetic.

    IMPORTANT: LSQ syndrome encoding only works when the residual (actual - predictor)
    is within [-L/2, L/2] where L = LATTICE_INTERVAL ≈ 2.58.

    If the residual exceeds this range, wrapping occurs and the decoded value
    will differ from the original. This is BY DESIGN - the Hadamard rotation
    applied before LSQ encoding ensures adjacent values are correlated enough
    that residuals stay within bounds.
    """
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)

    results = []
    half_L = LATTICE_INTERVAL / 2.0

    print(f"\nNote: Syndrome can only encode residuals in [-{half_L:.4f}, {half_L:.4f}]")
    print("Values outside this range will wrap (expected behavior).\n")

    # Tests that SHOULD PASS: residuals within [-L/2, L/2]
    print("--- Tests that should pass (residuals within ±L/2) ---\n")

    print("--- Test: Residual = 0 (exact match) ---")
    results.append(test_syndrome_roundtrip(0.0, 0.0))

    print("\n--- Test: Small positive residual (+0.5) ---")
    results.append(test_syndrome_roundtrip(0.5, 0.0))

    print("\n--- Test: Small negative residual (-0.5) ---")
    results.append(test_syndrome_roundtrip(-0.5, 0.0))

    print("\n--- Test: Near positive boundary (+1.0 < L/2) ---")
    results.append(test_syndrome_roundtrip(1.0, 0.0))

    print("\n--- Test: Near negative boundary (-1.0 > -L/2) ---")
    results.append(test_syndrome_roundtrip(-1.0, 0.0))

    print("\n--- Test: Negative predictor, small positive residual ---")
    results.append(test_syndrome_roundtrip(-0.5, -1.0))

    print("\n--- Test: Both negative, residual within bounds ---")
    results.append(test_syndrome_roundtrip(-3.5, -3.0))

    print("\n--- Test: Large predictor, small residual ---")
    results.append(test_syndrome_roundtrip(319.5, 320.0))

    print("\n--- Test: Large negative predictor, small residual ---")
    results.append(test_syndrome_roundtrip(-319.5, -320.0))

    # Show that out-of-range residuals wrap (not a bug, expected behavior)
    print("\n" + "-" * 40)
    print("Expected wrapping for out-of-range residuals (not bugs):")
    print("-" * 40)

    print("\n--- Out-of-range: Large positive residual (+5.0) ---")
    syn = syndrome_encode(5.0, 0.0)
    decoded = syndrome_decode(syn, 0.0)
    print(f"  actual=5.0, predictor=0.0 → residual=5.0 >> L/2={half_L:.4f}")
    print(f"  Wrapping expected. Decoded: {decoded:.4f}")
    print(f"  This is correct behavior - value is outside LSQ range.")

    print("\n--- Out-of-range: Large negative residual (-5.0) ---")
    syn = syndrome_encode(-5.0, 0.0)
    decoded = syndrome_decode(syn, 0.0)
    print(f"  actual=-5.0, predictor=0.0 → residual=-5.0 << -L/2={-half_L:.4f}")
    print(f"  Wrapping expected. Decoded: {decoded:.4f}")
    print(f"  This is correct behavior - value is outside LSQ range.")

    return all(results)


def test_wraparound_issue():
    """
    Specifically test the issue the user mentioned:
    With predictor 320.0 and actual 319.5, we should NOT get -320.5.

    The concern is that modulo arithmetic with negative numbers
    could potentially flip signs incorrectly.
    """
    print("\n" + "=" * 60)
    print("WRAPAROUND SIGN ISSUE TEST")
    print("=" * 60)

    predictor = 320.0
    actual = 319.5

    residual = actual - predictor  # = -0.5
    half_interval = LATTICE_INTERVAL / 2.0

    print(f"predictor = {predictor}")
    print(f"actual = {actual}")
    print(f"residual = actual - predictor = {residual}")
    print(f"LATTICE_INTERVAL = {LATTICE_INTERVAL:.6f}")
    print(f"half_interval (L/2) = {half_interval:.6f}")

    # Check if residual is within [-L/2, L/2]
    if -half_interval <= residual <= half_interval:
        print(f"residual {residual:.4f} is WITHIN [-L/2, L/2] = [{-half_interval:.4f}, {half_interval:.4f}]")
        print("No wrapping should occur.")
    else:
        print(f"residual {residual:.4f} is OUTSIDE [-L/2, L/2]")
        print("Wrapping will occur.")

    # Manual step through the encode
    offset_val = residual + half_interval + LATTICE_INTERVAL * 1000.0
    print(f"\nStep 1: offset_val = residual + L/2 + L*1000 = {offset_val:.6f}")

    wrapped_raw = math.fmod(offset_val, LATTICE_INTERVAL)
    print(f"Step 2: fmod(offset_val, L) = {wrapped_raw:.6f}")

    wrapped = wrapped_raw - half_interval
    print(f"Step 3: wrapped = fmod - L/2 = {wrapped:.6f}")

    # This should be approximately equal to residual since no wrapping needed
    print(f"Step 4: wrapped ≈ residual? {wrapped:.6f} ≈ {residual:.6f} ? diff={abs(wrapped-residual):.9f}")

    # Quantize
    q = (wrapped + half_interval) / SYNDROME_STEP
    syn = round(q)
    syn = max(0, min(30, syn))
    print(f"Step 5: quantize = round({q:.6f}) = {syn}")

    # Decode
    decoded_residual = (float(syn) * SYNDROME_STEP) - half_interval
    decoded = predictor + decoded_residual
    print(f"\nDecoding:")
    print(f"  decoded_residual = syn * STEP - L/2 = {decoded_residual:.6f}")
    print(f"  decoded = predictor + decoded_residual = {predictor} + {decoded_residual:.6f} = {decoded:.6f}")

    if abs(decoded - actual) < 0.5:
        print(f"\n✅ SUCCESS: decoded {decoded:.4f} ≈ actual {actual:.4f}")
        return True
    else:
        print(f"\n❌ FAILURE: decoded {decoded:.4f} ≠ actual {actual:.4f}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LSQ Syndrome Encode/Decode Unit Test")
    print("=" * 60)

    all_passed = True

    all_passed &= test_wraparound_issue()
    all_passed &= test_user_cases()
    all_passed &= test_edge_cases()

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 60)
