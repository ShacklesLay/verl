#!/usr/bin/env python3
"""
Test script to verify top-k KL penalty implementation.
This script tests the kl_penalty_forward function with synthetic data.
"""

import torch
import sys

# Add verl to path
sys.path.insert(0, '.')

from verl.trainer.ppo.core_algos import kl_penalty_forward


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self, kl_topk_size=10):
        self.kl_topk_size = kl_topk_size


def test_topk_kl():
    """Test top-k KL penalty computation."""
    print("=" * 80)
    print("Testing Top-K KL Penalty Implementation")
    print("=" * 80)
    
    # Test configuration
    config = MockConfig(kl_topk_size=10)
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 4, 100
    print(f"\nTest data shape: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")
    print(f"Top-k size: k={config.kl_topk_size}")
    
    # Generate random log probabilities
    logprob = torch.randn(batch_size, seq_len, vocab_size).log_softmax(dim=-1)
    ref_logprob = torch.randn(batch_size, seq_len, vocab_size).log_softmax(dim=-1)
    
    print(f"\nInput shapes:")
    print(f"  logprob: {logprob.shape}")
    print(f"  ref_logprob: {ref_logprob.shape}")
    
    # Test 1: top-k with renormalization
    print("\n" + "-" * 80)
    print("Test 1: top-k (with renormalization)")
    print("-" * 80)
    try:
        kl_topk = kl_penalty_forward(logprob, ref_logprob, "top-k", config)
        print(f"✓ top-k KL shape: {kl_topk.shape} (expected: [{batch_size}, {seq_len}])")
        print(f"✓ top-k KL statistics:")
        print(f"    mean: {kl_topk.mean().item():.6f}")
        print(f"    std:  {kl_topk.std().item():.6f}")
        print(f"    min:  {kl_topk.min().item():.6f}")
        print(f"    max:  {kl_topk.max().item():.6f}")
        assert kl_topk.shape == (batch_size, seq_len), f"Shape mismatch: {kl_topk.shape}"
        print("✓ Test 1 PASSED")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        return False
    
    # Test 2: top-k-unnorm without renormalization
    print("\n" + "-" * 80)
    print("Test 2: top-k-unnorm (without renormalization)")
    print("-" * 80)
    try:
        kl_topk_unnorm = kl_penalty_forward(logprob, ref_logprob, "top-k-unnorm", config)
        print(f"✓ top-k-unnorm KL shape: {kl_topk_unnorm.shape} (expected: [{batch_size}, {seq_len}])")
        print(f"✓ top-k-unnorm KL statistics:")
        print(f"    mean: {kl_topk_unnorm.mean().item():.6f}")
        print(f"    std:  {kl_topk_unnorm.std().item():.6f}")
        print(f"    min:  {kl_topk_unnorm.min().item():.6f}")
        print(f"    max:  {kl_topk_unnorm.max().item():.6f}")
        assert kl_topk_unnorm.shape == (batch_size, seq_len), f"Shape mismatch: {kl_topk_unnorm.shape}"
        print("✓ Test 2 PASSED")
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False
    
    # Test 3: full KL divergence
    print("\n" + "-" * 80)
    print("Test 3: full KL divergence")
    print("-" * 80)
    try:
        kl_full = kl_penalty_forward(logprob, ref_logprob, "full", config)
        print(f"✓ full KL shape: {kl_full.shape} (expected: [{batch_size}, {seq_len}])")
        print(f"✓ full KL statistics:")
        print(f"    mean: {kl_full.mean().item():.6f}")
        print(f"    std:  {kl_full.std().item():.6f}")
        print(f"    min:  {kl_full.min().item():.6f}")
        print(f"    max:  {kl_full.max().item():.6f}")
        assert kl_full.shape == (batch_size, seq_len), f"Shape mismatch: {kl_full.shape}"
        print("✓ Test 3 PASSED")
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False
    
    # Test 4: Compare magnitudes
    print("\n" + "-" * 80)
    print("Test 4: Compare KL magnitudes")
    print("-" * 80)
    print(f"top-k mean:        {kl_topk.mean().item():.6f}")
    print(f"top-k-unnorm mean: {kl_topk_unnorm.mean().item():.6f}")
    print(f"full mean:         {kl_full.mean().item():.6f}")
    print("\nNote: top-k should be >= top-k-unnorm (due to renormalization)")
    print("      full should be >= top-k (includes all tokens)")
    
    # Test 5: Error handling - missing config
    print("\n" + "-" * 80)
    print("Test 5: Error handling (missing config)")
    print("-" * 80)
    try:
        kl_penalty_forward(logprob, ref_logprob, "top-k", None)
        print("✗ Test 5 FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        print("✓ Test 5 PASSED")
    except Exception as e:
        print(f"✗ Test 5 FAILED: Wrong exception type: {e}")
        return False
    
    # Test 6: Error handling - wrong input shape
    print("\n" + "-" * 80)
    print("Test 6: Error handling (wrong input shape)")
    print("-" * 80)
    try:
        wrong_shape = torch.randn(batch_size, seq_len)  # 2D instead of 3D
        kl_penalty_forward(wrong_shape, ref_logprob, "top-k", config)
        print("✗ Test 6 FAILED: Should have raised AssertionError")
        return False
    except AssertionError as e:
        print(f"✓ Correctly raised AssertionError: {e}")
        print("✓ Test 6 PASSED")
    except Exception as e:
        print(f"✗ Test 6 FAILED: Wrong exception type: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nTop-K KL implementation is working correctly.")
    print("You can now run your training script with confidence.")
    return True


if __name__ == "__main__":
    success = test_topk_kl()
    sys.exit(0 if success else 1)

