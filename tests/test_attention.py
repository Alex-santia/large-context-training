"""Unit tests for attention module."""

import pytest
import torch


class TestRingAttention:
    def test_basic_forward(self):
        from src.attention.ring import RingAttention
        attn = RingAttention(head_dim=128, num_heads=8, max_seq_len=8192)
        q = torch.randn(1, 1024, 1024)
        k = torch.randn(1, 1024, 1024)
        v = torch.randn(1, 1024, 1024)
        out = attn(q, k, v)
        assert out.shape == (1, 1024, 1024)

    def test_gradient_flow(self):
        from src.attention.ring import RingAttention
        attn = RingAttention(head_dim=64, num_heads=4, max_seq_len=4096)
        q = torch.randn(1, 512, 256, requires_grad=True)
        k = torch.randn(1, 512, 256)
        v = torch.randn(1, 512, 256)
        out = attn(q, k, v)
        loss = out.sum()
        loss.backward()
        assert q.grad is not None

    def test_different_lengths(self):
        from src.attention.ring import RingAttention
        attn = RingAttention(head_dim=128, num_heads=8)
        for length in [128, 512, 2048]:
            q = torch.randn(1, length, 1024)
            k = torch.randn(1, length, 1024)
            v = torch.randn(1, length, 1024)
            out = attn(q, k, v)
            assert out.shape[1] == length