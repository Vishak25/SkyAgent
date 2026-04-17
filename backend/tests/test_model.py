"""
Tests for the ST-GNN model: architecture, forward pass, weight loading.

These tests verify the model works correctly without requiring full inference.
"""
import os
import sys
os.environ["USE_FIXTURES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from src.models.delay_gnn.model import STGNN


class TestSTGNNModel:
    """Tests for the ST-GNN model architecture."""

    def test_model_instantiation(self):
        """Model can be created with default params."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        assert model is not None

    def test_forward_pass_shape(self):
        """Forward pass produces correct output shape [N, 1]."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        model.eval()

        # Create a small test graph: 4 nodes, 5 features each
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index)

        assert out.shape == (4, 1), f"Expected shape (4, 1), got {out.shape}"

    def test_forward_pass_different_sizes(self):
        """Model handles graphs of different sizes."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        model.eval()

        for n_nodes in [2, 5, 10, 20]:
            x = torch.randn(n_nodes, 5)
            # Create a chain graph
            src = list(range(n_nodes - 1))
            dst = list(range(1, n_nodes))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

            with torch.no_grad():
                out = model(x, edge_index)

            assert out.shape == (n_nodes, 1), f"Failed for {n_nodes} nodes"

    def test_output_is_numeric(self):
        """Model output contains no NaN or Inf values."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        model.eval()

        x = torch.randn(5, 5)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
                                    [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            out = model(x, edge_index)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_model_parameter_count(self):
        """Model has a reasonable number of parameters."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        total_params = sum(p.numel() for p in model.parameters())
        # Should have parameters from 2 GCN layers + MLP
        assert total_params > 100, f"Too few params: {total_params}"
        assert total_params < 100000, f"Too many params: {total_params}"

    def test_weight_loading(self):
        """Model can load weights from disk (if they exist)."""
        weights_path = os.environ.get("MODEL_WEIGHTS_PATH", "")
        if not weights_path:
            # Check common locations
            for path in ["stgnn_best.pt", "training/stgnn_best.pt", "../stgnn_best.pt"]:
                full = os.path.join(os.path.dirname(__file__), "..", path)
                if os.path.isfile(full):
                    weights_path = full
                    break

        if not weights_path or not os.path.isfile(weights_path):
            pytest.skip("No model weights file found — skipping weight loading test")

        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        # Verify forward pass works with loaded weights
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        with torch.no_grad():
            out = model(x, edge_index)
        assert out.shape == (4, 1)
        assert not torch.isnan(out).any()

    def test_model_eval_mode(self):
        """Model can switch between train and eval mode."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)

        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

    def test_gradient_flow(self):
        """Gradients flow through the model during training."""
        model = STGNN(in_channels=5, hidden_channels=64, out_channels=1)
        model.train()

        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)

        out = model(x, edge_index)
        loss = out.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients flowing through model"
