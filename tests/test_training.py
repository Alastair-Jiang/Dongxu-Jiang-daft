"""Test training pipeline (Stage 1-3) — constructor and config validation."""

import pytest
import torch
import torch.nn as nn

from daft.training import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.models.experts import TrendExpert, ReversalExpert
from daft.data.panel import Panel


# ── Synthetic Panel helper ──────────────────────────────────────────────

def make_synthetic_panel(T=100, N=10):
    """Create a minimal synthetic Panel for trainer constructors."""
    values = torch.randn(T, N, 5)
    values[..., 3] = values[..., 3].abs() + 10.0  # positive close
    mask = torch.ones(T, N, dtype=torch.bool)
    return Panel(
        values=values,
        mask=mask,
        dates=[f"2024-01-{i+1:02d}" for i in range(T)],
        asset_ids=[f"stock_{j}" for j in range(N)],
        feature_names=["open", "high", "low", "close", "volume"],
    )


# ── Stage1ExpertTrainer ─────────────────────────────────────────────────

class TestStage1ExpertTrainer:
    @pytest.fixture
    def trainer(self):
        experts = nn.ModuleList([
            TrendExpert(input_dim=200, hidden_dim=64),
            ReversalExpert(input_dim=200, hidden_dim=64),
        ])
        panel = make_synthetic_panel()
        config = {"epochs": 10, "batch_size": 128, "lr": 0.001}
        return Stage1ExpertTrainer(experts, panel, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert len(trainer.experts) == 2
        assert trainer.config["epochs"] == 10
        assert trainer.device == torch.device("cpu")

    def test_train_all_exists(self, trainer):
        """Stage1ExpertTrainer exposes train_all()."""
        assert callable(trainer.train_all)


# ── RouterTrainer ───────────────────────────────────────────────────────

class TestRouterTrainer:
    @pytest.fixture
    def trainer(self, ensemble):
        config = {"epochs": 50, "batch_size": 2048, "lr": 0.0005,
                  "modulation_strength": 0.1}
        return RouterTrainer(ensemble, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert trainer.config["epochs"] == 50
        assert trainer.device == torch.device("cpu")
        assert hasattr(trainer, "layer_proj")

    def test_train_callable(self, trainer):
        """RouterTrainer exposes train(panel, val_panel)."""
        assert callable(trainer.train)


# ── JointTrainer ────────────────────────────────────────────────────────

class TestJointTrainer:
    @pytest.fixture
    def trainer(self, ensemble):
        config = {"epochs": 30, "batch_size": 2048, "lr": 1e-5,
                  "modulation_strength": 1.0}
        layer_proj = nn.ModuleDict({
            "l0": nn.Sequential(nn.Linear(200, 128), nn.SiLU(), nn.Linear(128, 64)),
            "l1": nn.Sequential(nn.Linear(200, 128), nn.SiLU(), nn.Linear(128, 64)),
            "l2": nn.Sequential(nn.Linear(200, 128), nn.SiLU(), nn.Linear(128, 64)),
        })
        return JointTrainer(ensemble, layer_proj, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert trainer.config["lr"] == 1e-5
        assert trainer.device == torch.device("cpu")

    def test_train_callable(self, trainer):
        """JointTrainer exposes train(panel, val_panel)."""
        assert callable(trainer.train)


# ── Training config consistency ─────────────────────────────────────────

class TestTrainingConfig:
    """Verify training config values are reasonable."""

    def test_stage1_config_defaults(self):
        experts = nn.ModuleList([TrendExpert()])
        panel = make_synthetic_panel()
        trainer = Stage1ExpertTrainer(experts, panel, {
            "epochs": 50, "batch_size": 2048,
            "lr": 0.001, "weight_decay": 1e-5,
        }, torch.device("cpu"))
        assert trainer.config["lr"] > 0
        assert trainer.config["epochs"] > 0
        assert trainer.config["batch_size"] > 0

    def test_stage2_config_defaults(self, ensemble):
        trainer = RouterTrainer(ensemble, {
            "epochs": 100, "batch_size": 2048,
            "lr": 0.0005, "modulation_strength": 0.1,
        }, torch.device("cpu"))
        assert 0 < trainer.config["modulation_strength"] <= 0.5

    def test_stage3_config_defaults(self, ensemble):
        layer_proj = nn.ModuleDict({
            "l0": nn.Linear(200, 64),
            "l1": nn.Linear(200, 64),
            "l2": nn.Linear(200, 64),
        })
        trainer = JointTrainer(ensemble, layer_proj, {
            "epochs": 50, "batch_size": 2048,
            "lr": 1e-5, "modulation_strength": 1.0,
        }, torch.device("cpu"))
        assert trainer.config["modulation_strength"] == 1.0
        assert trainer.config["lr"] < 0.001
