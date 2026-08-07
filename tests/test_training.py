"""Test training pipeline (Stage 1-3)."""

import pytest
import torch
import torch.nn as nn

from daft.training.expert_trainer import ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.models.experts import TrendExpert, ReversalExpert


# ── ExpertTrainer (Stage 1) ───────────────────────────────────────────

class TestExpertTrainer:
    @pytest.fixture
    def trainer(self):
        experts = [
            TrendExpert(input_dim=200, hidden_dim=64),
            ReversalExpert(input_dim=200, hidden_dim=64),
        ]
        config = {"epochs": 10, "batch_size": 128, "lr": 0.001}
        return ExpertTrainer(experts, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert len(trainer.experts) == 2
        assert trainer.config["epochs"] == 10
        assert trainer.device == torch.device("cpu")

    def test_train_not_implemented(self, trainer):
        with pytest.raises(NotImplementedError):
            trainer.train(None, None)


# ── RouterTrainer (Stage 2) ───────────────────────────────────────────

class TestRouterTrainer:
    @pytest.fixture
    def trainer(self):
        config = {"epochs": 50, "batch_size": 2048, "lr": 0.0005}
        return RouterTrainer(None, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert trainer.config["epochs"] == 50
        assert trainer.device == torch.device("cpu")

    def test_train_not_implemented(self, trainer):
        with pytest.raises(NotImplementedError):
            trainer.train(None, None)


# ── JointTrainer (Stage 3) ────────────────────────────────────────────

class TestJointTrainer:
    @pytest.fixture
    def trainer(self):
        config = {"epochs": 30, "batch_size": 2048, "lr": 1e-5}
        return JointTrainer(None, config, torch.device("cpu"))

    def test_init(self, trainer):
        assert trainer.config["lr"] == 1e-5
        assert trainer.device == torch.device("cpu")

    def test_train_not_implemented(self, trainer):
        with pytest.raises(NotImplementedError):
            trainer.train(None, None)


# ── Training config consistency ───────────────────────────────────────

class TestTrainingConfig:
    """Verify training config values are reasonable."""
    def test_stage1_config_defaults(self):
        experts = [TrendExpert()]
        trainer = ExpertTrainer(experts, {
            "epochs": 50, "batch_size": 2048,
            "lr": 0.001, "weight_decay": 1e-5,
        }, torch.device("cpu"))
        assert trainer.config["lr"] > 0
        assert trainer.config["epochs"] > 0
        assert trainer.config["batch_size"] > 0

    def test_stage2_config_defaults(self):
        trainer = RouterTrainer(None, {
            "epochs": 100, "batch_size": 2048,
            "lr": 0.0005, "modulation_strength": 0.1,
        }, torch.device("cpu"))
        assert 0 < trainer.config["modulation_strength"] <= 0.5

    def test_stage3_config_defaults(self):
        trainer = JointTrainer(None, {
            "epochs": 50, "batch_size": 2048,
            "lr": 1e-5, "modulation_strength": 1.0,
        }, torch.device("cpu"))
        assert trainer.config["modulation_strength"] == 1.0
        assert trainer.config["lr"] < 0.001
