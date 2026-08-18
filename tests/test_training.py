"""Test training pipeline (Stage 1-3) — constructor and config validation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from daft.training import Stage1ExpertTrainer
from daft.training.router_trainer import RouterTrainer
from daft.training.joint_trainer import JointTrainer
from daft.features.regime_features import RegimeFeatureExtractor
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


# ── A2: 标准化统计量 train-only 一致性 (2026-08-18) ─────────────────────────

class TestNormStatsConsistency:
    """K3 纲领 A2: Stage2/3 的 val 段必须复用训练段标准化统计量。

    修复前: `_build_dataset` 对 train/val 各自拟合 mean/std，早停与选型
    依据的 val-IC 分布 ≠ 推理时(train-only 统计)的分布。
    修复后: 训练段拟合并记录 `self.norm_stats`，val 段强制注入复用。
    """

    @staticmethod
    def _make_layer_proj():
        return nn.ModuleDict({
            "l0": nn.Linear(200, 64),
            "l1": nn.Linear(200, 64),
            "l2": nn.Linear(200, 64),
        })

    def test_router_trainer_records_stats_on_train_build(self, ensemble):
        trainer = RouterTrainer(ensemble, {"epochs": 1}, torch.device("cpu"))
        assert trainer.norm_stats is None
        trainer._build_dataset(make_synthetic_panel(T=60, N=6))
        assert trainer.norm_stats is not None
        mean, std = trainer.norm_stats
        assert mean.shape == (1, 200)
        assert std.shape == (1, 200)
        assert (std >= 1e-4).all()

    def test_router_trainer_val_does_not_refit(self, ensemble):
        trainer = RouterTrainer(ensemble, {"epochs": 1}, torch.device("cpu"))
        trainer._build_dataset(make_synthetic_panel(T=60, N=6))
        stats_train = trainer.norm_stats
        # val 段注入复用 → 不得覆盖已记录的训练段统计量
        trainer._build_dataset(make_synthetic_panel(T=40, N=6), norm_stats=stats_train)
        assert trainer.norm_stats is stats_train

    def test_injected_stats_are_actually_applied(self, ensemble):
        """精确复算: 注入统计量后输出 == 手工用该统计量归一化的特征。"""
        trainer = RouterTrainer(ensemble, {"epochs": 1}, torch.device("cpu"))
        trainer._build_dataset(make_synthetic_panel(T=60, N=6))
        mean, std = trainer.norm_stats

        panel_b = make_synthetic_panel(T=40, N=6)
        s_b, _, _, _ = trainer._build_dataset(panel_b, norm_stats=(mean, std))

        with torch.no_grad():
            raw = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)(panel_b)
        raw = raw[:-1]  # 与 _build_dataset 一致: s_t[:-1] 对齐 targets
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        expect = ((raw - mean) / std).clamp(-10.0, 10.0).reshape(-1, 200)
        mask_b = panel_b.mask[:-1].reshape(-1)
        assert torch.allclose(s_b, expect[mask_b], atol=1e-5)

    def test_joint_trainer_records_and_reuses(self, ensemble):
        trainer = JointTrainer(ensemble, self._make_layer_proj(),
                               {"epochs": 1}, torch.device("cpu"))
        assert trainer.norm_stats is None
        trainer._build_dataset(make_synthetic_panel(T=60, N=6))
        assert trainer.norm_stats is not None
        before = trainer.norm_stats
        trainer._build_dataset(make_synthetic_panel(T=40, N=6), norm_stats=before)
        assert trainer.norm_stats is before


# ── A3: KDA 记忆行语义 —— 训练/推理一致性 (2026-08-18) ─────────────────────


def make_masked_panel(T=24, N=6, seed=123, valid_prob=0.7):
    """合成面板 + 随机停牌 mask(保证无全线停牌日)。"""
    values = torch.randn(T, N, 5)
    values[..., 3] = values[..., 3].abs() + 10.0  # positive close
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(T, N, generator=g) < valid_prob
    mask[:, 0] = True   # 每日至少 1 只可交易(无全线停牌日)
    return Panel(
        values=values,
        mask=mask,
        dates=[f"2024-01-{i+1:02d}" for i in range(T)],
        asset_ids=[f"stock_{j}" for j in range(N)],
        feature_names=["open", "high", "low", "close", "volume"],
    )


class TestA3MemoryRowSemantics:
    """K3 纲领 A3: KDA 记忆的行语义在训练与推理间必须一致。

    修复前: 训练展平流按 mask 删行 → 记忆第 b 行对应的 (t, 资产) 每日漂移;
    推理时记忆行=固定资产列。"记忆无增益"结论的潜在混淆因子。
    修复后(方向① 按资产对齐记忆行): 不删行, 每批恰一个时间步(B=N),
    mask=0 行在模型内跳过更新(memory.py)—— 训练/推理同口径。
    """

    @staticmethod
    def _make_layer_proj():
        return nn.ModuleDict({
            "l0": nn.Linear(200, 64),
            "l1": nn.Linear(200, 64),
            "l2": nn.Linear(200, 64),
        })

    # ---- _build_dataset 不删行 ----

    def test_router_trainer_build_keeps_full_grid(self, ensemble):
        T, N = 24, 6
        panel = make_masked_panel(T=T, N=N)
        assert (~panel.mask).any(), "测试前提: 面板必须含停牌行"
        trainer = RouterTrainer(ensemble, {"epochs": 1}, torch.device("cpu"))
        s_2d, t_1d, m_1d, t_idx = trainer._build_dataset(panel)
        assert s_2d.shape == ((T - 1) * N, 200)
        assert m_1d.reshape(-1).tolist() == panel.mask[:-1].reshape(-1).tolist()
        expect_tidx = torch.arange(T - 1).repeat_interleave(N)
        assert torch.equal(t_idx, expect_tidx), "行序必须时间主序(每 N 行一天)"

    def test_joint_trainer_build_keeps_full_grid(self, ensemble):
        T, N = 24, 6
        panel = make_masked_panel(T=T, N=N)
        trainer = JointTrainer(ensemble, self._make_layer_proj(),
                                {"epochs": 1}, torch.device("cpu"))
        s_2d, t_1d, m_1d, t_idx = trainer._build_dataset(panel)
        assert s_2d.shape == ((T - 1) * N, 200)
        assert m_1d.reshape(-1).tolist() == panel.mask[:-1].reshape(-1).tolist()

    # ---- 每批恰一个时间步 → 记忆批量维 == N ----

    def test_router_trainer_val_epoch_day_aligned(self, ensemble):
        T, N = 24, 6
        panel = make_masked_panel(T=T, N=N)
        trainer = RouterTrainer(ensemble, {"epochs": 1}, torch.device("cpu"))
        s, t, m, t_idx = trainer._build_dataset(panel)
        loader = DataLoader(TensorDataset(s, t, m, t_idx),
                             batch_size=panel.N, shuffle=False)  # 与 train() 同口径
        ensemble.eval()
        trainer.layer_proj.eval()
        _, sigs, tgts, ti, msk = trainer._run_epoch(
            loader, None, False, return_predictions=True,
        )
        assert sigs.shape[0] == (T - 1) * N
        assert torch.equal(ti, torch.arange(T - 1).repeat_interleave(N))
        assert msk.reshape(-1).tolist() == panel.mask[:-1].reshape(-1).tolist()
        assert ensemble.memory.M.size(0) == panel.N, "记忆批量维必须等于资产数"

    # ---- 训练/推理记忆语义一致(纲领验收要求的守卫测试) ----

    def test_val_epoch_matches_oos_inference_router(self, ensemble):
        """Stage2 val 数据流 vs OOS 推理式逐日循环: 信号与记忆终态逐位一致。"""
        T, N = 24, 6
        panel = make_masked_panel(T=T, N=N)
        device = torch.device("cpu")

        trainer = RouterTrainer(ensemble, {"epochs": 1}, device)
        s, t, m, t_idx = trainer._build_dataset(panel)
        loader = DataLoader(TensorDataset(s, t, m, t_idx),
                             batch_size=panel.N, shuffle=False)

        ensemble.eval()
        trainer.layer_proj.eval()
        ensemble.router.temperature = 0.1  # 对齐 inference 模式温度

        # Path A: trainer val epoch(不删行 + mask 门控)
        _, sigs_A, _, _, _ = trainer._run_epoch(
            loader, None, False, return_predictions=True,
        )
        M_A = ensemble.memory.M.clone()

        # Path B: OOS 推理式逐日循环(与 run_full_pipeline_oos.py 同构)
        mu, sd = trainer.norm_stats
        extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
        with torch.no_grad():
            raw = extractor(panel)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        s_norm = ((raw - mu) / sd).clamp(-10.0, 10.0)

        sigs_B = torch.zeros(T - 1, N)
        ensemble.memory.reset_state(1, device)
        with torch.no_grad():
            for t in range(T - 1):
                s_b = s_norm[t].to(device)
                if ensemble.memory.M is None or ensemble.memory.M.size(0) != N:
                    ensemble.memory.reset_state(N, device)
                l0 = trainer.layer_proj["l0"](s_b)
                l1 = trainer.layer_proj["l1"](s_b)
                l2 = trainer.layer_proj["l2"](s_b)
                out = ensemble(s_b, [l0, l1, l2], mode="inference",
                               mask=panel.mask[t].to(device))
                sigs_B[t] = out["signal"].squeeze(-1).cpu()
                ensemble.memory.detach_state()

        sigs_A_2d = sigs_A.squeeze(-1).reshape(T - 1, N)  # 时间主序展开
        assert torch.allclose(sigs_A_2d, sigs_B, atol=1e-5), \
            "训练侧 val 与 OOS 推理的信号必须一致"
        assert torch.allclose(M_A, ensemble.memory.M, atol=1e-6), \
            "记忆终态必须一致 ⇒ 记忆行语义训练/推理对齐"

    def test_val_epoch_matches_oos_inference_joint(self, ensemble):
        """Stage3 同口径守卫: JointTrainer val 数据流 vs OOS 推理循环。"""
        T, N = 24, 6
        panel = make_masked_panel(T=T, N=N)
        device = torch.device("cpu")
        layer_proj = self._make_layer_proj()

        trainer = JointTrainer(ensemble, layer_proj, {"epochs": 1}, device)
        s, t, m, t_idx = trainer._build_dataset(panel)
        loader = DataLoader(TensorDataset(s, t, m, t_idx),
                             batch_size=panel.N, shuffle=False)

        ensemble.eval()
        layer_proj.eval()
        ensemble.router.temperature = 0.1

        _, sigs_A, _, _, _ = trainer._run_epoch(
            loader, None, False, return_predictions=True,
        )
        M_A = ensemble.memory.M.clone()

        mu, sd = trainer.norm_stats
        extractor = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
        with torch.no_grad():
            raw = extractor(panel)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        s_norm = ((raw - mu) / sd).clamp(-10.0, 10.0)

        sigs_B = torch.zeros(T - 1, N)
        ensemble.memory.reset_state(1, device)
        with torch.no_grad():
            for t in range(T - 1):
                s_b = s_norm[t].to(device)
                if ensemble.memory.M is None or ensemble.memory.M.size(0) != N:
                    ensemble.memory.reset_state(N, device)
                l0 = layer_proj["l0"](s_b)
                l1 = layer_proj["l1"](s_b)
                l2 = layer_proj["l2"](s_b)
                out = ensemble(s_b, [l0, l1, l2], mode="inference",
                               mask=panel.mask[t].to(device))
                sigs_B[t] = out["signal"].squeeze(-1).cpu()
                ensemble.memory.detach_state()

        sigs_A_2d = sigs_A.squeeze(-1).reshape(T - 1, N)  # 时间主序展开
        assert torch.allclose(sigs_A_2d, sigs_B, atol=1e-5)
        assert torch.allclose(M_A, ensemble.memory.M, atol=1e-6)
