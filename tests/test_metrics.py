"""Test utils/metrics.py — Rank IC utilities + A4 dual-condition mask."""

import pytest
import torch

from daft.utils.metrics import eligible_mask


# ── A4 (2026-08-18): 双条件入样 mask ─────────────────────────────────────


class TestA4EligibleMask:
    """K3 纲领 A4: DAFT 与 Ridge 的样本入样口径统一为
    mask[t] AND mask[t+1](信号日可交易且收益实现日可交易)。

    修复前: Ridge 侧入样/评估用 mask[1:] 单条件(未来端), DAFT 侧
    评估同样单条件 —— 涨停日(t 停)但次日复牌的样本被计入 IC,
    高估可交易信号预测力; 停牌恢复日两侧处理不对称。
    """

    def test_shape_and_semantics(self):
        """输出 (T-1, N); 第 t 行 = mask[t] & mask[t+1]。"""
        mask = torch.tensor([
            [True, True, True],
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ])
        out = eligible_mask(mask)
        assert out.shape == (3, 3)
        assert out.dtype == torch.bool
        # t=0: mask[0]&mask[1]
        assert out[0].tolist() == [True, False, True]
        # t=1: mask[1]&mask[2]
        assert out[1].tolist() == [False, False, True]
        # t=2: mask[2]&mask[3]
        assert out[2].tolist() == [False, True, False]

    def test_all_ones_passthrough(self):
        mask = torch.ones(10, 4, dtype=torch.bool)
        out = eligible_mask(mask)
        assert out.shape == (9, 4)
        assert out.all()

    def test_all_zeros(self):
        mask = torch.zeros(6, 3, dtype=torch.bool)
        out = eligible_mask(mask)
        assert out.shape == (5, 3)
        assert not out.any()

    def test_limit_up_day_excluded(self):
        """涨停日(信号日 mask=0)即使次日复牌, 其次日收益不计入 IC
        —— 单条件 mask[1:] 会错误保留该样本, 双条件正确剔除。
        注意 t=0 样本同样被剔除: 其收益实现日 t=1 涨停, 收益不可交易。"""
        mask = torch.tensor([
            [True],   # t=0 正常
            [False],  # t=1 涨停(不可建仓)
            [True],   # t=2 复牌
        ])
        out = eligible_mask(mask)
        # t=0: mask[0]&mask[1] = False(收益日涨停); t=1: False(信号日涨停)
        assert out.tolist() == [[False], [False]]
        # t=2 复牌后若 t=3 正常交易, 该样本入样
        mask2 = torch.cat([mask, torch.tensor([[True]])], dim=0)
        assert eligible_mask(mask2)[2].item() is True

    def test_resume_day_asymmetry_resolved(self):
        """停牌恢复日: 信号日复牌但收益端停的样本同样剔除
        (单条件两侧口径不对称的另一半)。"""
        mask = torch.tensor([
            [False],  # t=0 停牌
            [True],   # t=1 复牌(信号端可交易)
            [False],  # t=2 又停(收益端不可交易)
        ])
        out = eligible_mask(mask)
        # t=1 样本: mask[1]&mask[2] = False → 剔除
        assert out.tolist() == [[False], [False]]

    def test_matches_backtest_engine_formula(self):
        """与 BacktestEngine.run 内部 ret_mask = mask[:-1] & mask[1:]
        逐位一致 —— 评估层与回测层口径对齐。"""
        torch.manual_seed(42)
        mask = torch.rand(20, 8) < 0.8
        out = eligible_mask(mask)
        engine_formula = mask[:-1] & mask[1:]
        assert torch.equal(out, engine_formula)

    def test_float_input_coercion(self):
        """float 0/1 mask 亦接受(与脚本层 panel.mask 用法兼容)。"""
        mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        out = eligible_mask(mask)
        assert out.shape == (2, 2)
        # bool() 语义: 0.0 → False
        assert out[0].tolist() == [True, False]
        assert out[1].tolist() == [False, False]
