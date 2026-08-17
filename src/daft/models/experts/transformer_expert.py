"""Transformer-encoder expert (2026-08-17 研究项目: 架构升级).

MLP 专家层在全部维度下输给 Ridge (IC 0.032 vs 0.048), 且 --full /
宽度 512 / 300 股规模下进一步退化。下一步研究假设: 样本内 s_t 的
200 维特征之间存在结构化交互 (g1 价格 ↔ g5 截面 ↔ g6 动量), MLP
的全连接混合无法高效建模; 引入 Transformer 架构对特征做自注意力
(Set-Transformer 风格) 让模型显式学习特征间依赖。

设计
----
- 200 维特征 → 40 个 token × 5 维 (固定分块, 与数据管线零耦合)。
- token 线性嵌入 + 可学习位置编码 → pre-LN TransformerEncoder
  (GELU FFN, 4× 宽度), n_layers 个块。
- token 均值池化 → head → SiTU (与 BaseExpert 输出契约一致)。

接口完全兼容 BaseExpert: 与 Stage1/2/3、路由、CDAP、记忆模块无缝替换。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from daft.models.experts.base_expert import BaseExpert


class _FeatureTokenEncoder(nn.Module):
    """特征维自注意力编码器: (B, input_dim) → (B, hidden_dim)。"""

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tokens: int = 40,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_dim % n_tokens == 0, f"{input_dim} 不可被 {n_tokens} 个 token 整除"
        assert hidden_dim % n_heads == 0, f"hidden {hidden_dim} 不可被 {n_heads} 头整除"
        self.n_tokens = n_tokens
        token_dim = input_dim // n_tokens
        self.embed = nn.Linear(token_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: 深层训练稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = x.view(B, self.n_tokens, -1)          # (B, n_tokens, token_dim)
        h = self.embed(tokens) + self.pos              # (B, n_tokens, hidden)
        h = self.encoder(h)                            # (B, n_tokens, hidden)
        return h.mean(dim=1)                           # (B, hidden)


class TransformerExpert(BaseExpert):
    """通用 Transformer 专家 (特征自注意力主干)。

    与 5 类 regime 专家不同, 该专家不做 regime 过滤: 全量数据训练
    (此前实验已证伪 regime 专业化假设, 全量训练为最优专家层配置)。
    10 个实例共享架构、独立初始化, 由路由器学习分工。
    """

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tokens: int = 40,
        n_heads: int = 4,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            name="transformer",
        )
        # 替换 BaseExpert 的 MLP backbone 为 Transformer 编码器
        self.backbone = _FeatureTokenEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_tokens=n_tokens,
            n_heads=n_heads,
        )

    def _regime_filter(self, panel) -> torch.Tensor:
        """全量数据: 返回全 True (regime 专业化已被证伪)。"""
        T = panel.values.shape[0]
        return torch.ones(T, dtype=torch.bool)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """通用专家损失: 可交易掩码下的 MSE。"""
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        mask = mask.squeeze(-1)
        se = (target - pred) ** 2
        mask_f = mask.float()  # DirectML 兼容: bool.sum() 需转 float
        return (se * mask_f).sum() / mask_f.sum().clamp(min=1)
