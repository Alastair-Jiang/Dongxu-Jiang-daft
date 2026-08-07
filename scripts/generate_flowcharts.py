"""
DAFT 架构流程图生成器
生成原始项目(PR前) 和 PR后 的对比架构流程图 PDF
使用 matplotlib 原生绘制，无需额外依赖

输出路径: 用户桌面
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import os

# ---- 中文字体配置 ----
# Windows 系统查找可用中文字体
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans SC",
    "Noto Sans CJK SC", "KaiTi", "SimSun", "STKaiti",
]
_available = {f.name for f in fm.fontManager.ttflist}
_CN_FONT = None
for _f in _CN_FONT_CANDIDATES:
    if _f in _available:
        _CN_FONT = _f
        break

if _CN_FONT is None:
    # Fallback: try to find any font with CJK support
    for _f in fm.fontManager.ttflist:
        if any(k in _f.name for k in ["CJK", "CN", "SC", "Hei", "Song", "Ming"]):
            _CN_FONT = _f.name
            break

if _CN_FONT:
    # 强制使用中文字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 通过 FontProperties 获取字体路径，确保可用
    _CN_FONT_PATH = None
    for _f in fm.fontManager.ttflist:
        if _f.name == _CN_FONT:
            _CN_FONT_PATH = _f.fname
            break
    if _CN_FONT_PATH:
        from matplotlib.font_manager import FontProperties
        _CN_FP = FontProperties(fname=_CN_FONT_PATH)
        print(f"🀄 使用中文字体: {_CN_FONT} ({_CN_FONT_PATH})")
    else:
        _CN_FP = None
        print(f"🀄 使用中文字体: {_CN_FONT}")
else:
    _CN_FP = None
    print("⚠️ 未找到中文字体，中文将无法正常显示")
    plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 全局配色方案
# ============================================================
C = {
    "bg": "#FAFBFC",
    "title": "#1A1A2E",
    "subtitle": "#4A4A6A",
    "box_input": "#E3F2FD",
    "box_expert": "#E8F5E9",
    "box_router": "#FFF3E0",
    "box_memory": "#F3E5F5",
    "box_cdap": "#E0F7FA",
    "box_fusion": "#FCE4EC",
    "box_output": "#E8EAF6",
    "box_hardening": "#FFF8E1",
    "border_input": "#1565C0",
    "border_expert": "#2E7D32",
    "border_router": "#E65100",
    "border_memory": "#7B1FA2",
    "border_cdap": "#00838F",
    "border_fusion": "#C62828",
    "border_output": "#283593",
    "border_hardening": "#F9A825",
    "text": "#212121",
    "text_light": "#616161",
    "arrow": "#546E7A",
    "arrow_danger": "#D32F2F",
    "arrow_success": "#2E7D32",
    "danger_bg": "#FFEBEE",
    "danger_border": "#EF5350",
    "success_bg": "#E8F5E9",
    "success_border": "#43A047",
    "grid": "#ECEFF1",
}

# ============================================================
# 绘图工具函数
# ============================================================

def draw_box(ax, x, y, w, h, text, *, color_bg, color_border, fontsize=10,
             fontcolor=None, bold=False, linewidth=2, corner_radius=0.08,
             text_lines=None):
    """绘制一个圆角矩形框，支持多行文字"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={corner_radius}",
        facecolor=color_bg, edgecolor=color_border,
        linewidth=linewidth, zorder=2
    )
    ax.add_patch(box)

    fc = fontcolor or C["text"]
    if text_lines:
        # 多行文字
        main = text
        sub_lines = text_lines
        ax.text(x, y + h*0.08, main, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=fc, zorder=3, fontfamily="sans-serif")
        for i, line in enumerate(sub_lines):
            ax.text(x, y - h*0.08 - i*h*0.18, line, ha="center", va="center",
                    fontsize=fontsize-2, color=C["text_light"], zorder=3,
                    fontfamily="sans-serif")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal",
                color=fc, zorder=3, fontfamily="sans-serif")


def draw_arrow(ax, x1, y1, x2, y2, *, color=None, style="->", lw=1.8,
               connectionstyle="arc3,rad=0", zorder=1, linestyle="-"):
    """绘制箭头"""
    c = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                connectionstyle=connectionstyle,
                                linestyle=linestyle),
                zorder=zorder)


def draw_dashed_arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, zorder=1, label=""):
    """绘制虚线箭头"""
    c = color or C["arrow"]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw,
                                linestyle="--", connectionstyle="arc3,rad=0"),
                zorder=zorder)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7,
                color=c, fontweight="bold", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


def draw_danger_banner(ax, x, y, w, h, text):
    """绘制红色告警横幅"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=C["danger_bg"], edgecolor=C["danger_border"],
        linewidth=2, linestyle="--", zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9,
            fontweight="bold", color=C["danger_border"], zorder=3)


def draw_success_banner(ax, x, y, w, h, text):
    """绘制绿色成功横幅"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=C["success_bg"], edgecolor=C["success_border"],
        linewidth=2, linestyle="--", zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=9,
            fontweight="bold", color=C["success_border"], zorder=3)


def draw_section_label(ax, x, y, text, color=None):
    """绘制分区标签"""
    ax.text(x, y, text, ha="left", va="center", fontsize=12,
            fontweight="bold", color=color or C["title"], zorder=3,
            fontfamily="sans-serif")


def draw_legend(ax, items, x, y, spacing=0.55):
    """绘制图例"""
    for i, (label, color_bg, color_border) in enumerate(items):
        yi = y - i * spacing
        box = FancyBboxPatch(
            (x, yi - 0.12), 0.45, 0.24,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=color_bg, edgecolor=color_border, linewidth=1.5, zorder=2
        )
        ax.add_patch(box)
        ax.text(x + 0.55, yi, label, ha="left", va="center", fontsize=8,
                color=C["text"], fontfamily="sans-serif")


# ============================================================
# 图表 1: 原始 DAFT 架构 (PR 前)
# ============================================================

def draw_original_architecture(output_path):
    fig, ax = plt.subplots(1, 1, figsize=(22, 28))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 28)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # ---- 标题 ----
    ax.text(11, 27.2, "DAFT 架构流程图 — 原始版本 (PR #1 前)",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=C["title"], fontfamily="sans-serif")
    ax.text(11, 26.4, "CDAP 反馈回路断裂 · 批量级硬化决策 · 混合 Regime 风险",
            ha="center", va="center", fontsize=12,
            color=C["danger_border"], fontfamily="sans-serif")

    # ---- 第一列: 数据输入流 (x=5.5) ----
    draw_box(ax, 5.5, 24.8, 3.6, 1.2, "Market Data\n市场行情数据",
             color_bg=C["box_input"], color_border=C["border_input"],
             fontsize=10, bold=True)
    draw_arrow(ax, 5.5, 24.2, 5.5, 23.5, color=C["arrow"])

    draw_box(ax, 5.5, 23.2, 3.6, 1.2, "Feature Extraction\n特征工程 s_t",
             color_bg=C["box_input"], color_border=C["border_input"],
             fontsize=10, bold=True,
             text_lines=["(B, 200) 市场状态向量"])

    # 三条分支
    draw_arrow(ax, 5.5, 22.6, 5.5, 21.8, color=C["arrow"])

    # ---- 第二列: 三个并行模块 ----
    # 深度层表示 (左 x=1.8)
    draw_box(ax, 1.8, 21.3, 3.2, 1.0, "3-Layer Depth\n三层深度表示",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=9, bold=True,
             text_lines=["L0_raw / L1_base", "L2_composite"])

    # 路由 (中 x=5.5)
    draw_box(ax, 5.5, 21.3, 3.4, 1.0, "Regime Router\n市场状态路由器",
             color_bg=C["box_router"], color_border=C["border_router"],
             fontsize=9, bold=True,
             text_lines=["full_probs (B,8)", "z_t (B,16) latent"])

    # 记忆 (右 x=9.2)
    draw_box(ax, 9.2, 21.3, 3.4, 1.0, "KDA Market Memory\n市场记忆模块",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=9, bold=True,
             text_lines=["M_t (B,128,64)", "retrieved (B,64)"])

    # 连线
    for cx in [1.8, 5.5, 9.2]:
        draw_arrow(ax, 5.5, 22.6, cx, 21.8, color=C["arrow"])

    # ---- 第三层: 4 Experts ----
    draw_box(ax, 5.5, 19.6, 3.6, 1.0, "4 Strategy Experts\n四大策略专家",
             color_bg=C["box_expert"], color_border=C["border_expert"],
             fontsize=9, bold=True,
             text_lines=["Trend | Reversal", "Volatility | Event"])

    draw_arrow(ax, 5.5, 20.8, 5.5, 20.1, color=C["arrow"])
    draw_arrow(ax, 1.8, 20.8, 4.5, 19.7, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 9.2, 20.8, 6.5, 19.7, color=C["arrow"], lw=1.2)

    # ---- CDAP 模块 ----
    draw_box(ax, 5.5, 18.0, 8.0, 2.2, "CDAP: Cross-Dimension Attention Protocol\n交叉维度注意力协议",
             color_bg=C["box_cdap"], color_border=C["border_cdap"],
             fontsize=10, bold=True,
             text_lines=["Joint Attention → routing_mod (8D) | depth_weights (3D) | memory_gate (128D)",
                         "Routing ↔ Depth ↔ Memory 三链路联合调制"])

    draw_arrow(ax, 5.5, 19.1, 5.5, 18.9, color=C["arrow"])
    draw_arrow(ax, 1.8, 20.8, 3.0, 19.1, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 9.2, 20.8, 8.0, 19.1, color=C["arrow"], lw=1.2)

    # CDAP 三个输出
    # routing_mod → left
    draw_arrow(ax, 3.5, 17.3, 1.8, 16.3, color=C["arrow"])
    # depth_weights → center
    draw_arrow(ax, 5.5, 17.1, 5.5, 16.2, color=C["arrow"])
    # memory_gate → right side
    draw_dashed_arrow(ax, 7.8, 17.5, 12.5, 17.5, color=C["arrow_danger"], lw=1.8, label="memory_gate")
    draw_arrow(ax, 12.5, 17.5, 12.5, 15.3, color=C["arrow_danger"])

    # final_routing box
    draw_box(ax, 1.8, 15.8, 3.0, 0.9, "final_routing\n调制后路由权重",
             color_bg=C["box_router"], color_border=C["border_router"],
             fontsize=8, bold=True)
    draw_arrow(ax, 1.8, 15.35, 3.5, 14.2, color=C["arrow"])

    # fused_layers box
    draw_box(ax, 5.5, 15.8, 3.0, 0.9, "fused_layers\n深度层融合",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=8, bold=True)
    draw_arrow(ax, 5.5, 15.35, 4.5, 14.2, color=C["arrow"])

    # ---- Expert Fusion ----
    draw_box(ax, 5.5, 13.7, 5.0, 1.0, "Weighted Expert Fusion\n加权专家融合",
             color_bg=C["box_fusion"], color_border=C["border_fusion"],
             fontsize=9, bold=True,
             text_lines=["signal = Σ w_i · expert_i(s_t) + 0.1 · fused_layers"])

    draw_arrow(ax, 1.8, 15.35, 4.0, 14.15, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 5.5, 15.35, 5.5, 14.2, color=C["arrow"])

    # ---- Trading Signal ----
    draw_box(ax, 5.5, 12.5, 3.6, 0.9, "Trading Signal\n交易信号输出",
             color_bg=C["box_output"], color_border=C["border_output"],
             fontsize=9, bold=True,
             text_lines=["(B, 1) 预期收益"])

    draw_arrow(ax, 5.5, 13.2, 5.5, 12.95, color=C["arrow"])

    # ---- ❌ 问题标注: 反馈回路断裂 ----
    draw_danger_banner(ax, 12.5, 14.8, 8.0, 1.5,
                       "❌ 问题 1: CDAP 反馈回路断裂\n"
                       "memory_gate 被计算但从未传回记忆模块\n"
                       "Depth → Memory 链路缺失\n"
                       "set_external_gate() 方法不存在")

    draw_dashed_arrow(ax, 9.8, 21.3, 12.5, 16.0, color=C["arrow_danger"], lw=1.5,
                      label="应有的反馈回路 (缺失!)")

    # ---- 下半部分: 硬化路径 ----
    ax.plot([1, 20], [11.5, 11.5], color=C["grid"], lw=2, linestyle="-", zorder=1)
    ax.text(11, 11.2, "Inference 模式下的硬化 (Hardening) 路径",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["title"], fontfamily="sans-serif")

    # Regime ID 获取
    draw_box(ax, 5.5, 10.2, 4.0, 0.8, "get_regime_id(z_t)\n获取 Regime ID",
             color_bg=C["box_router"], color_border=C["border_router"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 12.0, 5.5, 10.6, color=C["arrow"])

    # Batch-average routing
    draw_box(ax, 5.5, 9.0, 4.5, 0.9, "routing_avg = mean(full_probs, dim=0)\n❌ 全 batch 取平均",
             color_bg=C["danger_bg"], color_border=C["danger_border"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 9.8, 5.5, 9.45, color=C["arrow"])

    # should_use_fast_path 决策
    draw_box(ax, 5.5, 7.6, 5.0, 1.0,
             "should_use_fast_path(\n"
             "  regime_id[0],  ← ❌ 只看第一个样本!\n"
             "  routing_avg     ← ❌ batch平均分布!\n"
             ")",
             color_bg=C["danger_bg"], color_border=C["danger_border"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 8.55, 5.5, 8.1, color=C["arrow"])

    # 分支: Fast vs Slow
    # Fast Path
    draw_box(ax, 2.2, 6.3, 3.2, 1.0, "FAST PATH\n快速路径",
             color_bg=C["success_bg"], color_border=C["success_border"],
             fontsize=9, bold=True,
             text_lines=["cached_weights[0]\n→ expand to (B,8)"])

    # Slow Path
    draw_box(ax, 8.8, 6.3, 3.2, 1.0, "SLOW PATH\n完整 CDAP",
             color_bg=C["box_cdap"], color_border=C["border_cdap"],
             fontsize=9, bold=True,
             text_lines=["完整 CDAP 计算\n(昂贵路径)"])

    draw_arrow(ax, 4.2, 7.1, 3.8, 6.8, color=C["arrow_success"])
    draw_arrow(ax, 6.8, 7.1, 7.2, 6.8, color=C["arrow"])

    ax.text(5.5, 7.1, "全 batch\n同一路径",
            ha="center", va="center", fontsize=8, color=C["danger_border"],
            fontweight="bold")

    # 底部融合
    draw_box(ax, 5.5, 4.8, 4.0, 0.8, "Weighted Expert Fusion\n加权融合 → Signal",
             color_bg=C["box_fusion"], color_border=C["border_fusion"],
             fontsize=8, bold=True)

    draw_arrow(ax, 2.2, 5.8, 4.5, 5.2, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 8.8, 5.8, 6.5, 5.2, color=C["arrow"], lw=1.2)

    # ---- ❌ 问题标注: 批量级硬化 ----
    draw_danger_banner(ax, 12.5, 9.0, 8.0, 3.8,
                       "❌ 问题 2: 批量级硬化决策\n\n"
                       "regime_id[0] — 仅用第一个样本判断\n"
                       "routing_avg = mean(full_probs, dim=0)\n"
                       "  — 全 batch 路由分布取平均\n\n"
                       "后果:\n"
                       "· 混合 Regime 的 batch → 全走同一路径\n"
                       "· 样本 0 是牛市 → 其余 31 个也被当牛市\n"
                       "· 闪崩样本被错走 fast path → 精度损失")

    # ---- 图例 ----
    draw_legend(ax, [
        ("数据输入层", C["box_input"], C["border_input"]),
        ("策略专家模块", C["box_expert"], C["border_expert"]),
        ("路由模块", C["box_router"], C["border_router"]),
        ("记忆模块", C["box_memory"], C["border_memory"]),
        ("CDAP 协议", C["box_cdap"], C["border_cdap"]),
        ("融合/输出层", C["box_fusion"], C["border_fusion"]),
        ("问题/断裂链路", C["danger_bg"], C["danger_border"]),
    ], x=1.0, y=3.5)

    # 底部署名
    ax.text(11, 0.5, "DAFT (Dynamic Asset-Flow Transformer) — Original Architecture Before PR #1",
            ha="center", va="center", fontsize=9, color=C["text_light"],
            fontfamily="sans-serif", fontstyle="italic")

    plt.tight_layout(pad=1)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=C["bg"],
                edgecolor="none", format="pdf")
    plt.close(fig)
    print(f"✅ 原始架构流程图已保存: {output_path}")


# ============================================================
# 图表 2: PR 后 DAFT 架构
# ============================================================

def draw_pr_architecture(output_path):
    fig, ax = plt.subplots(1, 1, figsize=(22, 28))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 28)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # ---- 标题 ----
    ax.text(11, 27.2, "DAFT 架构流程图 — PR #1 修复后",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=C["title"], fontfamily="sans-serif")
    ax.text(11, 26.4, "CDAP 反馈回路闭合 · 逐样本硬化决策 · 混合 Batch 正确处理",
            ha="center", va="center", fontsize=12,
            color=C["success_border"], fontfamily="sans-serif")

    # ---- 第一列: 数据输入流 ----
    draw_box(ax, 5.5, 24.8, 3.6, 1.2, "Market Data\n市场行情数据",
             color_bg=C["box_input"], color_border=C["border_input"],
             fontsize=10, bold=True)
    draw_arrow(ax, 5.5, 24.2, 5.5, 23.5, color=C["arrow"])

    draw_box(ax, 5.5, 23.2, 3.6, 1.2, "Feature Extraction\n特征工程 s_t",
             color_bg=C["box_input"], color_border=C["border_input"],
             fontsize=10, bold=True,
             text_lines=["(B, 200) 市场状态向量"])

    draw_arrow(ax, 5.5, 22.6, 5.5, 21.8, color=C["arrow"])

    # ---- 三个并行模块 ----
    draw_box(ax, 1.8, 21.3, 3.2, 1.0, "3-Layer Depth\n三层深度表示",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=9, bold=True)
    draw_box(ax, 5.5, 21.3, 3.4, 1.0, "Regime Router\n市场状态路由器",
             color_bg=C["box_router"], color_border=C["border_router"],
             fontsize=9, bold=True)
    draw_box(ax, 9.2, 21.3, 3.4, 1.0, "KDA Market Memory\n市场记忆模块",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=9, bold=True)

    for cx in [1.8, 5.5, 9.2]:
        draw_arrow(ax, 5.5, 22.6, cx, 21.8, color=C["arrow"])

    # ---- 4 Experts ----
    draw_box(ax, 5.5, 19.6, 3.6, 1.0, "4 Strategy Experts\n四大策略专家",
             color_bg=C["box_expert"], color_border=C["border_expert"],
             fontsize=9, bold=True)

    draw_arrow(ax, 5.5, 20.8, 5.5, 20.1, color=C["arrow"])
    draw_arrow(ax, 1.8, 20.8, 4.5, 19.7, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 9.2, 20.8, 6.5, 19.7, color=C["arrow"], lw=1.2)

    # ---- CDAP 模块 ----
    draw_box(ax, 5.5, 17.8, 8.5, 2.4, "CDAP: Cross-Dimension Attention Protocol\n交叉维度注意力协议",
             color_bg=C["box_cdap"], color_border=C["border_cdap"],
             fontsize=10, bold=True,
             text_lines=["Joint Attention → routing_mod | depth_weights | memory_gate",
                         "Routing ↔ Depth ↔ Memory 三链路联合调制"])

    draw_arrow(ax, 5.5, 19.1, 5.5, 18.9, color=C["arrow"])
    draw_arrow(ax, 1.8, 20.8, 3.0, 18.9, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 9.2, 20.8, 8.0, 18.9, color=C["arrow"], lw=1.2)

    # CDAP 三个输出
    draw_arrow(ax, 3.5, 17.0, 1.8, 15.8, color=C["arrow"])
    draw_arrow(ax, 5.5, 16.8, 5.5, 15.8, color=C["arrow"])
    # ✅ memory_gate → set_external_gate (绿色)
    draw_arrow(ax, 8.0, 17.0, 9.2, 16.3, color=C["arrow_success"], lw=2.2)

    draw_box(ax, 1.8, 15.3, 3.0, 0.9, "final_routing\n调制后路由权重",
             color_bg=C["box_router"], color_border=C["border_router"],
             fontsize=8, bold=True)
    draw_box(ax, 5.5, 15.3, 3.0, 0.9, "fused_layers\n深度层融合",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=8, bold=True)

    # ---- ✅ 修复标注 ----
    draw_success_banner(ax, 12.5, 16.6, 8.5, 2.4,
                        "✅ 修复 1: CDAP 反馈回路闭合\n\n"
                        "① CDAP 输出 memory_gate (B,128)\n"
                        "② 调用 self.memory.set_external_gate(memory_gate)\n"
                        "③ 下一时间步: alpha *= _external_gate\n"
                        "④ 消费后自动清除 (一次性门控)\n\n"
                        "Depth → Memory 链路贯通 ✓")

    draw_arrow(ax, 9.2, 16.0, 9.2, 15.3, color=C["arrow"], lw=1.2)

    # ---- 记忆模块详细展示 ----
    draw_box(ax, 9.2, 14.8, 3.6, 1.0, "KDA Memory\n(forget 门控更新)",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=9, bold=True,
             text_lines=["alpha *= _external_gate",
                         "M_t = α · M_{t-1} + β · k ⊗ v"])

    # 反馈回路闭合箭头
    draw_arrow(ax, 11.0, 14.8, 11.0, 15.8, color=C["arrow_success"], lw=1.8,
               connectionstyle="arc3,rad=-0.4")
    ax.text(12.8, 15.3, "闭合\n回路", ha="center", va="center", fontsize=7,
            color=C["success_border"], fontweight="bold")

    # ---- Expert Fusion ----
    draw_box(ax, 5.5, 13.3, 5.0, 1.0, "Weighted Expert Fusion\n加权专家融合",
             color_bg=C["box_fusion"], color_border=C["border_fusion"],
             fontsize=9, bold=True)

    draw_arrow(ax, 1.8, 14.85, 4.0, 13.75, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 5.5, 14.85, 5.5, 13.8, color=C["arrow"])
    draw_arrow(ax, 9.2, 14.3, 6.5, 13.8, color=C["arrow"], lw=1.2)

    # ---- Trading Signal ----
    draw_box(ax, 5.5, 12.0, 3.6, 0.9, "Trading Signal\n交易信号输出",
             color_bg=C["box_output"], color_border=C["border_output"],
             fontsize=9, bold=True)
    draw_arrow(ax, 5.5, 12.8, 5.5, 12.45, color=C["arrow"])

    # ---- 下半部分: 新的硬化路径 ----
    ax.plot([1, 20], [11.0, 11.0], color=C["grid"], lw=2, linestyle="-", zorder=1)
    ax.text(11, 10.7, "Inference 模式下的硬化 (Hardening) 路径 — 逐样本决策",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C["title"], fontfamily="sans-serif")

    # Per-sample regime_id
    draw_box(ax, 5.5, 9.8, 4.5, 0.8, "get_regime_id(z_t) → regime_id (B,)\n✅ 每个样本独立 Regime ID",
             color_bg=C["success_bg"], color_border=C["success_border"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 11.5, 5.5, 10.2, color=C["arrow"])

    # Per-sample for loop
    draw_box(ax, 5.5, 8.3, 5.5, 1.1,
             "for i in range(B):                    ✅ 逐样本循环\n"
             "  rid = regime_id[i]                   ✅ 独立 regime\n"
             "  rprob = full_probs[i]                ✅ 独立路由分布\n"
             "  should_use_fast_path(rid, rprob)     ✅ 独立判断",
             color_bg=C["success_bg"], color_border=C["success_border"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 9.4, 5.5, 8.85, color=C["arrow"])

    # fast_mask
    draw_box(ax, 5.5, 7.0, 4.5, 0.8, "fast_mask = BoolTensor (B,)\n✅ 每个样本独立标记",
             color_bg=C["success_bg"], color_border=C["success_border"],
             fontsize=8, bold=True)

    draw_arrow(ax, 5.5, 7.75, 5.5, 7.4, color=C["arrow"])

    # 分组处理
    ax.text(5.5, 6.1, "分组并行处理:", ha="center", va="center", fontsize=9,
            fontweight="bold", color=C["title"])

    # Fast Path 组
    draw_box(ax, 2.2, 5.2, 3.4, 1.0, "FAST PATH\n快速路径 (n_fast 样本)",
             color_bg=C["success_bg"], color_border=C["success_border"],
             fontsize=9, bold=True,
             text_lines=["cached_weights[i]\n→ 独立缓存权重",
                         "depth = [1/3,1/3,1/3] 均匀",
                         "gate = 1 (不调制记忆)"])

    # Slow Path 组
    draw_box(ax, 8.8, 5.2, 3.4, 1.0, "SLOW PATH\n完整 CDAP (n_slow 样本)",
             color_bg=C["box_cdap"], color_border=C["border_cdap"],
             fontsize=9, bold=True,
             text_lines=["完整 CDAP 仅对慢样本计算",
                         "slow_idx 切片: M[slow_idx]",
                         "gate ← CDAP memory_gate"])

    draw_arrow(ax, 4.2, 6.5, 3.5, 5.7, color=C["arrow_success"], lw=1.8)
    draw_arrow(ax, 6.8, 6.5, 7.5, 5.7, color=C["arrow"], lw=1.8)

    ax.text(5.5, 6.5, "✅ 同一 batch\n不同路径",
            ha="center", va="center", fontsize=8, color=C["success_border"],
            fontweight="bold")

    # Gate 合并
    draw_box(ax, 5.5, 3.8, 5.5, 0.8,
             "full_gate = ones(B,128); full_gate[slow_idx] = memory_gate_slow",
             color_bg=C["box_memory"], color_border=C["border_memory"],
             fontsize=8, bold=True)

    draw_arrow(ax, 2.2, 4.7, 4.0, 4.2, color=C["arrow"], lw=1.2)
    draw_arrow(ax, 8.8, 4.7, 7.0, 4.2, color=C["arrow"], lw=1.2)

    # 底部融合
    draw_box(ax, 5.5, 2.8, 4.5, 0.9, "Weighted Expert Fusion\n加权融合 → Signal",
             color_bg=C["box_fusion"], color_border=C["border_fusion"],
             fontsize=9, bold=True)

    draw_arrow(ax, 5.5, 3.4, 5.5, 3.25, color=C["arrow"])

    # ---- ✅ 修复标注 ----
    draw_success_banner(ax, 13.5, 8.0, 7.5, 2.8,
                        "✅ 修复 2: 逐样本硬化决策\n\n"
                        "① 每个样本独立判断 fast/slow\n"
                        "② 使用 per-sample regime_id[i]\n"
                        "③ 使用 per-sample full_probs[i]\n"
                        "④ 同一 batch 可混合 fast+slow\n"
                        "⑤ slow 样本 gate 来自 CDAP\n"
                        "⑥ fast 样本 gate=1 (不调制)")

    # ---- 图例 ----
    draw_legend(ax, [
        ("数据输入层", C["box_input"], C["border_input"]),
        ("策略专家模块", C["box_expert"], C["border_expert"]),
        ("路由模块", C["box_router"], C["border_router"]),
        ("记忆模块", C["box_memory"], C["border_memory"]),
        ("CDAP 协议", C["box_cdap"], C["border_cdap"]),
        ("融合/输出层", C["box_fusion"], C["border_fusion"]),
        ("✅ 修复/正确链路", C["success_bg"], C["success_border"]),
    ], x=1.0, y=1.8)

    # 底部署名
    ax.text(11, 0.5, "DAFT (Dynamic Asset-Flow Transformer) — Post PR #1 Architecture",
            ha="center", va="center", fontsize=9, color=C["text_light"],
            fontfamily="sans-serif", fontstyle="italic")

    plt.tight_layout(pad=1)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=C["bg"],
                edgecolor="none", format="pdf")
    plt.close(fig)
    print(f"✅ PR修复后架构流程图已保存: {output_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.home() / "桌面"

    desktop.mkdir(parents=True, exist_ok=True)

    path1 = desktop / "DAFT_Architecture_Original_Before_PR.pdf"
    path2 = desktop / "DAFT_Architecture_After_PR.pdf"

    print(f"📁 输出目录: {desktop}")
    print(f"📄 正在生成原始架构流程图...")
    draw_original_architecture(str(path1))

    print(f"📄 正在生成 PR 修复后架构流程图...")
    draw_pr_architecture(str(path2))

    print(f"\n✅ 完成! 两份 PDF 已保存到桌面:")
    print(f"  1. {path1}")
    print(f"  2. {path2}")
