"""
流程图 PDF 生成引擎 (Flowchart PDF Engine)
============================================
基于 matplotlib 的高质量流程图/架构图 PDF 生成工具。

特性:
- 纯 matplotlib 实现，无外部依赖 (无需 graphviz)
- 支持中文 (自动检测系统字体)
- 圆角矩形、箭头、颜色分组、图例
- 导出矢量 PDF，任意缩放不失真

使用方法:
    from flowchart_engine import Canvas, draw_box, draw_arrow, FlowchartConfig

    cfg = FlowchartConfig(title="我的架构图")
    cv = Canvas(cfg, figsize=(22, 18))

    # 绘制组件
    cv.box(5, 10, "数据输入", color="input")
    cv.box(5, 8, "模型处理", color="expert")
    cv.arrow(5, 9.5, 5, 8.5)

    # 导出 PDF
    cv.save("output.pdf")
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Literal
import numpy as np
import os


# ============================================================
# 中文字体自动检测
# ============================================================
_CN_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans SC",
    "Noto Sans CJK SC", "KaiTi", "SimSun", "STKaiti",
    "PingFang SC", "Heiti SC", "Source Han Sans SC",
]

_available_fonts = {f.name for f in fm.fontManager.ttflist}
_CN_FONT_NAME: Optional[str] = None
for _f in _CN_FONT_CANDIDATES:
    if _f in _available_fonts:
        _CN_FONT_NAME = _f
        break

if _CN_FONT_NAME:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_CN_FONT_NAME, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 预设配色方案
# ============================================================
@dataclass
class ColorScheme:
    """一组配色: (填充色, 边框色, 文字色)"""
    fill: str
    border: str
    text: str = "#212121"

# 内置配色
COLORS: Dict[str, ColorScheme] = {
    "input":     ColorScheme("#E3F2FD", "#1565C0"),  # 蓝色 — 数据输入
    "expert":    ColorScheme("#E8F5E9", "#2E7D32"),  # 绿色 — 专家/策略
    "router":    ColorScheme("#FFF3E0", "#E65100"),  # 橙色 — 路由
    "memory":    ColorScheme("#F3E5F5", "#7B1FA2"),  # 紫色 — 记忆
    "cdap":      ColorScheme("#E0F7FA", "#00838F"),  # 青色 — CDAP/注意力
    "fusion":    ColorScheme("#FCE4EC", "#C62828"),  # 粉色 — 融合
    "output":    ColorScheme("#E8EAF6", "#283593"),  # 靛蓝 — 输出
    "danger":    ColorScheme("#FFEBEE", "#D32F2F"),  # 红色 — 问题/错误
    "success":   ColorScheme("#E8F5E9", "#2E7D32"),  # 绿色 — 修复/成功
    "warning":   ColorScheme("#FFF8E1", "#F9A825"),  # 黄色 — 警告
    "hardening": ColorScheme("#FFF8E1", "#F57F17"),  # 琥珀 — 硬化
    "default":   ColorScheme("#F5F5F5", "#9E9E9E"),  # 灰色 — 默认
}

# 全局绘图参数
ARROW_COLOR = "#546E7A"
GRID_COLOR = "#ECEFF1"
TITLE_COLOR = "#1A1A2E"
SUBTITLE_COLOR = "#616161"
BG_COLOR = "#FAFBFC"


# ============================================================
# Canvas — 绘图画布
# ============================================================
@dataclass
class FlowchartConfig:
    """流程图配置"""
    title: str = "Flowchart"
    subtitle: str = ""
    figsize: Tuple[float, float] = (22, 20)
    dpi: int = 200
    output_format: str = "pdf"
    bg_color: str = BG_COLOR


class Canvas:
    """流程图绘制画布。

    坐标系: 左下角为原点 (0, 0)，单位为用户自定义。

    使用示例:
        cfg = FlowchartConfig(title="系统架构", subtitle="v2.0")
        cv = Canvas(cfg)

        cv.box(5, 10, 3, 1, "输入", color="input", fontsize=10)
        cv.box(5, 8,  3, 1, "处理", color="expert")
        cv.arrow(5, 9.5, 5, 8.5)

        cv.danger_banner(5, 6, "这里是问题")
        cv.section_label(1, 12, "训练阶段")

        cv.save("output.pdf")
    """

    def __init__(self, config: FlowchartConfig):
        self.cfg = config
        self.fig, self.ax = plt.subplots(1, 1, figsize=config.figsize)
        self.ax.set_xlim(0, config.figsize[0])
        self.ax.set_ylim(0, config.figsize[1])
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.fig.patch.set_facecolor(config.bg_color)
        self.ax.set_facecolor(config.bg_color)

    # ---- 基础图形 ----

    def box(self, x: float, y: float, w: float, h: float, text: str, *,
            color: str = "default", fontsize: int = 9, bold: bool = False,
            sub_lines: Optional[List[str]] = None,
            corner_radius: float = 0.06, lw: float = 2.0,
            linestyle: str = "-"):
        """绘制带文本的圆角矩形。

        Parameters
        ----------
        x, y : 矩形中心坐标
        w, h : 宽, 高
        text : 主文字 (可用 \\n 换行)
        color : 配色名 (input/expert/router/memory/cdap/fusion/output/danger/success)
        sub_lines : 副文字行列表 (灰色小字)
        """
        cs = COLORS.get(color, COLORS["default"])
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f"round,pad=0.04,rounding_size={corner_radius}",
            facecolor=cs.fill, edgecolor=cs.border,
            linewidth=lw, linestyle=linestyle, zorder=2
        )
        self.ax.add_patch(patch)
        self.ax.text(x, y + (h * 0.06 if sub_lines else 0), text,
                     ha="center", va="center", fontsize=fontsize,
                     fontweight="bold" if bold else "normal",
                     color=cs.text, zorder=3)
        if sub_lines:
            for i, line in enumerate(sub_lines):
                self.ax.text(x, y - h * 0.06 - i * h * 0.16, line,
                             ha="center", va="center", fontsize=fontsize - 2,
                             color=SUBTITLE_COLOR, zorder=3)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *,
              color: Optional[str] = None, lw: float = 1.8,
              style: str = "->", linestyle: str = "-",
              connectionstyle: str = "arc3,rad=0", zorder: int = 1,
              label: str = ""):
        """绘制箭头。

        Parameters
        ----------
        x1, y1 : 起点
        x2, y2 : 终点
        color : 颜色 (默认 ARROW_COLOR)
        linestyle : "-" 实线, "--" 虚线
        label : 箭头中点的标签文字
        """
        c = color or ARROW_COLOR
        self.ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle=style, color=c, lw=lw,
                                         connectionstyle=connectionstyle,
                                         linestyle=linestyle),
                         zorder=zorder)
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(mx, my, label, ha="center", va="bottom", fontsize=7,
                         color=c, fontweight="bold", zorder=4,
                         bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # ---- 标注 ----

    def banner(self, x: float, y: float, w: float, h: float, text: str, *,
               color: str = "danger", fontsize: int = 9):
        """绘制告警/提示横幅"""
        cs = COLORS.get(color, COLORS["danger"])
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            facecolor=cs.fill, edgecolor=cs.border,
            linewidth=2, linestyle="--", zorder=2
        )
        self.ax.add_patch(patch)
        self.ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                     fontweight="bold", color=cs.border, zorder=3)

    def section_label(self, x: float, y: float, text: str):
        """分区标题"""
        self.ax.text(x, y, text, ha="left", va="center", fontsize=13,
                     fontweight="bold", color=TITLE_COLOR, zorder=3)

    def divider(self, y: float):
        """水平分隔线"""
        xlim = self.cfg.figsize[0]
        self.ax.plot([1, xlim - 1], [y, y], color=GRID_COLOR, lw=2, zorder=1)

    def legend(self, items: List[Tuple[str, str]], x: float, y: float,
               spacing: float = 0.55):
        """绘制图例。

        Parameters
        ----------
        items : [(标签, 配色名), ...]
        """
        for i, (label, color_name) in enumerate(items):
            yi = y - i * spacing
            cs = COLORS.get(color_name, COLORS["default"])
            patch = FancyBboxPatch(
                (x, yi - 0.12), 0.45, 0.24,
                boxstyle="round,pad=0.01,rounding_size=0.03",
                facecolor=cs.fill, edgecolor=cs.border, linewidth=1.5, zorder=2
            )
            self.ax.add_patch(patch)
            self.ax.text(x + 0.55, yi, label, ha="left", va="center",
                         fontsize=8, color="#212121")

    def footer(self, text: str):
        """页脚"""
        self.ax.text(self.cfg.figsize[0] / 2, 0.5, text,
                     ha="center", va="center", fontsize=9,
                     color=SUBTITLE_COLOR, fontstyle="italic")

    def save(self, path: str):
        """保存到文件 (PDF/PNG/SVG)"""
        self.fig.savefig(path, dpi=self.cfg.dpi, bbox_inches="tight",
                         facecolor=self.cfg.bg_color, edgecolor="none",
                         format=self.cfg.output_format)
        plt.close(self.fig)
        print(f"✅ 流程图已保存: {path}")


# ============================================================
# 快捷函数 (单次生成用)
# ============================================================

def quick_flowchart(title: str, output_path: str, figsize=(22, 20),
                    subtitle: str = "", dpi: int = 200):
    """快捷创建一个流程图 Canvas。

    Returns
    -------
    Canvas
    """
    return Canvas(FlowchartConfig(
        title=title, subtitle=subtitle, figsize=figsize, dpi=dpi
    ))


# ============================================================
# 预定义流程图模板
# ============================================================

def template_linear_pipeline(stages: List[str], output_path: str,
                              title: str = "Pipeline"):
    """生成线性管道流程图"""
    n = len(stages)
    width = max(16, n * 5)
    height = 10
    cv = Canvas(FlowchartConfig(title=title, figsize=(width, height)))

    spacing = width / (n + 1)
    for i, stage in enumerate(stages):
        x = spacing * (i + 1)
        y = height * 0.6
        cv.box(x, y, 3.5, 1.2, stage, color="input", fontsize=10, bold=True)
        if i < n - 1:
            cv.arrow(x + 1.8, y, x + spacing - 1.8, y)

    cv.footer("Generated by DAFT Flowchart Engine")
    cv.save(output_path)


def template_side_by_side(left_items: List[Tuple[str, str]],
                           right_items: List[Tuple[str, str]],
                           output_path: str,
                           title: str = "Comparison",
                           left_title: str = "Before",
                           right_title: str = "After"):
    """生成左右对比流程图"""
    cv = Canvas(FlowchartConfig(title=title, figsize=(22, 16)))

    # 左列
    cv.section_label(2, 15, left_title)
    for i, (text, color) in enumerate(left_items):
        y = 14 - i * 1.5
        cv.box(6, y, 7, 1.0, text, color=color, fontsize=9)
        if i < len(left_items) - 1:
            cv.arrow(6, y - 0.5, 6, y - 1.0)

    # 右列
    cv.section_label(13, 15, right_title)
    for i, (text, color) in enumerate(right_items):
        y = 14 - i * 1.5
        cv.box(17, y, 7, 1.0, text, color=color, fontsize=9)
        if i < len(right_items) - 1:
            cv.arrow(17, y - 0.5, 17, y - 1.0)

    cv.footer("Generated by DAFT Flowchart Engine")
    cv.save(output_path)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import sys
    print(f"Flowchart Engine loaded. Chinese font: {_CN_FONT_NAME or 'NONE'}")
    print("Available color schemes:", ", ".join(COLORS.keys()))
    print()
    print("Usage example:")
    print("  from flowchart_engine import Canvas, FlowchartConfig")
    print("  cv = Canvas(FlowchartConfig(title='My Chart'))")
    print("  cv.box(5, 10, 3, 1, 'Step 1', color='input')")
    print("  cv.save('output.pdf')")
