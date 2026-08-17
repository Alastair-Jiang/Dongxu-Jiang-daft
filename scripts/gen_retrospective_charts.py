"""复盘报告图表生成器 (2026-08-17) — 输出 SVG 到 docs/assets/retrospective/。

数据来源(全部可追溯, 详见图表内注):
  docs/EXPERIMENT_REGISTRY.md §3 (EXP-20260816-05~13, EXP-20260817-51)
  docs/DECISION_20260930.md §二.阶段5 (周线数据)
  docs/RESEARCH_FINDINGS_2026-08-17.md (300股扩规模 / 容量扫描)
  README.md Current Status (updated 2026-08-16)
"""
from __future__ import annotations

from pathlib import Path
from xml.dom.minidom import parseString

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "retrospective"
OUT.mkdir(parents=True, exist_ok=True)

FONT = 'font-family="Segoe UI,Microsoft YaHei,sans-serif"'

# ---------------------------------------------------------------- 数据(带源)
# (标签, IC, t, Sharpe, 换手, 来源)
DAILY = [
    ("Ridge 基线",          0.0482, 5.19,  0.555, 1.85,  "EXP-20260816-05"),
    ("DAFT quick",          0.0368, 3.65, -1.72,  2.34,  "EXP-20260816-06"),
    ("DAFT λ*=0.7",         0.0274, 2.36, -0.60,  0.98,  "EXP-20260816-07"),
    ("DAFT freq5+分数",      0.0353, 3.50,  0.25,  0.63,  "EXP-20260816-08"),
    ("DAFT --full",         0.0251, 2.51, -1.33,  2.15,  "EXP-20260816-11"),
    ("DAFT 128×4 锚点",      0.0331, 3.11, -0.886, 2.18,  "EXP-20260817-51"),
]
SCALE = [  # 规模/频率维度
    ("Ridge 300股",   0.0535, None, "RESEARCH_FINDINGS_2026-08-17.md"),
    ("DAFT 300股",    0.0,    None, "RESEARCH_FINDINGS_2026-08-17.md (崩至~0)"),
    ("Ridge 周线",    0.0287, None, "DECISION_20260930.md §二.5"),
    ("DAFT 周线最优", -0.0023, None, "DECISION_20260930.md §二.5 (A·123)"),
]

TIMELINE = [
    # (日期, 标题, [副标题行], level: 0=上 1=下 2=下下, anchor)
    ("2026-07-25", "v0.1.0 雏形", ["Initial commit, 框架骨架"], 0, "middle"),
    ("2026-08-07", "v0.2.0 全管道", ["消除 NotImplementedError", "v0.3.0 日志: 样本外对决"], 1, "middle"),
    ("2026-08-09", "说明书 v1.0", ["safe gate 修复(#6)", "n_experts 8→10(#7)"], 0, "middle"),
    ("2026-08-16", "工程修复批次", ["PR#9/#10: 通道契约", "此前实验全部作废"], 1, "middle"),
    ("2026-08-17", "NO-GO 落闸", ["周线红灯 → 正式 NO-GO", "项目转研究"], 0, "end"),
    ("2026-08-17", "研究+Transformer", ["消融/容量/特征/专家层", "Transformer 批次(暂停)"], 2, "end"),
]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def save(name: str, body: str) -> None:
    p = OUT / name
    p.write_text(body, encoding="utf-8")
    parseString(body)  # XML 合法性自检
    print(f"  {name} ({p.stat().st_size}B)")


# ---------------------------------------------------------------- 1. 时间线
def timeline() -> None:
    W, H = 1060, 320
    x0, x1, y = 90, 1010, 150
    def px(d: str) -> float:
        m, dd = int(d[5:7]), int(d[8:10])
        day = (m - 7) * 31 + dd   # 07-25→25, 08-17→48
        return x0 + (day - 25) / 23 * (x1 - x0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{W/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" {FONT}>DAFT 版本演进时间线 (2026-07-25 → 2026-08-17)</text>',
             f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#555" stroke-width="2"/>']
    colors = ["#8ab4f8", "#8ab4f8", "#8ab4f8", "#fdd663", "#f28b82", "#81c995"]
    LEVEL_TY = {0: y - 78, 1: y + 32, 2: y + 118}
    for i, (d, title, subs, level, anchor) in enumerate(TIMELINE):
        x = px(d)
        ty = LEVEL_TY[level]
        tx = min(x, 1040) if anchor == "end" else x
        parts.append(f'<circle cx="{x:.0f}" cy="{y}" r="7" fill="{colors[i]}" stroke="#333"/>')
        link_y = ty + 14 if level == 0 else ty - 14
        parts.append(f'<line x1="{x:.0f}" y1="{y}" x2="{x:.0f}" y2="{link_y}" stroke="#999"/>')
        parts.append(f'<text x="{tx:.0f}" y="{ty}" text-anchor="{anchor}" font-size="12.5" font-weight="bold" {FONT}>{esc(title)}</text>')
        parts.append(f'<text x="{tx:.0f}" y="{ty+15}" text-anchor="{anchor}" font-size="10.5" fill="#555" {FONT}>{d}</text>')
        for j, line in enumerate(subs):
            parts.append(f'<text x="{tx:.0f}" y="{ty+29+j*13}" text-anchor="{anchor}" font-size="9.5" fill="#777" {FONT}>{esc(line)}</text>')
    parts.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="10" fill="#888" {FONT}>数据来源: git log (d7aa39c…6142f1f, 50 commits) · 版本号取自提交信息与 README/说明书标题</text>')
    parts.append("</svg>")
    save("timeline.svg", "".join(parts))


# ---------------------------------------------------------------- 2/3. 柱状图
def bars(name: str, title: str, rows, idx: int, unit: str, ridge_line: float | None, src: str) -> None:
    W, H = 1060, 420
    left, right, top, bottom = 70, 30, 50, 110
    plot_w, plot_h = W - left - right, H - top - bottom
    vals = [r[idx] for r in rows]
    vmax = max(max(vals), 0.0)
    vmin = min(min(vals), 0.0)
    if ridge_line is not None:
        vmax = max(vmax, ridge_line)
        vmin = min(vmin, ridge_line)
    span = (vmax - vmin) or 1.0
    vmax += span * 0.12
    vmin -= span * 0.12
    span = vmax - vmin

    def py(v: float) -> float:
        return top + (vmax - v) / span * plot_h

    n = len(rows)
    slot = plot_w / n
    bw = slot * 0.58
    zero_y = py(0.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{W/2}" y="28" text-anchor="middle" font-size="17" font-weight="bold" {FONT}>{esc(title)}</text>',
             f'<line x1="{left}" y1="{zero_y:.1f}" x2="{W-right}" y2="{zero_y:.1f}" stroke="#333" stroke-width="1.2"/>']
    for i, r in enumerate(rows):
        v = r[idx]
        x = left + i * slot + (slot - bw) / 2
        y1, y2 = py(max(v, 0)), py(min(v, 0))
        color = "#1a73e8" if v >= 0 else "#d93025"
        if "Ridge" in r[0]:
            color = "#188038" if v >= 0 else "#d93025"
        parts.append(f'<rect x="{x:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{max(y2-y1,1):.1f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y1-6:.1f}" text-anchor="middle" font-size="12" font-weight="bold" {FONT}>{v:+.4f}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{H-84}" text-anchor="middle" font-size="11.5" {FONT}>{esc(r[0])}</text>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{H-70}" text-anchor="middle" font-size="9" fill="#888" {FONT}>{esc(r[-1][:36])}</text>')
    if ridge_line is not None:
        ly = py(ridge_line)
        parts.append(f'<line x1="{left}" y1="{ly:.1f}" x2="{W-right}" y2="{ly:.1f}" stroke="#188038" stroke-width="1.4" stroke-dasharray="6,4"/>')
        parts.append(f'<text x="{W-right-4}" y="{ly-5:.1f}" text-anchor="end" font-size="11" fill="#188038" {FONT}>Ridge 基线 {ridge_line:+.4f}</text>')
    # y 轴刻度
    import math
    step = span / 6
    for k in range(7):
        v = vmax - k * step
        yy = py(v)
        parts.append(f'<line x1="{left-4}" y1="{yy:.1f}" x2="{W-right}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="10" fill="#666" {FONT}>{v:.3f}</text>')
    parts.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="10" fill="#888" {FONT}>{esc(src)}</text>')
    parts.append("</svg>")
    save(name, "".join(parts))


# ---------------------------------------------------------------- 4. 雷达图
def radar() -> None:
    import math
    # 维度: IC, t-stat, 净Sharpe, 换手控制(min/turnover)
    rows = [r for r in DAILY if r[0] in ("Ridge 基线", "DAFT quick", "DAFT freq5+分数", "DAFT 128×4 锚点")]
    dims = ["Rank IC", "IC t-stat", "净 Sharpe(归一)", "换手控制(归一)"]
    ic = [r[1] for r in rows]
    tt = [r[2] for r in rows]
    sp = [r[3] for r in rows]
    to = [r[4] for r in rows]
    smin, smax = min(sp), max(sp)
    tomin = min(to)
    norm = []
    for i in range(len(rows)):
        norm.append([
            ic[i] / max(ic),
            tt[i] / max(tt),
            (sp[i] - smin) / (smax - smin),
            tomin / to[i],
        ])
    W, H = 720, 520
    cx, cy, R = 330, 260, 170
    colors = ["#188038", "#1a73e8", "#f9ab00", "#7b1fa2"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<text x="{W/2}" y="30" text-anchor="middle" font-size="17" font-weight="bold" {FONT}>能力雷达图 (100 股 hs300 日频, 各维归一到对比集最大值)</text>']
    nd = len(dims)
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{cx + R*ring*math.cos(2*math.pi*k/nd - math.pi/2):.1f},{cy + R*ring*math.sin(2*math.pi*k/nd - math.pi/2):.1f}" for k in range(nd))
        parts.append(f'<polygon points="{pts}" fill="none" stroke="#ddd"/>')
    for k in range(nd):
        ang = 2 * math.pi * k / nd - math.pi / 2
        x2, y2 = cx + R * math.cos(ang), cy + R * math.sin(ang)
        parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#ccc"/>')
        lx, ly = cx + (R + 26) * math.cos(ang), cy + (R + 26) * math.sin(ang)
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="12.5" font-weight="bold" {FONT}>{esc(dims[k])}</text>')
    for i, r in enumerate(rows):
        pts = " ".join(
            f"{cx + R*norm[i][k]*math.cos(2*math.pi*k/nd - math.pi/2):.1f},{cy + R*norm[i][k]*math.sin(2*math.pi*k/nd - math.pi/2):.1f}"
            for k in range(nd))
        parts.append(f'<polygon points="{pts}" fill="{colors[i]}33" stroke="{colors[i]}" stroke-width="2"/>')
        parts.append(f'<text x="540" y="{120 + i*24}" font-size="12.5" fill="{colors[i]}" {FONT}>■ {esc(r[0])}</text>')
    parts.append(f'<text x="540" y="{120 + len(rows)*24 + 8}" font-size="10" fill="#666" {FONT}>Sharpe 归一: (s−min)/(max−min)</text>')
    parts.append(f'<text x="540" y="{120 + len(rows)*24 + 22}" font-size="10" fill="#666" {FONT}>换手控制: min(turnover)/turnover</text>')
    parts.append(f'<text x="{W/2}" y="{H-14}" text-anchor="middle" font-size="10" fill="#888" {FONT}>数据来源: docs/EXPERIMENT_REGISTRY.md EXP-20260816-05/06/08, EXP-20260817-51</text>')
    parts.append("</svg>")
    save("radar.svg", "".join(parts))


if __name__ == "__main__":
    timeline()
    bars("ic_bars.svg", "样本外 Rank IC 对比 (100 股 hs300 日频 + 规模/频率维度)",
         DAILY + SCALE, 1, "IC", 0.0482,
         "数据来源: EXPERIMENT_REGISTRY.md §3 / DECISION_20260930.md §二.5 / RESEARCH_FINDINGS_2026-08-17.md · DAFT=蓝 Ridge=绿 负值=红")
    bars("sharpe_bars.svg", "样本外净 Sharpe 对比 (扣 5bp+1bp 成本, 100 股日频)",
         DAILY, 3, "Sharpe", 0.555,
         "数据来源: EXPERIMENT_REGISTRY.md §3 (EXP-20260816-05~11, EXP-20260817-51) · DAFT=蓝 Ridge=绿 负值=红")
    radar()
    print("charts done ->", OUT)
