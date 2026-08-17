# -*- coding: utf-8 -*-
"""DAFT 全景图生成器 v2(三段式 + 自动布局 lint) — 输出 SVG 到 docs/assets/architecture/。

Part 1  通俗视图: "会自我校准的 10 人投资小组"比喻, 面向无金融/ML 背景读者
Part 2  工程视图: 数据流 × 路由/记忆/深度三维调制闭环(信息无损, 专业公式)
Part 3  具体案例: 一个交易日的输入 → 五步处理 → 输出信号与组合动作

v2 布局 lint(0 问题才允许产出):
  L1 文字越画布 / L2 文字溢出所属框 / L3 文字-文字重叠 / L4 文字-异属框重叠
  L5 框-框重叠(药丸 vs 卡片) / L6 箭头穿过文字(非 halo) / L7 箭头切入框内部

用法: python scripts/gen_architecture_overview.py
"""
from __future__ import annotations
import os
import re

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def tw(s, size):
    """保守文本宽度估算(CJK≈1.04em, ASCII≈0.58em)。"""
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o >= 0x2E80 or o >= 0x1F000:
            w += size * 1.04
        elif ch in "WM%@Ω←→⊙⊗∈≈·":
            w += size * 0.92
        elif ch == " ":
            w += size * 0.32
        else:
            w += size * 0.58
    return w

W = 1600
PART2_Y = 660
PART3_Y = PART2_Y + 1428
H = PART3_Y + 660

P = []
ISSUES = []
CUR = {"tag": ""}                 # 当前文字所属元素标签
OFF = {"dy": 0}                   # 当前 Y 平移(Part 2 在 <g translate> 内)
TB = []                           # 文字 bbox 列表(已含平移)
RC = []                           # 框(矩形)列表(已含平移)
AR = []                           # 箭头采样点列表(已含平移)

def A(s): P.append(s)

C = dict(
    data=("#ECEFF1", "#546E7A", "#37474F"),
    blue=("#E3F2FD", "#1565C0", "#0D47A1"),
    green=("#E8F5E9", "#2E7D32", "#1B5E20"),
    purple=("#F3E5F5", "#6A1B9A", "#4A148C"),
    orange=("#FFF3E0", "#E65100", "#BF360C"),
    pink=("#FCE4EC", "#AD1457", "#880E4F"),
    red=("#FFEBEE", "#C62828", "#B71C1C"),
    ok=("#E8F5E9", "#2E7D32", "#1B5E20"),
    cream=("#FFFDE7", "#F9A825", "#F57F17"),
)

def text(x, y, s, size=12, fill="#26323B", anchor="start", weight="400",
         halo=False, italic=False):
    w = tw(s, size)
    x0, x1 = (x - w / 2, x + w / 2) if anchor == "middle" else \
             ((x - w, x) if anchor == "end" else (x, x + w))
    y0, y1 = y - size * 0.82, y + size * 0.28
    TB.append(dict(x0=x0, y0=y0 + OFF["dy"], x1=x1, y1=y1 + OFF["dy"],
                   tag=CUR["tag"], halo=halo, s=s))
    st = 'font-style="italic" ' if italic else ''
    if halo:
        A(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
          f'font-weight="{weight}" {st}stroke="#FAFBFC" stroke-width="3.5" paint-order="stroke">{esc(s)}</text>')
    else:
        A(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
          f'font-weight="{weight}" {st}>{esc(s)}</text>')

def rect(x, y, w, h, tag, fill, stroke, sw=1.8, rx=10, opacity=None):
    RC.append(dict(x0=x, y0=y + OFF["dy"], x1=x + w, y1=y + h + OFF["dy"], tag=tag))
    op = f' fill-opacity="{opacity}"' if opacity else ''
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"'
      f'{op} stroke="{stroke}" stroke-width="{sw}"/>')

def box(x, y, w, h, ckey, title, lines, tsize=14, lsize=11.5, lh=17, pad=12):
    fill, stroke, tcol = C[ckey]
    rect(x, y, w, h, title, fill, stroke, 1.8)
    CUR["tag"] = title
    ty = y + 22
    text(x + pad, ty, title, tsize, tcol, weight="700")
    yy = ty + 22
    for ln in lines:
        text(x + pad, yy, ln, lsize, "#37474F")
        yy += lh
    CUR["tag"] = ""
    return yy

def pill(cx, cy, s, ckey, size=10.5):
    fill, stroke, tcol = C[ckey]
    w = tw(s, size) + 18
    rect(cx - w / 2, cy - 11, w, 22, s, fill, stroke, 1.2, rx=11)
    CUR["tag"] = s
    text(cx, cy + 3.8, s, size, tcol, anchor="middle", weight="600")
    CUR["tag"] = ""

_NUM = r"-?[\d.]+"
_TOK = re.compile(r"([MLC])([^MLC]+)")
def _sample_path(d):
    """采样 M/L/C 路径为点列。"""
    pts, cur = [], None
    for cmd, args in _TOK.findall(d):
        nums = [float(v) for v in re.findall(_NUM, args)]
        if cmd == "M":
            cur = (nums[0], nums[1]); pts.append(cur)
        elif cmd == "L":
            for i in range(0, len(nums), 2):
                p = (nums[i], nums[i + 1])
                for k in range(1, 11):
                    t = k / 10
                    pts.append((cur[0] + (p[0] - cur[0]) * t,
                                cur[1] + (p[1] - cur[1]) * t))
                cur = p
        elif cmd == "C" and len(nums) == 6:
            p0, p1, p2, p3 = cur, (nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])
            for k in range(1, 25):
                t = k / 24; u = 1 - t
                pts.append((u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                            u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]))
            cur = p3
    return pts

def arrow(d, ckey, w=2.2, dash=None):
    stroke = dict(gray="#546E7A", blue="#1565C0", green="#2E7D32",
                  purple="#6A1B9A", orange="#E65100", pink="#AD1457")[ckey]
    AR.append([(px, py + OFF["dy"]) for (px, py) in _sample_path(d)])
    dd = f' stroke-dasharray="{dash}"' if dash else ''
    A(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{w}"{dd} '
      f'marker-end="url(#ah-{ckey})" opacity="0.9"/>')

# ════════════════════════ 画布 ════════════════════════
A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
  f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Microsoft YaHei, PingFang SC, sans-serif">')
A(f'<rect width="{W}" height="{H}" fill="#FAFBFC"/>')
A('<defs>')
for k, hexc in [("gray", "#546E7A"), ("blue", "#1565C0"), ("green", "#2E7D32"),
                ("purple", "#6A1B9A"), ("orange", "#E65100"), ("pink", "#AD1457")]:
    A(f'<marker id="ah-{k}" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto">'
      f'<polygon points="0 0, 9 3.5, 0 7" fill="{hexc}"/></marker>')
A('</defs>')

# ════════════════════════ Part 1 · 通俗视图 ════════════════════════
text(40, 52, "Part 1 · 三分钟看懂: DAFT 是一个会自我校准的「10 人投资小组」", 26, "#263238", weight="800")
text(40, 82, "不需要任何金融或机器学习背景 · 每个气泡都是大白话 · 括号里是 Part 2 工程视图中的对应模块", 13, "#546E7A")

CARDS = [
    ("① 市场日报", "#ECEFF1", "#546E7A", "#37474F", "报纸",
     ["每天自动下载 300 只股票的", "价格和成交量 —— 相当于", "一份「股市日报」。", "", "(工程视图: 数据层 Panel)"]),
    ("② 10 位分析师", "#FCE4EC", "#AD1457", "#880E4F", "人们",
     ["10 位风格不同的分析师读", "同一份日报: 有趋势派、反转", "派、波动派…… 每人给出一个", "−1 到 +1 的分数(看空到看多)。", "(工程视图: 专家池 MoE)"]),
    ("③ 组长分派", "#E3F2FD", "#1565C0", "#0D47A1", "方向盘",
     ["组长看一眼当前行情:", "「现在是震荡市, 多听反转派,", "少听趋势派。」 每个行情下", "听谁的权重都不一样。", "(工程视图: RegimeRouter 路由)"]),
    ("④ 团队笔记本", "#E8F5E9", "#2E7D32", "#1B5E20", "笔记本",
     ["小组有一本会自动淡忘的", "笔记本: 记住最近发生过的", "行情, 太久远的慢慢忘掉,", "以免被旧经验误导。", "(工程视图: KDA 市场记忆)"]),
    ("⑤ 圆桌会议", "#FFF3E0", "#E65100", "#BF360C", "圆桌",
     ["组长、笔记本、多层经验", "三者坐下来互相校准:", "笔记不确定时就不乱改主意;", "行情突变时组长重新分派。", "(工程视图: CDAP ★原创)"]),
]
CX0, CW, CGAP, CY, CH = 50, 284, 20, 116, 296
for i, (title, fill, stroke, tcol, icon, lines) in enumerate(CARDS):
    x = CX0 + i * (CW + CGAP)
    rect(x, CY, CW, CH, title, fill, stroke, 2, rx=12, opacity=0.55)
    ix, iy = x + CW / 2, CY + 46
    if icon == "报纸":
        A(f'<rect x="{ix-20}" y="{iy-15}" width="40" height="30" rx="2" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.6"/>')
        A(f'<rect x="{ix-15}" y="{iy-9}" width="10" height="7" fill="{stroke}"/>')
        for k in range(3):
            A(f'<line x1="{ix-2}" y1="{iy-8+k*7}" x2="{ix+15}" y2="{iy-8+k*7}" stroke="{stroke}" stroke-width="1.4"/>')
    elif icon == "人们":
        for k in range(3):
            px = ix - 14 + k * 14
            A(f'<circle cx="{px}" cy="{iy-8}" r="5" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.8"/>')
            A(f'<path d="M {px-6} {iy+12} Q {px} {iy-1} {px+6} {iy+12} Z" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.6"/>')
    elif icon == "方向盘":
        A(f'<circle cx="{ix}" cy="{iy}" r="15" fill="#FFFFFF" stroke="{stroke}" stroke-width="2.4"/>')
        A(f'<circle cx="{ix}" cy="{iy}" r="3.2" fill="{stroke}"/>')
        for dx, dy in [(-13, 0), (13, 0), (0, 13)]:
            A(f'<line x1="{ix}" y1="{iy}" x2="{ix+dx}" y2="{iy+dy}" stroke="{stroke}" stroke-width="2.2"/>')
    elif icon == "笔记本":
        A(f'<rect x="{ix-15}" y="{iy-14}" width="30" height="28" rx="3" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.8"/>')
        for k in range(3):
            A(f'<line x1="{ix-9}" y1="{iy-6+k*8}" x2="{ix+9}" y2="{iy-6+k*8}" stroke="{stroke}" stroke-width="1.4"/>')
        A(f'<circle cx="{ix-15}" cy="{iy-8}" r="2.6" fill="none" stroke="{stroke}" stroke-width="1.4"/>')
        A(f'<circle cx="{ix-15}" cy="{iy+6}" r="2.6" fill="none" stroke="{stroke}" stroke-width="1.4"/>')
    elif icon == "圆桌":
        A(f'<ellipse cx="{ix}" cy="{iy+4}" rx="22" ry="9" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.8"/>')
        for dx, dy in [(-24, -6), (24, -6), (0, -14)]:
            A(f'<circle cx="{ix+dx}" cy="{iy+dy}" r="4.5" fill="{stroke}"/>')
    CUR["tag"] = title
    text(x + CW / 2, CY + 92, title, 15.5, tcol, anchor="middle", weight="800")
    yy = CY + 120
    for ln in lines:
        if ln:
            text(x + 16, yy, ln, 11.8, "#37474F")
        yy += 18.5
    CUR["tag"] = ""
    if i < 4:
        ax = x + CW + 3
        arrow(f"M {ax} {CY+CH/2} L {ax+14} {CY+CH/2}", "gray", 3)

rect(50, CY + CH + 24, 1500, 76, "结果条", "#FFFFFF", "#90A4AE", 1.6)
CUR["tag"] = "结果条"
text(72, CY + CH + 52, "每天产出:", 14, "#263238", weight="800")
text(160, CY + CH + 52, "每只股票一个分数 → 全市场排名 → 分数最高的前 20% 买入、最低的 20% 卖出(做多做空)", 13.5, "#37474F")
text(160, CY + CH + 72, "→ 每笔交易扣除约 0.06% 成本 → 每日/每周重复, 用 5 年历史数据检验总成绩", 13.5, "#546E7A")
CUR["tag"] = ""

rect(50, CY + CH + 118, 1500, 64, "创新横幅", "#FFF3E0", "#E65100", 2)
CUR["tag"] = "创新横幅"
text(800, CY + CH + 146, "★ 项目核心创新 = 第 ⑤ 步圆桌会议: 让「听谁的(路由)」「记什么(记忆)」「信哪层经验(深度)」互相影响", 15, "#BF360C", anchor="middle", weight="800")
text(800, CY + CH + 168, "传统系统里这三件事各干各的; DAFT 让它们坐上同一张桌子 —— 这是对 Kimi K3 三个组件的闭环化改造", 12.5, "#E65100", anchor="middle")
CUR["tag"] = ""

# ════════════════════════ Part 2 · 工程视图 ════════════════════════
A(f'<g transform="translate(0,{PART2_Y})">')
OFF["dy"] = PART2_Y

text(40, 44, "Part 2 · 工程视图 —— 数据流 × 三维调制闭环(信息无损)", 24, "#263238", weight="800")
text(40, 72, "路由 / 记忆 / 深度三维度在 R^64 联合潜空间互相调制 —— 对 Kimi K3 三组件「孤岛」的闭环化改造 · 与 Part 1 五个气泡一一对应", 12.5, "#546E7A")

box(40, 150, 215, 86, "data", "市场数据 Panel",
    ["(T, N, 5) OHLCV · mask", "baostock 前复权 · 涨跌停 mask"])
arrow("M 147 236 L 147 264", "gray")
box(40, 268, 215, 84, "data", "通道契约",
    ["ensure_base_panel", "OHLCV → 5 基础特征 · 唯一转换点"], lsize=10.8)
arrow("M 147 352 L 147 364", "gray")
_feat = box(40, 368, 215, 150, "data", "特征层级 features",
            ["L0 → L1 → L2 · 6 组因子", "因果 · mask 感知推导"])
CUR["tag"] = "特征层级 features"
for i, (lab, fc) in enumerate([("L0 原始 close/vol", "#CFD8DC"),
                               ("L1 基础 MA/RSI/vol", "#B0BEC5"),
                               ("L2 复合 regime/risk", "#90A4AE")]):
    rect(54, 428 + i * 27, 187, 22, "特征层级 features", fc, "#B0BEC5", 0.8, rx=5)
    text(62, 443 + i * 27, lab, 10.5, "#263238")
CUR["tag"] = ""
arrow("M 255 438 L 296 438", "gray")

rect(300, 395, 180, 86, "s_t ∈ R^200", "#37474F", "#263238", 2, rx=12)
CUR["tag"] = "s_t ∈ R^200"
text(312, 420, "s_t ∈ R^200", 16, "#FFFFFF", weight="800")
text(312, 441, "逐特征 z-norm", 11, "#CFD8DC")
text(312, 458, "(train-only 统计 · A2 已修)", 10.5, "#CFD8DC")
CUR["tag"] = ""

arrow("M 480 415 C 514 415, 508 215, 536 215", "blue")
arrow("M 480 432 L 536 424", "green")
arrow("M 480 455 C 514 455, 508 620, 536 620", "purple")

# 路由维
pill(615, 137, "K3: Stable LatentMoE", "data", 10)
pill(775, 137, "消融 +0.018 IC", "ok", 10)
box(540, 150, 260, 130, "blue", "路由维 RegimeRouter",
    ["z_t = LN(W↑·SiTU(W↓·s_t)) ∈ R^16",
     "logits = W·z_t + QuantileBalance",
     "p = softmax(logits / T) → 10 专家权重",
     "训练期探索噪声 · 推理温度 0.1"])
pill(650, 294, "A7 top-3 未落实(稠密加权)", "red", 10)

# 记忆维
pill(640, 318, "K3: KDA", "data", 10)
pill(770, 318, "A3 行语义 训练≠推理(待修)", "red", 10)
box(540, 330, 260, 180, "green", "记忆维 KDAMarketMemory",
    ["α = lb+(1−lb)·σ(exp(A)·(f(s_t)+b))",
     "β = σ(w·s_t) 可学习写入步长",
     "M ← α⊙M − β·k⊗(Mk) + β·k⊗v δ规则",
     "o = RMSNorm(gate(Mᵀq)) ∈ R^64",
     "槽位: 每股一份 (N,128,64) ≈ 32KB"])
pill(766, 524, "消融 ≈ 0", "ok", 10)

# z_t: 路由 → 记忆遗忘门(左侧通道, 避开徽标)
arrow("M 560 284 C 545 298, 545 312, 560 328", "blue", 1.8)
text(496, 310, "z_t 调制遗忘", 10.5, "#1565C0", anchor="end", halo=True)

# 深度维
pill(612, 547, "K3: AttnRes", "data", 10)
box(540, 560, 260, 120, "purple", "深度维 layer_proj",
    ["l0 / l1 / l2 : 200→128→64 投影",
     "输出特征层级 [L0, L1, L2]",
     "fused = Σ w_k·L_k (w 由 CDAP 重标)"])
pill(620, 694, "128×4 层 +0.012 IC", "ok", 10)

# CDAP
rect(880, 300, 270, 270, "CDAP 联合空间 R^64", "#FFF3E0", "#E65100", 2.4, rx=14, opacity=0.97)
CUR["tag"] = "CDAP 联合空间 R^64"
text(894, 326, "CDAP 联合空间 R^64", 15, "#BF360C", weight="800")
text(894, 350, "正向投影:", 11.5, "#37474F", weight="600")
text(894, 368, "e = E(p) m = E_m(M̄) d = E_d(L0;L1;L2)", 10.8, "#37474F")
text(1015, 412, "j = e ⊙ m ⊙ d", 21, "#E65100", anchor="middle", weight="800")
text(894, 448, "反向投影 (tanh scale 零初始化):", 11.5, "#37474F", weight="600")
text(894, 468, "→ 路由 bias(logit 空间)", 11.5, "#37474F")
text(894, 488, "→ 记忆门 g = σ(·) → α ← α·g", 11.5, "#37474F")
text(894, 508, "→ 深度权重 w = softmax(·)", 11.5, "#37474F")
text(894, 540, "核心创新: 三维互相调制的闭环", 12, "#BF360C", weight="700")
CUR["tag"] = ""

arrow("M 800 210 C 842 210, 846 338, 876 346", "blue")
text(826, 268, "e ← p", 11, "#1565C0", halo=True)
arrow("M 800 424 L 876 428", "green")
text(874, 410, "m←M̄ 池化", 10.5, "#2E7D32", anchor="end", halo=True)
arrow("M 800 612 C 842 612, 846 522, 876 508", "purple")
text(828, 584, "d ← 层堆叠", 10.5, "#6A1B9A", halo=True)

arrow("M 878 334 C 828 262, 832 186, 804 170", "orange", 2.4, dash="7 5")
text(884, 252, "p′ 反馈", 11, "#E65100", halo=True)
arrow("M 878 436 L 804 434", "orange", 2.4, dash="7 5")
text(876, 460, "g: α ← α·g", 11, "#E65100", anchor="end", halo=True)
arrow("M 878 542 C 828 542, 834 626, 804 618", "orange", 2.4, dash="7 5")
text(884, 600, "w 重标定", 11, "#E65100", halo=True)

# p′ 主分发: 顶部走廊 → 右侧竖井 → 融合右缘(避开专家池)
arrow("M 800 166 C 900 58, 1400 58, 1545 168 L 1545 796 L 1496 796", "blue", 2.6)
text(1052, 118, "p′ 调制后权重 → 加权专家输出", 12, "#1565C0", weight="700", halo=True)

# 专家池
_exp = box(1160, 150, 330, 330, "pink", "专家池 MoE · 10 = 5 类 × 2",
           ["y_i = SiTU(head(backbone(s_t)))",
            "MLP 200→64×2→1 · 或 Transformer 变体",
            "Stage1 独立训练后冻结"])
CUR["tag"] = "专家池 MoE · 10 = 5 类 × 2"
for i, (lab, fc) in enumerate([("趋势 Trend ×2", "#F8BBD0"), ("反转 Reversal ×2", "#F48FB1"),
                               ("波动 Volatility ×2", "#F06292"), ("事件 Event ×2", "#EC407A"),
                               ("动量 Momentum ×2", "#E91E63")]):
    rect(1174, 258 + i * 30, 302, 24, "专家池 MoE · 10 = 5 类 × 2", fc, "#F48FB1", 0.8, rx=6, opacity=0.55)
    text(1184, 274 + i * 30, lab, 11.5, "#880E4F", weight="600")
CUR["tag"] = ""
arrow("M 1325 482 L 1325 736", "pink", 2.4)
text(1338, 600, "y_i ∈ (−1,1)", 11, "#AD1457", halo=True)

# s_t 总线
arrow("M 390 483 L 390 712 L 1505 712 L 1505 318 L 1494 318", "gray", 2.8)
text(402, 732, "s_t 总线 → 全体专家共享输入(分发: 同一市场状态, 10 份个性化解读)", 11.5, "#546E7A", halo=True)

arrow("M 800 502 C 935 648, 1070 800, 1156 806", "green", 2.2)
text(856, 656, "retrieved o (记忆检索)", 11, "#2E7D32", halo=True)
arrow("M 800 660 C 950 682, 1050 834, 1156 836", "purple", 2.2)
text(1015, 824, "fused (深度融合)", 11, "#6A1B9A", halo=True)

# 融合与输出
box(1160, 740, 330, 120, "pink", "信号融合",
    ["signal = Σ_i p′_i · y_i(s_t)",
     "      + 0.1·mean(fused + retrieved)",
     "p′ : CDAP 调制后的路由权重"])
pill(1300, 872, "A6 0.1 系数硬编码(待修)", "red", 10)
arrow("M 1408 862 L 1408 901", "gray", 2.2)
box(1160, 905, 330, 72, "data", "输出 signal (T−1, N)",
    ["→ 回测: Rank IC · 净 Sharpe(5bp+1bp)"], lsize=11)

# 设计特点
text(40, 1032, "设计特点(公式级不变量)", 16, "#263238", weight="700")
cards = [
    ("① 三维联合空间 j = e⊙m⊙d",
     ["任一维低激活 ⇒ 整体调制受抑", "记忆不确定时无法扭曲路由", "K3 三孤岛 → DAFT 闭环(原创)"]),
    ("② 零扰动不变量 (logit 修复)",
     ["δ=0 ⇒ p′≡p · g≡0.5 · w≡均匀", "scale 零初始化 ⇒ 启动无扰动", "修复史: 概率空间 → logit 空间"]),
    ("③ KDA δ规则 + safe gate",
     ["M ← αM − βk⊗(Mk) + βk⊗v", "α∈(lb,1) 防「金鱼记忆」", "O(d_k·d_v) 定长 · 序列无关"]),
    ("④ SiTU 有界激活 + 温度退火",
     ["σ(x)⊙tanh(x) ∈ (−1,1)", "多空信号量纲可比 · 防漂移", "Stage2 温度 1.0→0.5 防熵塌缩"]),
    ("⑤ 负载均衡 KL (方案A)",
     ["L_bal = Σ 1/E·log((1/E)/frac)", "替代抵消的熵正则对(塌缩根因)", "balance 0.01 · 每 50 步 QB"]),
]
for i, (t, ls) in enumerate(cards):
    x = 40 + i * 304
    rect(x, 1048, 292, 130, t, "#FFFFFF", "#90A4AE", 1.4)
    CUR["tag"] = t
    text(x + 14, 1072, t, 12.5, "#37474F", weight="700")
    yy = 1094
    for ln in ls:
        text(x + 14, yy, ln, 10.8, "#546E7A")
        yy += 18
    CUR["tag"] = ""

# 训练与判定
text(40, 1226, "分阶段训练与最终判定", 16, "#263238", weight="700")
stages = [
    (40, 330, "blue", "Stage 1 · 专家独立训练",
     ["regime 子集 / 全量 · 各自早停", "(A1: 随机切分泄漏 → 待修时序切分)"]),
    (430, 360, "green", "Stage 2 · 路由+记忆+CDAP",
     ["δ=0.1 · 专家冻结 · 温度 1→0.5 · KL", "早停按 val (A2 已修 ✅)"]),
    (846, 330, "purple", "Stage 3 · 联合微调",
     ["全解冻 · δ=1.0 · lr=1e-5 · 专家×0.1", "温度 0.1 近离散路由"]),
]
for x, w, ck, t, ls in stages:
    box(x, 1242, w, 78, ck, t, ls, tsize=13, lsize=11, lh=17, pad=10)
arrow("M 374 1281 L 426 1281", "gray", 2.2)
arrow("M 794 1281 L 842 1281", "gray", 2.2)
rect(1286, 1242, 310, 78, "判定", "#37474F", "#263238", 1.8)
CUR["tag"] = "判定"
text(1298, 1266, "判定 · 🔴 NO-GO(架构)", 13.5, "#FFCDD2", weight="800")
text(1298, 1286, "DAFT IC≈0.032 vs Ridge 0.048~0.054", 11, "#ECEFF1")
text(1298, 1304, "转研究项目 · 负结果归档 + 论文线", 11, "#ECEFF1")
CUR["tag"] = ""
arrow("M 1178 1281 L 1282 1281", "gray", 2.2)

# 图例
leg = [("数据流", "#546E7A"), ("路由维", "#1565C0"), ("记忆维", "#2E7D32"),
       ("深度维", "#6A1B9A"), ("专家/融合", "#AD1457"), ("CDAP 调制(虚线=反馈)", "#E65100")]
lx = 40
for lab, cc in leg:
    A(f'<rect x="{lx}" y="1348" width="16" height="16" rx="4" fill="{cc}"/>')
    text(lx + 22, 1361, lab, 11.5, "#546E7A")
    lx += 30 + tw(lab, 11.5) + 34
pill(lx + 40, 1356, "红 = A级待修项", "red", 10.5)
pill(lx + 190, 1356, "绿 = 消融实证贡献", "ok", 10.5)

A('</g>')
OFF["dy"] = 0

# ════════════════════════ Part 3 · 具体案例 ════════════════════════
text(40, PART3_Y + 42, "Part 3 · 一个具体案例: 2025-06-20(周五) 的某只股票", 26, "#263238", weight="800")
text(40, PART3_Y + 72, "从「今天喂进什么数据」到「今天输出什么决定」 · 数字为示意值, 用于演示机制而非真实行情", 13, "#546E7A")

box(50, PART3_Y + 100, 400, 330, "data", "输入(全部自动下载, 无人手工填写)",
    ["来源: baostock · 前复权日线", "", "股票: 600519 贵州茅台(示例)", "今日收盘: 1480.0 元(比昨天 +1.2%)",
     "今日成交量: 比平时多 3 成", "近 20 日每天波动约 ±1.8%", "当日未涨停跌停 → 可以交易", "",
     "同一时刻, 全市场 300 只股票的", "同类数据一起喂入(截面比较用)"])
pill(250, PART3_Y + 452, "对应 Part 1 气泡 ①「市场日报」", "data", 10.5)

arrow(f"M 454 {PART3_Y+265} L 502 {PART3_Y+265}", "gray", 2.6)

box(506, PART3_Y + 100, 520, 330, "cream", "五步处理(对应 Part 1 气泡 ②③④⑤)",
    ["1. 变成 200 个数字: 例如「近 20 日累计涨幅 = +3.2%」、",
     "   「成交量相对常态 = +0.41」…… 全部机器可读。",
     "2. 10 位分析师各自打分: 趋势派 +0.31 · 动量派 +0.44 ·",
     "   反转派 −0.12 · 波动派 +0.05 · 事件派 +0.18 ……",
     "3. 组长看市况: 判断当前像「震荡市」→ 这次多听",
     "   反转派和波动派, 少听趋势派。",
     "4. 笔记本提醒: 3 周前出现过类似形态, 之后小涨 →",
     "   把各专家的意见轻微上调一点点。",
     "5. 圆桌合成最终听谁: 动量 0.32 / 事件 0.27 / 趋势 0.24",
     "   / 反转 0.17(权重相加 = 1, 不是简单平均)。"], lsize=11.8)

arrow(f"M 1030 {PART3_Y+265} L 1078 {PART3_Y+265}", "gray", 2.6)

_OT = "输出(当天唯一产物: 一张打分表)"
rect(1082, PART3_Y + 100, 468, 330, _OT, "#E8F5E9", "#2E7D32", 2, rx=10, opacity=0.6)
CUR["tag"] = _OT
text(1100, PART3_Y + 128, _OT, 14, "#1B5E20", weight="800")
outs = [
    ("本股今日信号 = +0.42", "在 −1(最看空) ~ +1(最看多) 之间"),
    ("全市场排名: 第 4 / 300", "按信号从高到低排序"),
    ("→ 进入「买入组」(前 20%)", "后 20% 做空, 中间不动"),
    ("次日实际走势: +0.8% ✓", "这次方向猜对了"),
    ("扣除成本 0.06% → 净 +0.74%", "买与卖各付一次费用"),
]
yy = PART3_Y + 158
for big, small in outs:
    text(1100, yy, big, 14.5, "#1B5E20", weight="700")
    text(1100, yy + 19, small, 11.5, "#546E7A")
    yy += 46
CUR["tag"] = ""

arrow(f"M 1316 {PART3_Y+434} L 1316 {PART3_Y+470}", "gray", 2.4)
rect(50, PART3_Y + 476, 1500, 96, "然后呢", "#FFFFFF", "#90A4AE", 1.6)
CUR["tag"] = "然后呢"
text(74, PART3_Y + 508, "然后呢?", 14, "#263238", weight="800")
text(160, PART3_Y + 502, "同样的流程在 5 年 × 约 1200 个交易日 × 300 只股票上每天重复一遍, 得到组合的长期成绩单:", 13, "#37474F")
text(160, PART3_Y + 526, "方向预测准确率仅比抛硬币略好(IC≈0.03 ≈ 51%), 扣除成本后整体未能跑赢一个简单的线性统计方法 ——", 13, "#B71C1C")
text(160, PART3_Y + 550, "这正是项目 2026-08-17 判定 🔴 NO-GO(架构)、转向研究归档的原因。诚实报告负结果, 与报告正结果同等重要。", 13, "#B71C1C")
CUR["tag"] = ""

text(40, PART3_Y + 612, "生成 2026-08-18(v2 布局 lint 通过) · 基于全量代码审读 · 配套纲领 docs/K3_GUIDANCE_2026-08-18.md · 修复线: A2✅(PR 16) → A1 → A3 → A4 → A5 → A6", 11, "#90A4AE")

A('</svg>')

# ════════════════════════ 布局 lint ════════════════════════
def inter(a, b):
    w = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    h = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return w * h if (w > 0 and h > 0) else 0.0

# L1 越画布 / L3 文字-文字 / L4 文字-异属框
for i, t in enumerate(TB):
    if t["x0"] < 2 or t["x1"] > W - 2 or t["y0"] < 2 or t["y1"] > H - 2:
        ISSUES.append(f"L1 越画布: {t['s'][:30]} bbox=({t['x0']:.0f},{t['y0']:.0f},{t['x1']:.0f},{t['y1']:.0f})")
    for t2 in TB[i + 1:]:
        if inter(t, t2) > 8:
            ISSUES.append(f"L3 文字重叠: 「{t['s'][:16]}」×「{t2['s'][:16]}」 {inter(t, t2):.0f}px²")
    for r in RC:
        if t["tag"] != r["tag"] and inter(t, r) > 8:
            ISSUES.append(f"L4 文字压框: 「{t['s'][:20]}」压[{r['tag'][:20]}] {inter(t, r):.0f}px²")

# L2 文字溢出所属框(同 tag + 框的 y 范围包含文字中心, 取最窄匹配)
for t in TB:
    if not t["tag"]:
        continue
    cand = [r for r in RC
            if r["tag"] == t["tag"] and r["y0"] <= (t["y0"] + t["y1"]) / 2 <= r["y1"]]
    if not cand:
        continue
    r = min(cand, key=lambda r: r["x1"] - r["x0"])
    if t["x0"] < r["x0"] - 1 or t["x1"] > r["x1"] + 1:
        ISSUES.append(f"L2 文字出框: 「{t['s'][:26]}」超 [{r['tag'][:18]}] 右缘 {t['x1'] - r['x1']:.0f}px")

# L5 框-框(药丸 vs 卡片)
for i, r in enumerate(RC):
    for r2 in RC[i + 1:]:
        if r["tag"] != r2["tag"] and inter(r, r2) > 6:
            ISSUES.append(f"L5 框重叠: [{r['tag'][:18]}]×[{r2['tag'][:18]}] {inter(r, r2):.0f}px²")

# L6 箭头穿文字(非 halo) / L7 箭头切框内部(距边>10px; 端点豁免)
for pts in AR:
    npp = len(pts)
    for i, (px, py) in enumerate(pts):
        if i < 2 or i >= npp - 2:      # 端点贴边属正常连接
            continue
        for t in TB:
            if not t["halo"] and t["x0"] - 1 < px < t["x1"] + 1 and t["y0"] - 1 < py < t["y1"] + 1:
                ISSUES.append(f"L6 箭头穿字: 「{t['s'][:20]}」 at ({px:.0f},{py:.0f})")
        for r in RC:
            if r["x0"] < px < r["x1"] and r["y0"] < py < r["y1"]:
                d = min(px - r["x0"], r["x1"] - px, py - r["y0"], r["y1"] - py)
                if d > 10:
                    ISSUES.append(f"L7 箭头切框: [{r['tag'][:20]}] 内 ({px:.0f},{py:.0f}) 深{d:.0f}px")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "docs", "assets", "architecture", "daft-architecture-overview.svg")
out = os.path.normpath(out)
os.makedirs(os.path.dirname(out), exist_ok=True)
if ISSUES:
    print(f"!! {len(ISSUES)} layout issues — SVG 未写出:")
    seen = set()
    for m in ISSUES:
        if m not in seen:
            seen.add(m)
            print("  -", m.encode("gbk", errors="replace").decode("gbk"))
    raise SystemExit(1)
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(P))
print("OK", out, f"{W}x{H}")
print(f"lint: 0 issues (texts={len(TB)} rects={len(RC)} arrows={len(AR)})")
