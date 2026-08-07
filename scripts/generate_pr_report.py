#!/usr/bin/env python3
"""
PR 预更新报告自动生成器
=======================
由 GitHub Actions 触发: 每次 PR 提交/更新时自动生成三页 PDF 报告
本地也可手动运行: python scripts/generate_pr_report.py

输出: PR-Report.pdf (三页)
  第1页 - 更新说明 (PR 元数据 + 变更摘要 + 模块影响分析)
  第2页 - 架构可视化 (Mermaid 流程图, 标注受影响模块)
  第3页 - 代码 Diff (git diff 原始输出)
"""
import subprocess
import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Fix Windows stdout encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# 配置
# ============================================================
@dataclass
class Config:
    pr_number: str = os.environ.get("PR_NUMBER", "?")
    pr_title: str = os.environ.get("PR_TITLE", "Untitled PR")
    pr_author: str = os.environ.get("PR_AUTHOR", "unknown")
    pr_created: str = os.environ.get("PR_CREATED", datetime.now().isoformat())
    base_ref: str = os.environ.get("BASE_REF", "main")
    head_ref: str = os.environ.get("HEAD_REF", "feature")
    repo: str = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
    output: str = "PR-Report.pdf"
    work_dir: str = "."


# ============================================================
# Git 数据获取
# ============================================================

def run(cmd: str, cwd: str = ".") -> str:
    """Run a shell command and return stdout (UTF-8 safe, cross-platform)."""
    # On Windows, force using bash to avoid GBK encoding issues with cmd.exe
    if sys.platform == "win32":
        bash_paths = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
        bash = None
        for bp in bash_paths:
            if os.path.exists(bp):
                bash = bp
                break
        if bash:
            r = subprocess.run(
                [bash, "-c", cmd], cwd=cwd, capture_output=True,
                text=True, encoding="utf-8", errors="replace"
            )
        else:
            r = subprocess.run(
                cmd, shell=True, cwd=cwd, capture_output=True,
                text=True, encoding="utf-8", errors="replace"
            )
    else:
        r = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace"
        )
    return r.stdout.strip()


def get_diff_stats(cfg: Config) -> str:
    """git diff --stat"""
    # 使用三点语法 (...), 表示从 merge-base 到 head 的差异
    base = f"origin/{cfg.base_ref}" if not cfg.base_ref.startswith("origin/") else cfg.base_ref
    head = f"origin/{cfg.head_ref}" if not cfg.head_ref.startswith("origin/") else cfg.head_ref
    out = run(f"git diff {base}...{head} --stat 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff HEAD~1 --stat 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff --stat 2>/dev/null", cfg.work_dir)
    return out


def get_diff_full(cfg: Config) -> str:
    """git diff 完整输出"""
    base = f"origin/{cfg.base_ref}" if not cfg.base_ref.startswith("origin/") else cfg.base_ref
    head = f"origin/{cfg.head_ref}" if not cfg.head_ref.startswith("origin/") else cfg.head_ref
    out = run(f"git diff {base}...{head} 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff HEAD~1 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff 2>/dev/null", cfg.work_dir)
    return out


def get_changed_files(cfg: Config) -> List[str]:
    """获取变更文件列表"""
    base = f"origin/{cfg.base_ref}" if not cfg.base_ref.startswith("origin/") else cfg.base_ref
    head = f"origin/{cfg.head_ref}" if not cfg.head_ref.startswith("origin/") else cfg.head_ref
    out = run(f"git diff {base}...{head} --name-only 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff HEAD~1 --name-only 2>/dev/null", cfg.work_dir)
    if not out:
        out = run("git diff --name-only 2>/dev/null", cfg.work_dir)
    return [f for f in out.split("\n") if f.strip()]


def get_diff_staged() -> str:
    """获取当前的 diff (works for CI checkout)"""
    # 在 CI 中, PR 已经被 checkout, 使用 HEAD 的 diff
    out = run("git diff HEAD~1 --stat 2>/dev/null")
    if not out:
        out = run("git diff --cached --stat 2>/dev/null")
    if not out:
        out = run("git log --oneline -5")
    return out


# ============================================================
# 模块影响分析
# ============================================================

# DAFT 项目模块映射: 文件路径模式 → 模块名称
MODULE_MAP = [
    (r"src/daft/models/ensemble\.py", "ExpertEnsemble (核心调度)"),
    (r"src/daft/models/memory\.py", "KDA Market Memory"),
    (r"src/daft/models/router\.py", "Regime Router"),
    (r"src/daft/models/cross_dim_attn\.py", "CDAP (交叉注意力)"),
    (r"src/daft/models/hardening\.py", "Hardening Engine (硬化)"),
    (r"src/daft/models/experts/", "策略专家 (Trend/Reversal/Volatility/Event)"),
    (r"src/daft/features/", "特征工程模块"),
    (r"src/daft/training/", "训练管线"),
    (r"src/daft/backtest/", "回测引擎"),
    (r"src/daft/portfolio/", "组合优化"),
    (r"checkpoint/", "开发检查点文档"),
    (r"configs/", "配置文件"),
    (r"tests/", "测试模块"),
]


def analyze_impact(files: List[str]) -> List[Tuple[str, str]]:
    """分析受影响模块"""
    affected = []
    seen = set()
    for f in files:
        for pattern, module_name in MODULE_MAP:
            if re.search(pattern, f) and module_name not in seen:
                affected.append((f, module_name))
                seen.add(module_name)
    return affected


# ============================================================
# Mermaid 架构图生成
# ============================================================

ARCH_DIAGRAM_ORIGINAL = """graph TB
    A["Market Data"] --> B["Feature Extraction s_t"]
    B --> C["3-Layer Depth"] & D["Regime Router"] & E["KDA Memory"]
    C & D & E --> F["4 Experts"]
    F --> CDAP["CDAP Protocol"]
    C & D & E --> CDAP
    CDAP --> R["routing_mod"] & DW["depth_weights"] & MG["memory_gate"]
    R & DW --> H["Weighted Fusion"]
    H --> I["Trading Signal"]
    style A fill:#e8eaf6,stroke:#283593
    style B fill:#e8eaf6,stroke:#283593
    style I fill:#e8eaf6,stroke:#283593
    style CDAP fill:#e0f7fa,stroke:#00695c
    style H fill:#fce4ec,stroke:#c62828
    style F fill:#e8f5e9,stroke:#2e7d32"""


def generate_architecture_diagram(affected_modules: List[Tuple[str, str]]) -> str:
    """根据受影响的模块生成带高亮的 Mermaid 图"""
    if not affected_modules:
        return ARCH_DIAGRAM_ORIGINAL

    highlight_colors = [
        ("#FFEB3B", "#F57F17"),  # yellow highlight
        ("#FF9800", "#E65100"),  # orange highlight
        ("#FF5722", "#BF360C"),  # deep orange
    ]

    lines = ["graph TB"]
    lines.append('    A["Market Data"] --> B["Feature Extraction s_t"]')
    lines.append('    B --> C["3-Layer Depth"] & D["Regime Router"] & E["KDA Memory"]')
    lines.append('    C & D & E --> F["4 Experts"]')
    lines.append('    F --> CDAP["CDAP Protocol"]')
    lines.append('    C & D & E --> CDAP')
    lines.append('    CDAP --> R["routing_mod"] & DW["depth_weights"] & MG["memory_gate"]')
    lines.append('    R & DW --> H["Weighted Fusion"]')
    lines.append('    H --> I["Trading Signal"]')

    # Standard styles
    lines.append('    style A fill:#e8eaf6,stroke:#283593')
    lines.append('    style B fill:#e8eaf6,stroke:#283593')
    lines.append('    style I fill:#e8eaf6,stroke:#283593')
    lines.append('    style CDAP fill:#e0f7fa,stroke:#00695c')
    lines.append('    style H fill:#fce4ec,stroke:#c62828')
    lines.append('    style F fill:#e8f5e9,stroke:#2e7d32')

    # Highlight affected nodes
    node_map = {
        "ExpertEnsemble": "H", "KDA Market Memory": "E",
        "Regime Router": "D", "CDAP": "CDAP",
        "Hardening Engine": "H", "策略专家": "F",
        "特征工程": "B", "训练管线": "F",
        "回测引擎": "I", "组合优化": "I",
    }

    for i, (file_path, module_name) in enumerate(affected_modules[:5]):
        for key, node_id in node_map.items():
            if key.lower() in module_name.lower() or key.lower() in file_path.lower():
                bg, border = highlight_colors[i % len(highlight_colors)]
                lines.append(f'    style {node_id} fill:{bg},stroke:{border},stroke-width:3px')
                break

    # Add affected module legend
    for i, (file_path, module_name) in enumerate(affected_modules[:5]):
        bg, border = highlight_colors[i % len(highlight_colors)]
        short_name = module_name[:40]
        safe_name = f"LEGEND{i}"
        lines.append(f'    {safe_name}["📌 {short_name}"]')
        lines.append(f'    style {safe_name} fill:{bg},stroke:{border},stroke-width:2px')

    return "\n".join(lines)


# ============================================================
# HTML 报告生成
# ============================================================

def _html_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', sans-serif; color: #212121; line-height: 1.7; font-size: 13px; }
.cover { text-align: center; padding-top: 60px; }
.cover h1 { font-size: 26px; color: #1a1a2e; margin-bottom: 6px; }
.cover .subtitle { font-size: 16px; color: #555; margin-bottom: 6px; }
.cover .meta { font-size: 12px; color: #888; margin-bottom: 24px; }
.cover .summary-box { display: inline-block; text-align: left; background: #f5f5f5; padding: 14px 20px; border-radius: 6px; font-size: 12px; line-height: 2; }
h2 { color: #1a1a2e; border-bottom: 2px solid #1565c0; padding-bottom: 4px; margin-top: 22px; font-size: 17px; }
h3 { color: #333; margin-top: 16px; font-size: 14px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }
th { background: #1565c0; color: #fff; padding: 7px 9px; text-align: left; }
td { padding: 6px 9px; border-bottom: 1px solid #e0e0e0; }
tr:nth-child(even) { background: #fafafa; }
.diff-box { background: #1e1e1e; color: #d4d4d4; padding: 10px 14px; border-radius: 4px; font-family: 'Consolas','Courier New',monospace; font-size: 5.5px; line-height: 1.45; white-space: pre; overflow-x: auto; max-height: 480px; overflow-y: auto; }
.diff-add { color: #4ec9b0; }
.diff-del { color: #f44747; }
.diff-hdr { color: #569cd6; }
.diff-meta { color: #888; }
.page-break { page-break-after: always; }
.img-full { width: 100%; max-height: 90vh; object-fit: contain; margin: 10px auto; display: block; }
.impact-tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 700; margin: 2px; }
.impact-high { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; }
.impact-mid { background: #fff3e0; color: #e65100; border: 1px solid #ffcc80; }
.impact-low { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; }
.file-list { font-size: 12px; }
.file-list li { padding: 3px 0; }
.file-list .path { font-family: monospace; color: #1565c0; }
.file-list .change { font-size: 11px; color: #888; }
"""


def build_html(cfg: Config, diff_full: str, diff_stat: str,
               changed_files: List[str], impact: List[Tuple[str, str]],
               arch_svg_path: str = "") -> str:
    """构建完整的三页 HTML"""

    # --- 变更统计 ---
    total_added = 0
    total_removed = 0
    for line in diff_stat.split("\n"):
        m = re.search(r'(\d+) insertion', line)
        if m: total_added += int(m.group(1))
        m = re.search(r'(\d+) deletion', line)
        if m: total_removed += int(m.group(1))

    file_count = len(changed_files)

    # --- 影响级别 ---
    impact_level = "mid"
    if any("ensemble.py" in f or "memory.py" in f for f in changed_files):
        impact_level = "high"
    elif file_count <= 2 and all(f.endswith(".md") for f in changed_files):
        impact_level = "low"
    impact_label = {"high": "🔴 高影响 (核心模块变更)", "mid": "🟡 中等影响", "low": "🟢 低影响 (文档/配置变更)"}

    # --- 构建页面1: 更新说明 ---
    date_str = cfg.pr_created[:10] if cfg.pr_created else datetime.now().strftime("%Y-%m-%d")

    affected_rows = ""
    for file_path, module_name in impact[:10]:
        affected_rows += f"<tr><td><code>{_html_escape(file_path)}</code></td><td>{_html_escape(module_name)}</td></tr>"
    if not affected_rows:
        affected_rows = "<tr><td colspan='2' style='color:#888;text-align:center'>无模块匹配 (新文件或非核心路径)</td></tr>"

    file_rows = ""
    for f in changed_files[:30]:
        ext = Path(f).suffix
        icon = {"py": "🐍", "md": "📝", "yaml": "⚙️", "yml": "⚙️", "json": "📋", "csv": "📊"}.get(ext.lstrip("."), "📄")
        file_rows += f"<tr><td>{icon} <code>{_html_escape(f)}</code></td><td>{ext}</td></tr>"

    page1 = f"""<div class="cover">
  <h1>DAFT PR #{cfg.pr_number} — 预更新报告</h1>
  <div class="subtitle">{_html_escape(cfg.pr_title)}</div>
  <div class="meta">
    仓库: {_html_escape(cfg.repo)} &nbsp;|&nbsp;
    作者: @{_html_escape(cfg.pr_author)} &nbsp;|&nbsp;
    日期: {date_str} &nbsp;|&nbsp;
    分支: {_html_escape(cfg.head_ref)} → {_html_escape(cfg.base_ref)}
  </div>
  <div class="summary-box">
    <strong>变更概要</strong><br>
    📄 {file_count} 个文件 &nbsp;
    <span style="color:#2e7d32">+{total_added}</span> /
    <span style="color:#d32f2f">-{total_removed}</span> 行<br>
    影响级别: <span class="impact-tag impact-{impact_level}">{impact_label[impact_level]}</span>
  </div>
</div>

<h2>一、受影响模块</h2>
<table><tr><th>文件</th><th>模块</th></tr>{affected_rows}</table>

<h2>二、文件变更清单</h2>
<table><tr><th>文件路径</th><th>类型</th></tr>{file_rows}</table>

<h2>三、Diff 统计</h2>
<pre style="background:#f5f5f5;padding:10px;border-radius:4px;font-size:11px;overflow-x:auto;">{_html_escape(diff_stat)}</pre>"""

    # --- 构建页面2: 架构可视化 ---
    page2 = '<h2 style="text-align:center;">DAFT 架构图 — 受影响模块高亮</h2>'
    page2 += '<p style="text-align:center;color:#888;font-size:12px;">黄色/橙色标注 = 本次 PR 修改的模块</p>'
    if arch_svg_path and os.path.exists(arch_svg_path):
        # 嵌入 SVG
        svg_content = Path(arch_svg_path).read_text(encoding="utf-8")
        page2 += f'<div style="text-align:center;padding:10px;">{svg_content}</div>'
    else:
        page2 += '<p style="text-align:center;color:#888;padding:40px;">架构图生成中...请查看 Artifact 中的 PDF</p>'

    if impact:
        page2 += '<h3>受影响组件详情</h3><ul>'
        for file_path, module_name in impact[:10]:
            page2 += f'<li><code>{_html_escape(file_path)}</code> → <strong>{_html_escape(module_name)}</strong></li>'
        page2 += '</ul>'

    # --- 构建页面3: 代码 Diff ---
    # 语法高亮处理
    diff_html = _html_escape(diff_full)

    # 简单着色 (在 HTML 中无法做复杂 regex, 用 CSS class 标记)
    colored_lines = []
    for line in diff_html.split("\n"):
        if line.startswith("+"):
            colored_lines.append(f'<span class="diff-add">{line}</span>')
        elif line.startswith("-"):
            colored_lines.append(f'<span class="diff-del">{line}</span>')
        elif line.startswith("@@"):
            colored_lines.append(f'<span class="diff-hdr">{line}</span>')
        elif line.startswith("diff ") or line.startswith("index ") or line.startswith("---") or line.startswith("+++"):
            colored_lines.append(f'<span class="diff-meta">{line}</span>')
        else:
            colored_lines.append(line)

    diff_colored = "\n".join(colored_lines)

    page3 = f"""<h2>完整代码 Diff</h2>
<p style="font-size:12px;color:#888;">{_html_escape(cfg.head_ref)} → {_html_escape(cfg.base_ref)} &nbsp;|&nbsp; {file_count} 文件, +{total_added}/-{total_removed} 行</p>
<div class="diff-box">{diff_colored}</div>"""

    # --- 组装完整 HTML ---
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><style>{CSS}</style></head>
<body>
{page1}
<div class="page-break"></div>
{page2}
<div class="page-break"></div>
{page3}
</body>
</html>"""
    return html


# ============================================================
# PDF 渲染
# ============================================================

def find_chrome() -> Optional[str]:
    """Find Chrome/Chromium executable"""
    # Windows paths first (check existence)
    win_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    # Linux/macOS commands
    nix_cmds = ["google-chrome", "chromium-browser", "chromium",
                "/usr/bin/google-chrome", "/usr/bin/chromium-browser"]

    for p in win_paths:
        if os.path.exists(p):
            return p
    for c in nix_cmds:
        r = subprocess.run(["which", c], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    return None


def render_pdf(html_path: str, pdf_path: str) -> bool:
    """使用 Chrome headless 渲染 HTML → PDF"""
    chrome = find_chrome()
    if not chrome:
        print("[WARN] Chrome not found, trying puppeteer fallback...")
        return _render_via_puppeteer(html_path, pdf_path)

    abs_html = os.path.abspath(html_path).replace("\\", "/")
    abs_pdf = os.path.abspath(pdf_path).replace("\\", "/")
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf}",
        f"file:///{abs_html}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode == 0 and os.path.exists(pdf_path):
        print(f"[OK] PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
        return True
    else:
        print(f"[FAIL] Chrome render error: {r.stderr[:200] if r.stderr else 'none'}")
        return False


def _render_via_puppeteer(html_path: str, pdf_path: str) -> bool:
    """Fallback: use Node.js puppeteer to render"""
    js_code = f"""
const puppeteer = require('puppeteer');
(async () => {{
    const browser = await puppeteer.launch({{ headless: 'new', args: ['--no-sandbox'] }});
    const page = await browser.newPage();
    await page.goto('file:///{os.path.abspath(html_path).replace(chr(92), "/")}', {{ waitUntil: 'networkidle0' }});
    await page.pdf({{ path: '{pdf_path}', format: 'A3', printBackground: true }});
    await browser.close();
}})();
"""
    r = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, timeout=60)
    if r.returncode == 0:
        print(f"[OK] PDF (puppeteer): {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
        return True
    print(f"[FAIL] Puppeteer fallback: {r.stderr[:200] if r.stderr else 'none'}")
    return False


# ============================================================
# 主流程
# ============================================================

def generate_report(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    print(f"[1/4] Generating PR #{cfg.pr_number} report...")
    print(f"   Title: {cfg.pr_title}")
    print(f"   Branch: {cfg.head_ref} -> {cfg.base_ref}")

    # 1. Get git data
    print("\n[2/4] Fetching git diff...")
    diff_full = get_diff_full(cfg)
    diff_stat = get_diff_stats(cfg)
    changed_files = get_changed_files(cfg)

    if not changed_files:
        # CI fallback
        changed_files = get_diff_staged().split("\n")
        diff_full = get_diff_staged()
        diff_stat = get_diff_staged()

    print(f"   Changed files: {len(changed_files)}")
    for f in changed_files[:10]:
        print(f"     - {f}")

    # 2. Impact analysis
    print("\n[3/4] Analyzing module impact...")
    impact = analyze_impact(changed_files)
    for file_path, module_name in impact:
        print(f"   -> {module_name} : {file_path}")

    # 3. Generate HTML
    print("\n[4/4] Generating HTML report...")
    html = build_html(cfg, diff_full, diff_stat, changed_files, impact)
    html_path = os.path.join(cfg.work_dir, "PR-Report.html")
    Path(html_path).write_text(html, encoding="utf-8")
    print(f"   HTML: {html_path}")

    # 4. Render PDF
    print("\n[5/5] Rendering PDF...")
    pdf_path = os.path.join(cfg.work_dir, cfg.output)
    success = render_pdf(html_path, pdf_path)

    if not success:
        print("\n[WARN] PDF rendering failed, HTML saved. Creating placeholder...")
        # Fallback placeholder PDF
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (1240, 1754), 'white')
            d = ImageDraw.Draw(img)
            d.text((100, 100), f"PR #{cfg.pr_number} Report", fill='black')
            d.text((100, 140), cfg.pr_title, fill='#555')
            d.text((100, 180), "PDF rendering failed in CI - see HTML artifact instead", fill='#888')
            img.save(pdf_path, 'PDF')
            print(f"   [WARN] Placeholder PDF: {pdf_path}")
        except Exception as e:
            print(f"   [ERROR] Placeholder failed: {e}")

    # 5. Result
    print(f"\n{'='*60}")
    if os.path.exists(pdf_path):
        print(f"[OK] Report: {pdf_path} ({os.path.getsize(pdf_path)/1024:.1f} KB)")
    else:
        print(f"[FAIL] Report generation failed")
    print(f"{'='*60}")

    return pdf_path if os.path.exists(pdf_path) else None


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    cfg = Config()
    # 允许命令行覆盖
    if len(sys.argv) > 1:
        cfg.pr_number = sys.argv[1]
    if len(sys.argv) > 2:
        cfg.pr_title = sys.argv[2]

    generate_report(cfg)
