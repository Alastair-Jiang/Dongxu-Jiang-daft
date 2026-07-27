#!/usr/bin/env python3
"""
流程图 PDF 生成器 (基于 Mermaid + Chrome)
==========================================

依赖: @mermaid-js/mermaid-cli (mmdc) + Chrome/Edge 浏览器

安装 (一次性):
    npm install -g @mermaid-js/mermaid-cli

使用:
    python flowchart_to_pdf.py diagram.mmd -o output.pdf
    python flowchart_to_pdf.py diagram.mmd -o output.pdf -s 3   # 3x 缩放
    python flowchart_to_pdf.py diagram.mmd -o output.png -f png
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Optional


# Chrome 路径 (Windows)
_CHROME_PATHS = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
]


def _find_chrome() -> Optional[str]:
    for p in _CHROME_PATHS:
        if os.path.exists(p):
            return p
    return None


def mermaid_to_pdf(
    input_file: str,
    output_file: str,
    scale: int = 2,
    background: str = "transparent",
    theme: str = "base",
) -> bool:
    """将 Mermaid .mmd 文件渲染为 PDF。

    Parameters
    ----------
    input_file : str
        输入 .mmd 文件路径
    output_file : str
        输出 PDF 文件路径
    scale : int
        渲染缩放倍率 (1-4), 越大越清晰
    background : str
        背景色 ("transparent" / "white" / "#RRGGBB")
    theme : str
        Mermaid 主题 ("default" / "neutral" / "dark" / "forest" / "base")

    Returns
    -------
    bool
    """
    chrome = _find_chrome()
    if not chrome:
        print("❌ 找不到 Chrome/Edge 浏览器")
        return False

    # 设置 Puppeteer 执行路径
    env = os.environ.copy()
    env["PUPPETEER_EXECUTABLE_PATH"] = chrome

    cmd = [
        "mmdc",
        "-i", str(input_file),
        "-o", str(output_file),
        "-b", background,
        "-s", str(scale),
        "-t", theme,
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode == 0:
        size_kb = os.path.getsize(output_file) / 1024
        print(f"✅ 流程图已生成: {output_file} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"❌ 渲染失败:\n{result.stderr}")
        return False


def mermaid_template(title: str, direction: str = "TB") -> str:
    """生成 Mermaid flowchart 模板。

    Parameters
    ----------
    title : str
        图表标题
    direction : str
        流向: "TB" (上到下), "LR" (左到右), "BT" (下到上), "RL" (右到左)
    """
    return f"""%%{{init: {{'theme': 'base', 'themeVariables': {{
    'fontSize': '16px',
    'fontFamily': 'Microsoft YaHei, SimHei, sans-serif'
}}}}}}%%
graph {direction}
    %% {title}

    A[Start] --> B[Process]
    B --> C[End]

    classDef defaultStyle fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    class A,B,C defaultStyle
"""


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mermaid 流程图 → PDF/PNG 渲染器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python flowchart_to_pdf.py diagram.mmd -o output.pdf
  python flowchart_to_pdf.py diagram.mmd -o output.png -f png -s 3
  python flowchart_to_pdf.py --template "My Chart" -o quick.pdf
        """,
    )
    parser.add_argument("input", nargs="?", help="输入 .mmd 文件路径")
    parser.add_argument("-o", "--output", default="output.pdf", help="输出文件路径")
    parser.add_argument("-s", "--scale", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("-b", "--background", default="transparent")
    parser.add_argument("-t", "--theme", default="base",
                        choices=["default", "neutral", "dark", "forest", "base"])
    parser.add_argument("--template", help="生成 Mermaid 模板文件")
    parser.add_argument("--check", action="store_true", help="检查环境是否就绪")

    args = parser.parse_args()

    if args.check:
        chrome = _find_chrome()
        print(f"Chrome: {'✅ ' + chrome if chrome else '❌ 未找到'}")
        mmdc_path = shutil.which("mmdc") or shutil.which("mmdc.cmd")
        if mmdc_path:
            result = subprocess.run([mmdc_path, "--version"], capture_output=True, text=True)
            print(f"mmdc: ✅ {result.stdout.strip()}")
        else:
            print("mmdc: ❌ 未找到 (请确保 npm install -g @mermaid-js/mermaid-cli)")
        print("\n安装 mmdc: npm install -g @mermaid-js/mermaid-cli")
        sys.exit(0)

    if args.template:
        content = mermaid_template(args.template)
        out_path = args.output.replace(".pdf", ".mmd")
        Path(out_path).write_text(content, encoding="utf-8")
        print(f"✅ 模板已生成: {out_path}")
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    mermaid_to_pdf(args.input, args.output, args.scale, args.background, args.theme)
