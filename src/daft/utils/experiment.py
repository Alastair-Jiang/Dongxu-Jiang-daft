"""实验产物唯一命名与 config hash (2026-08-16 新增)。

背景: 历史实验产物使用固定文件名(如 full_pipeline_oos_report.json),
后续实验会覆盖前序产物, 导致登记表"数字必须来自 outputs/*.json"的
硬规则无法执行(EXP-01 换手 0.03% 与 JSON 0.14% 不符、EXP-03 无产物)。

本模块提供:
- next_exp_path(): 产出 outputs/EXP-YYYYMMDD-NN-<prefix>.json 唯一路径
  (同日内按已有最大 NN 递增, 不覆盖任何历史产物)
- config_hash(): 配置字典的 sha256 前 8 位, 用于登记表"Config 快照"
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def config_hash(config: Dict, length: int = 8) -> str:
    """Canonical JSON hash of a config dict (stable, order-insensitive)."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def next_exp_path(out_dir: Path, prefix: str) -> Path:
    """Unique experiment output path: EXP-YYYYMMDD-NN-<prefix>.json.

    扫描同日前缀的现有产物, 取最大 NN + 1, 保证不覆盖历史产物。
    """
    today = datetime.now().strftime("%Y%m%d")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_nn = 0
    for p in out_dir.glob(f"EXP-{today}-*-{prefix}.json"):
        stem = p.name[len(f"EXP-{today}-"):]
        nn_str = stem.split("-", 1)[0]
        if nn_str.isdigit():
            max_nn = max(max_nn, int(nn_str))

    return out_dir / f"EXP-{today}-{max_nn + 1:02d}-{prefix}.json"
