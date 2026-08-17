"""EXP-20260809-02(重写): 信号平滑对比 — DAFT OOS checkpoint 复用 + EMA 平滑扫描。

目的: 验证信号 EMA 平滑能否压低换手、改善净 Sharpe(Kimi K3 评审建议)。

2026-08-16 重写要点:
  1. 对齐统一 k→k+1: signals[t] 预测 p[t+1]-p[t](与 run_full_pipeline_oos 一致);
  2. λ 在 **val 段**选择(净 Sharpe 最优), test 段仅用于报告——修复
     "λ 在 test 上调参"的样本外污染;
  3. 每个 λ 的 IC 用平滑后的信号计算(旧版各 λ 的 IC 恒同值);
  4. 产物用唯一文件名(EXP-YYYYMMDD-NN-*.json), 不再覆盖历史报告。

用法: python scripts/run_smoothing_ablation.py [--stocks 30] [--lambdas 0,0.3,0.5,0.7]
前置: checkpoints/oos/ 必须存在(EXP-20260809-01 产物; 若不存在请先跑
      python scripts/run_full_pipeline_oos.py)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from daft.data.adapters.baostock_adapter import BaostockAdapter
from daft.features.regime_features import RegimeFeatureExtractor
from daft.models.factory import build_model
from daft.training.joint_trainer import JointTrainer
from daft.backtest.engine import BacktestEngine
from daft.utils.metrics import rank_info_coefficient, ic_summary, hit_rate
from daft.utils.experiment import config_hash, next_exp_path

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def build_ablation_model():
    """推理用: 10 专家 + 温度 0.1(与 EXP-20260809-01 产物 checkpoint 对齐)。"""
    return build_model(
        cdap_strength=1.0, router_temperature=0.1, noisy_gating_std=0.0,
    )


def ema_smooth(signals: torch.Tensor, lam: float) -> torch.Tensor:
    """因果 EMA 平滑: s'_t = (1-λ)·s_t + λ·s'_{t-1}(只用过去, 无 look-ahead)。"""
    if lam <= 0:
        return signals
    smoothed = torch.zeros_like(signals)
    smoothed[0] = signals[0]
    for t in range(1, signals.shape[0]):
        smoothed[t] = (1 - lam) * signals[t] + lam * smoothed[t - 1]
    return smoothed


def evaluate(engine_cfg: dict, signals_seg: torch.Tensor, prices_seg: torch.Tensor,
             mask_seg: torch.Tensor, lam: float):
    """对一个信号段做平滑+回测+IC, 返回指标字典。"""
    smoothed = ema_smooth(signals_seg, lam)
    engine = BacktestEngine(engine_cfg)  # signal_smoothing 已在脚本内应用
    bt = engine.run(smoothed, prices_seg, mask=mask_seg)

    log_pt = torch.log(prices_seg.clamp(min=1e-8))
    returns = log_pt[1:] - log_pt[:-1]
    ic_series = rank_info_coefficient(smoothed[:-1], returns, mask_seg[1:],
                                      per_timestep=True)
    ic = ic_summary(ic_series)
    return {
        "sharpe": bt["sharpe_ratio"],
        "annual_return": bt["annual_return"],
        "max_drawdown": bt["max_drawdown"],
        "turnover": bt["turnover"],
        "ic": ic["ic_mean"], "ic_t": ic["ic_t_stat"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--lambdas", default="0,0.3,0.5,0.7")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--universe", default="hs300", choices=["hs300", "sample"])
    args = parser.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",")]

    t0 = time.time()
    print("=== EXP-20260809-02(v2): 信号平滑对比 (λ 在 val 选) ===")

    panel = BaostockAdapter({"start_date":args.start,"end_date":args.end,
                             "frequency":"d","n_stocks":args.stocks,
                             "universe":args.universe,"adjust":"2"}).load()
    T, N, F = panel.shape
    t_train_end, t_val_end = int(T*0.6), int(T*0.8)
    print(f"Panel: (T={T}, N={N})  train:[0,{t_train_end}) val:[{t_train_end},{t_val_end}) test:[{t_val_end},{T})")

    model, layer_proj = build_ablation_model()
    ckpt_dir = PROJECT_ROOT/"checkpoints"/"oos"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"{ckpt_dir} 不存在 — 先运行 scripts/run_full_pipeline_oos.py "
            "(EXP-20260809-01) 生成 checkpoint。"
        )
    JointTrainer.load_checkpoints(model, layer_proj, str(ckpt_dir))
    model.eval(); layer_proj.eval()
    print("Checkpoints 已加载 (EXP-20260809-01 产物)")

    # 标准化(仅 train)
    ext = RegimeFeatureExtractor(n_base_factors=50, output_dim=200)
    with torch.no_grad():
        s_tr = ext(panel.slice_time(0, t_train_end))
    s_tr = torch.nan_to_num(s_tr, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6,1e6)
    mu = s_tr.reshape(-1,200).mean(0,keepdim=True)
    sd = s_tr.reshape(-1,200).std(0,keepdim=True).clamp(min=1e-4)

    # 全量特征 + 因果信号生成
    print("生成信号(逐时步,记忆因果推进)...")
    with torch.no_grad():
        s_all = ext(panel)
    s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6,1e6)
    s_all = ((s_all - mu) / sd).clamp(-10.0, 10.0)

    device = torch.device("cpu")
    model.memory.reset_state(1, device)
    signals = torch.zeros(T-1, N)
    with torch.no_grad():
        for t in range(T-1):
            s_b = s_all[t].to(device)
            if model.memory.M is None or model.memory.M.size(0) != N:
                model.memory.reset_state(N, device)
            l0 = layer_proj["l0"](s_b); l1 = layer_proj["l1"](s_b); l2 = layer_proj["l2"](s_b)
            out = model(s_b, [l0,l1,l2], mode="inference")
            signals[t] = out["signal"].squeeze(-1).cpu()
            model.memory.detach_state()
    print(f"信号: {signals.shape}")

    # ── 段切片(对齐 k→k+1) ──
    # engine.run 要求 len(signals)==len(prices): 用 signals[:-1] 与
    # returns(log_p[1:]-log_p[:-1]) 配对。
    # val 段: signals_full[t_train_end:t_val_end] 恰有 t_val_end-t_train_end 行,
    #         与 prices 同长(最后一行信号被 engine 丢弃)。
    # test 段: signals_full 无 T-1 行, 需补一行哑元与 prices 对齐。
    signals_val = signals[t_train_end:t_val_end]
    prices_val = panel.values[t_train_end:t_val_end, :, 3]
    mask_val = panel.mask[t_train_end:t_val_end]

    signals_test = torch.cat([signals[t_val_end:], torch.zeros(1, N)], dim=0)
    prices_test = panel.values[t_val_end:, :, 3]
    mask_test = panel.mask[t_val_end:]

    engine_cfg = {
        "transaction_cost_bps": 5.0, "slippage_bps": 1.0,
        "top_quantile": 0.2, "long_only": False,
    }

    # 每个 λ 先跑 val(选参), 再跑 test(只报告)
    val_results, test_results = {}, {}
    for lam in lambdas:
        val_results[lam] = evaluate(engine_cfg, signals_val, prices_val, mask_val, lam)
        test_results[lam] = evaluate(engine_cfg, signals_test, prices_test, mask_test, lam)
        vr, tr = val_results[lam], test_results[lam]
        print(f"\n  λ={lam}: VAL  Sharpe={vr['sharpe']:+.4f} Turnover={vr['turnover']:.4f} IC={vr['ic']:+.4f}")
        print(f"         TEST Sharpe={tr['sharpe']:+.4f} Turnover={tr['turnover']:.4f} IC={tr['ic']:+.4f} t={tr['ic_t']:+.2f}")

    # λ* 由 val 净 Sharpe 决定(并列时取更小 λ = 更接近原始信号)
    lam_star = max(lambdas, key=lambda l: (val_results[l]["sharpe"], -l))
    print(f"\n  λ* = {lam_star} (由 val 净 Sharpe 选择)")

    report = {
        "experiment": "EXP-20260809-02-v2",
        "alignment": "k→k+1",
        "lambda_selection": "val 段净 Sharpe 最优 (test 仅报告)",
        "ckpt_dir": str(ckpt_dir),
        "stocks": N, "lambdas": lambdas, "lambda_star": lam_star,
        "engine_cfg": engine_cfg,
        "config_hash": config_hash({"stocks": N, "lambdas": lambdas, **engine_cfg}),
        "val": {str(k): v for k, v in val_results.items()},
        "test": {str(k): v for k, v in test_results.items()},
        "time_seconds": round(time.time()-t0, 1),
    }
    out_path = next_exp_path(OUTPUT_DIR, "daft-smoothing-ablation")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n    报告 → {out_path}")


if __name__ == "__main__":
    main()
