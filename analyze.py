#!/usr/bin/env python3
"""Sprint 09B diagnostics: weights, per-window bootstraps, side concentration, sign-flip null, training curves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import parser, require_columns, resolve  # noqa: E402
from adjoint_policy import GEOM, OWN, PARTNER, exact_replay, fit_scales, prepare, to_windows, train_policy  # noqa: E402

SEED = 7
EXP: Path
FEATS_PER_PAIR = [
    ("own", 0, "ZERO_FEE_BOUND", 0.0),
    ("own", 0, "ILLUSTRATIVE_FEE_STRESS", 0.0),
    ("own+partner", 0, "ZERO_FEE_BOUND", 0.0),
    ("own+partner", 0, "ILLUSTRATIVE_FEE_STRESS", 0.0),
    ("own+partner+geom", 0, "ZERO_FEE_BOUND", 0.0),
]
FEATS = {"own": OWN, "own+partner": OWN + PARTNER, "own+partner+geom": OWN + PARTNER + GEOM}


def boot_ci(per_window: np.ndarray, n=2000, seed=SEED):
    """Cluster bootstrap over windows (quarter-hours). Returns mean, lo, hi of ¢/window."""
    if len(per_window) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_window), size=(n, len(per_window)))
    m = per_window[idx].mean(1)
    return float(per_window.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def train_sign_flip(train_w, val_w, n_in, hidden, fee_mode, lam, epochs=300, seed=SEED):
    """Null: multiply each training window's dm by an independent ±1. Destroys feature→drift relations, keeps dynamics."""
    g = torch.Generator().manual_seed(seed + 1)
    tw = []
    for w in train_w:
        w2 = dict(w)
        s = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
        w2["dm"] = w["dm"] * s
        tw.append(w2)
    return train_policy(tw, val_w, n_in, hidden, fee_mode, lam, epochs=epochs, seed=seed)


def main() -> int:
    global EXP
    SRC, EXP = resolve(parser("Sprint 09B diagnostics: weights, bootstrap, sign-flip null.").parse_args())
    df = prepare(pd.read_parquet(SRC))
    require_columns(df)
    scores = pd.read_csv(EXP / "policy_scores.csv")
    trades = pd.read_parquet(EXP / "paper_trades.parquet")
    hist = pd.read_csv(EXP / "training_history.csv")

    # ---- 1. per-window bootstrap + side concentration for every ADJOINT / P1 config on val + holdout
    rows = []
    key = ["pair", "policy", "feat_set", "hidden", "fee_mode", "lambda", "null", "split"]
    for _, r in scores[scores.split.isin(["validation", "holdout"])].iterrows():
        n_w = int(r.windows)
        if r.policy == "ADJOINT":
            t = trades[(trades.pair == r.pair) & (trades.policy == "ADJOINT") & (trades.feat_set == r.feat_set) & (trades.hidden == r.hidden) & (trades["lambda"] == r["lambda"]) & (trades["null"] == r["null"]) & (trades.fee_mode == r.fee_mode) & (trades.split == r.split)]
        else:
            t = trades.iloc[0:0]
        wins = sorted(df[(df.pair == r.pair) & (df.split == r.split)].window_id.unique())
        pw = pd.Series(0.0, index=wins)
        if len(t):
            pw = pw.add(t.groupby("window_id").pnl_cents.sum(), fill_value=0.0)
        mean, lo, hi = boot_ci(pw.to_numpy())
        rows.append({**{k: r[k] for k in key}, "windows": n_w, "n_cycles": int(r.n_cycles), "per_window_mean": mean, "ci_lo": lo, "ci_hi": hi, "clears_zero": bool(lo > 0), "frac_yes": float((t.side == "YES").mean()) if len(t) else np.nan, "frac_forced_close": float(t.forced_close.mean()) if len(t) and "forced_close" in t else np.nan, "median_hold_steps": float((t.exit_i - t.entry_i).median()) if len(t) and "exit_i" in t else np.nan})
    boot = pd.DataFrame(rows)
    boot.to_csv(EXP / "bootstrap_by_window.csv", index=False)

    # ---- 2. retrain headline configs → weights + sign-flip null + training curve summary
    weights, nullrows = {}, []
    configs = [(pair, *spec) for pair in sorted(df.pair.unique()) for spec in FEATS_PER_PAIR]
    for pair, fs, hidden, fee_mode, lam in configs:
        cols = FEATS[fs]
        sub = df[df.pair == pair]
        train = sub[sub.split.eq("discovery")]
        if train.empty:
            print("skip", pair, "no discovery rows", flush=True)
            continue
        sc = fit_scales(train[train.level_ok], cols)
        wins = to_windows(sub, cols, sc)
        tr = [w for w in wins if w["split"] == "discovery"]
        va = [w for w in wins if w["split"] == "validation"]
        ho = [w for w in wins if w["split"] == "holdout"]
        pol, h, best_ep = train_policy(tr, va, len(cols), hidden, fee_mode, lam, epochs=300, seed=SEED)
        W = pol.l1.weight.detach().numpy()[0]
        b = float(pol.l1.bias.detach().numpy()[0])
        tag = f"{pair}|{fs}|h{hidden}|{fee_mode}|lam{lam}"
        # effective no-trade band: |w_q| gives hysteresis; bias gives standing directional tilt
        weights[tag] = {"weights": {c: float(W[i]) for i, c in enumerate(cols)}, "w_q_prev": float(W[-1]), "bias": b, "best_epoch": int(best_ep), "train_at_best_cents": float(h.loc[best_ep, "train_surrogate_cents"]), "val_at_best_cents": float(h.loc[best_ep, "val_surrogate_cents"]), "val_at_epoch0_cents": float(h.loc[0, "val_surrogate_cents"]), "val_at_last_cents": float(h.iloc[-1]["val_surrogate_cents"])}
        # how much of the output is the bias alone?  q with all features at median (x=0), q_prev=0
        weights[tag]["q_at_median_state"] = float(np.tanh(b))
        # exact replay with the bias removed: does the edge survive without the standing tilt?
        with torch.no_grad():
            pol.l1.bias.zero_()
        for split_name, ws in (("validation", va), ("holdout", ho)):
            cyc = [c for w in ws for c in exact_replay(pol, w, fee_mode)]
            weights[tag][f"nobias_{split_name}_sum_cents"] = float(sum(c["pnl_cents"] for c in cyc))
            weights[tag][f"nobias_{split_name}_cycles"] = len(cyc)
        # sign-flip null
        for s in (SEED, SEED + 10, SEED + 20):
            npol, _, _ = train_sign_flip(tr, va, len(cols), hidden, fee_mode, lam, seed=s)
            for split_name, ws in (("validation", va), ("holdout", ho)):
                cyc = [c for w in ws for c in exact_replay(npol, w, fee_mode)]
                pw = pd.Series(0.0, index=[w["window_id"] for w in ws]).add(pd.DataFrame(cyc).groupby("window_id").pnl_cents.sum() if cyc else pd.Series(dtype=float), fill_value=0.0)
                nullrows.append({"tag": tag, "null": "sign_flip", "seed": s, "split": split_name, "windows": len(ws), "n_cycles": len(cyc), "sum_cents": float(pw.sum()), "per_window": float(pw.mean()), "mean_abs_q": float(np.mean([npol.rollout(w["X"], w["ok"], w["T_left"]).detach().abs().mean().item() for w in ws])) if ws else np.nan})
        print("done", tag, flush=True)
    (EXP / "learned_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    pd.DataFrame(nullrows).to_csv(EXP / "sign_flip_null.csv", index=False)

    # ---- 3. training-curve summary for all configs
    summ = hist.groupby("tag").apply(lambda g: pd.Series({"best_epoch": int(g.val_surrogate_cents.idxmax() - g.index[0]), "val_best": g.val_surrogate_cents.max(), "val_ep0": g.val_surrogate_cents.iloc[0], "val_last": g.val_surrogate_cents.iloc[-1], "train_last": g.train_surrogate_cents.iloc[-1]}), include_groups=False).reset_index()
    summ.to_csv(EXP / "training_curve_summary.csv", index=False)

    # ---- 4. saliency + adjoint summaries
    sal = pd.read_csv(EXP / "input_saliency.csv")
    cols = [c for c in sal.columns if c not in ("pair", "window_id", "split")]
    sal_summary = sal.groupby("pair")[cols].mean().T
    sal_summary.to_csv(EXP / "saliency_summary.csv")
    adj = pd.read_csv(EXP / "adjoint_paths.csv")
    adj_summary = adj.groupby("pair").apply(lambda g: pd.Series({"lambda_abs_median_cents": 100 * g.lambda_t.abs().median(), "lambda_abs_p90_cents": 100 * g.lambda_t.abs().quantile(0.9), "half_spread_median_cents": (g.spread_cents / 2).median(), "frac_abs_lambda_gt_half_spread": float((100 * g.lambda_t.abs() > g.spread_cents / 2).mean())}), include_groups=False)
    adj_summary.to_csv(EXP / "adjoint_summary.csv")

    pd.set_option("display.width", 250)
    print(boot[(boot.policy == "ADJOINT") & (boot["null"] == "none")].sort_values(["pair", "fee_mode", "feat_set", "hidden", "lambda", "split"]).to_string(index=False))
    print(pd.DataFrame(nullrows).to_string(index=False))
    print(json.dumps(weights, indent=1))
    print(sal_summary)
    print(adj_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
