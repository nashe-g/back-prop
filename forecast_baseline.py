#!/usr/bin/env python3
"""Forecast-then-threshold baseline on identical features: ridge → ŷ(15s) → trade when |ŷ| > half-spread (+fee). Exact replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import parser, require_columns, resolve  # noqa: E402
from adjoint_policy import OWN, PARTNER, _arrays, _replay, fit_scales, illustrative_fee, prepare, to_windows  # noqa: E402

EXP: Path


def ridge(X, y, alpha=1.0):
    Xb = np.column_stack([X, np.ones(len(X))])
    A = Xb.T @ Xb + alpha * np.eye(Xb.shape[1])
    A[-1, -1] -= alpha
    return np.linalg.solve(A, Xb.T @ y)


def main() -> int:
    global EXP
    SRC, EXP = resolve(parser("Forecast-then-threshold ridge baseline on the same features.").parse_args())
    df = prepare(pd.read_parquet(SRC))
    require_columns(df)
    cols = OWN + PARTNER
    rows = []
    for pair in sorted(df.pair.unique()):
        sub = df[df.pair == pair]
        train = sub[sub.split.eq("discovery")]
        if train.empty:
            print("skip", pair, "no discovery rows", flush=True)
            continue
        sc = fit_scales(train[train.level_ok], cols)
        wins = to_windows(sub, cols, sc)
        # label: 15s-ahead mid change (3 steps), only where all steps valid
        Xs, ys = [], []
        for w in wins:
            if w["split"] != "discovery":
                continue
            dm = w["dm"].numpy()
            ok = w["ok"].numpy()
            y = dm + np.r_[dm[1:], 0] + np.r_[dm[2:], 0, 0]
            good = ok & np.r_[ok[1:], False] & np.r_[ok[2:], False, False]
            Xs.append(w["X"].numpy()[good])
            ys.append(y[good])
        if not Xs or sum(len(x) for x in Xs) == 0:
            print("skip", pair, "no labeled discovery rows", flush=True)
            continue
        beta = ridge(np.vstack(Xs), np.concatenate(ys))
        for fee_mode in ("ZERO_FEE_BOUND", "ILLUSTRATIVE_FEE_STRESS"):
            for mult in (1.0, 2.0):
                for split_name in ("validation", "holdout"):
                    ws = [w for w in wins if w["split"] == split_name]
                    cyc = []
                    n_sig = 0
                    for w in ws:
                        X = w["X"].numpy()
                        pred = X @ beta[:-1] + beta[-1]
                        ok, bB, aB, tl, _ = _arrays(w)
                        sp = np.nan_to_num(w["df"].spreadB.to_numpy(float), nan=0.02)
                        mid = np.nan_to_num(w["df"].pB.to_numpy(float), nan=0.5)
                        thr = mult * (0.5 * sp + (np.array([illustrative_fee(m) for m in mid]) if fee_mode != "ZERO_FEE_BOUND" else 0.0))
                        n_sig += int((np.abs(pred) > thr).sum())

                        def decide(i, pos, side, pred=pred, thr=thr):
                            if pos == "LONG":
                                return None
                            if pred[i] > thr[i]:
                                return "YES"
                            if pred[i] < -thr[i]:
                                return "NO"
                            return None

                        cyc.extend(_replay(w, fee_mode, decide, max_cycles=3, max_hold=3))
                    pw = pd.Series(0.0, index=[w["window_id"] for w in ws])
                    if cyc:
                        pw = pw.add(pd.DataFrame(cyc).groupby("window_id").pnl_cents.sum(), fill_value=0.0)
                    rng = np.random.default_rng(7)
                    arr = pw.to_numpy()
                    bs = arr[rng.integers(0, len(arr), size=(2000, len(arr)))].mean(1)
                    rows.append({"pair": pair, "policy": "FORECAST_THRESHOLD_RIDGE", "threshold_mult": mult, "fee_mode": fee_mode, "split": split_name, "windows": len(ws), "signals": n_sig, "n_cycles": len(cyc), "sum_cents": float(pw.sum()), "per_window": float(arr.mean()), "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975))})
        print(pair, "ridge beta", dict(zip(cols + ["bias"], np.round(beta, 5))), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(EXP / "forecast_threshold_baseline.csv", index=False)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
