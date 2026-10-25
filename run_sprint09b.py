#!/usr/bin/env python3
"""Sprint 09B runner: adjoint policy learning vs fixed rules vs abstention. No live orders."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cli import parser, require_columns, resolve  # noqa: E402
from adjoint_policy import (  # noqa: E402
    GEOM,
    OWN,
    PARTNER,
    adjoint_path,
    exact_replay,
    fit_scales,
    fixed_rule_replay,
    input_saliency,
    prepare,
    to_windows,
    train_policy,
)

HERE = Path(__file__).resolve().parent
PLAN_SRC = HERE / "analysis_plan.yaml"
SEED = 7
EXP: Path  # set in main()


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main() -> int:
    global EXP
    SRC, EXP = resolve(parser("Sprint 09B: adjoint policy learning. No live orders.").parse_args())
    shutil.copy2(PLAN_SRC, EXP / "analysis_plan.yaml")
    ph = sha(EXP / "analysis_plan.yaml")
    (EXP / "plan_freeze.json").write_text(
        json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "plan_hash": ph, "source_states_sha256": sha(SRC), "torch": torch.__version__}, indent=2),
        encoding="utf-8",
    )
    print("plan", ph, "states", SRC, "out", EXP, flush=True)
    torch.set_num_threads(4)

    df = prepare(pd.read_parquet(SRC))
    require_columns(df)
    results, cycles_all, hist_all, sal_all, adj_all = [], [], [], [], []

    for pair in sorted(df.pair.unique()):
        sub = df[df.pair == pair].copy()
        train = sub[sub.split.eq("discovery")]
        if train.empty:
            print("skip", pair, "no discovery rows", flush=True)
            continue
        domain = sub.domain.iloc[0] if "domain" in sub.columns else ""
        for feat_set, cols in (("own", OWN), ("own+partner", OWN + PARTNER), ("own+partner+geom", OWN + PARTNER + GEOM)):
            sc = fit_scales(train[train.level_ok], cols)
            wins = to_windows(sub, cols, sc)
            tr = [w for w in wins if w["split"] == "discovery"]
            va = [w for w in wins if w["split"] == "validation"]
            ho = [w for w in wins if w["split"] == "holdout"]
            for hidden in (0, 8):
                for fee_mode in ("ZERO_FEE_BOUND", "ILLUSTRATIVE_FEE_STRESS"):
                    for lam in (0.0, 0.001):
                        if lam > 0 and hidden > 0:
                            continue
                        for null in ("none", "label_shuffle"):
                            if null != "none" and not (feat_set == "own+partner" and hidden == 0 and lam == 0.0):
                                continue
                            tag = f"{pair}|{feat_set}|h{hidden}|{fee_mode}|lam{lam}|{null}"
                            print("train", tag, flush=True)
                            pol, hist, best_ep = train_policy(tr, va, len(cols), hidden, fee_mode, lam, epochs=300, lr=0.01, seed=SEED, shuffle_labels=(null == "label_shuffle"))
                            hist["tag"] = tag
                            hist_all.append(hist)
                            # exact replay all splits
                            for split_name, ws in (("discovery", tr), ("validation", va), ("holdout", ho)):
                                cyc = []
                                for w in ws:
                                    cyc.extend(exact_replay(pol, w, fee_mode))
                                c = pd.DataFrame(cyc)
                                qmean = float(np.mean([np.abs(pol.rollout(w["X"], w["ok"], w["T_left"]).detach().numpy()).mean() for w in ws])) if ws else np.nan
                                results.append(
                                    {
                                        "domain": domain,
                                        "pair": pair,
                                        "policy": "ADJOINT",
                                        "feat_set": feat_set,
                                        "hidden": hidden,
                                        "fee_mode": fee_mode,
                                        "lambda": lam,
                                        "null": null,
                                        "split": split_name,
                                        "windows": len(ws),
                                        "n_cycles": int(len(c)),
                                        "sum_pnl_cents": float(c.pnl_cents.sum()) if len(c) else 0.0,
                                        "mean_pnl_cents": float(c.pnl_cents.mean()) if len(c) else np.nan,
                                        "pnl_per_window_cents": (float(c.pnl_cents.sum()) if len(c) else 0.0) / max(len(ws), 1),
                                        "mean_abs_q": qmean,
                                        "best_epoch": best_ep,
                                        "fill_label": "NOT_REAL_FILL",
                                    }
                                )
                                for cc in cyc:
                                    cc.update({"policy": "ADJOINT", "feat_set": feat_set, "hidden": hidden, "lambda": lam, "null": null})
                                cycles_all.extend(cyc)
                            # saliency + adjoint on holdout for primary config
                            if feat_set == "own+partner+geom" and hidden == 0 and fee_mode == "ZERO_FEE_BOUND" and lam == 0.0 and null == "none":
                                for w in (va + ho)[:40]:
                                    S = input_saliency(pol, w, fee_mode, lam)
                                    sal_all.append({"pair": pair, "window_id": w["window_id"], "split": w["split"], **{c: float(np.abs(S[:, i]).mean()) for i, c in enumerate(cols)}})
                                for w in ho[:3]:
                                    lam_t = adjoint_path(pol, w, fee_mode, lam)
                                    g = w["df"].reset_index(drop=True)
                                    for i in range(len(g)):
                                        adj_all.append({"pair": pair, "window_id": w["window_id"], "t_ns": int(g.t_ns.iloc[i]), "phase": float(g.phase.iloc[i]), "lambda_t": float(lam_t[i]), "spread_cents": 100 * float(g.spreadB.iloc[i])})
        # fixed baselines
        wins = to_windows(sub, OWN, fit_scales(train[train.level_ok], OWN))
        for fee_mode in ("ZERO_FEE_BOUND", "ILLUSTRATIVE_FEE_STRESS"):
            for branch in ("momentum", "fade"):
                for split_name in ("discovery", "validation", "holdout"):
                    ws = [w for w in wins if w["split"] == split_name]
                    cyc = []
                    for w in ws:
                        cyc.extend(fixed_rule_replay(w, fee_mode, branch))
                    c = pd.DataFrame(cyc)
                    results.append(
                        {
                            "domain": domain,
                            "pair": pair,
                            "policy": f"P1_{branch}",
                            "feat_set": "own",
                            "hidden": 0,
                            "fee_mode": fee_mode,
                            "lambda": 0.0,
                            "null": "none",
                            "split": split_name,
                            "windows": len(ws),
                            "n_cycles": int(len(c)),
                            "sum_pnl_cents": float(c.pnl_cents.sum()) if len(c) else 0.0,
                            "mean_pnl_cents": float(c.pnl_cents.mean()) if len(c) else np.nan,
                            "pnl_per_window_cents": (float(c.pnl_cents.sum()) if len(c) else 0.0) / max(len(ws), 1),
                            "mean_abs_q": np.nan,
                            "best_epoch": -1,
                            "fill_label": "NOT_REAL_FILL",
                        }
                    )
            for split_name in ("discovery", "validation", "holdout"):
                results.append({"domain": domain, "pair": pair, "policy": "P0", "feat_set": "none", "hidden": 0, "fee_mode": fee_mode, "lambda": 0.0, "null": "none", "split": split_name, "windows": len([w for w in wins if w["split"] == split_name]), "n_cycles": 0, "sum_pnl_cents": 0.0, "mean_pnl_cents": 0.0, "pnl_per_window_cents": 0.0, "mean_abs_q": 0.0, "best_epoch": -1, "fill_label": "NOT_REAL_FILL"})

    res = pd.DataFrame(results)
    res.to_csv(EXP / "policy_scores.csv", index=False)
    pd.DataFrame(cycles_all).to_parquet(EXP / "paper_trades.parquet", index=False)
    pd.concat(hist_all, ignore_index=True).to_csv(EXP / "training_history.csv", index=False)
    pd.DataFrame(sal_all).to_csv(EXP / "input_saliency.csv", index=False)
    pd.DataFrame(adj_all).to_csv(EXP / "adjoint_paths.csv", index=False)
    print(res[res.split.isin(["validation", "holdout"])].sort_values(["pair", "fee_mode", "policy"])[["pair", "policy", "feat_set", "hidden", "fee_mode", "lambda", "null", "split", "windows", "n_cycles", "sum_pnl_cents", "pnl_per_window_cents", "mean_abs_q"]].to_string(index=False), flush=True)
    _plots(res, pd.DataFrame(adj_all), pd.DataFrame(sal_all))
    print("done", EXP, flush=True)
    return 0


def _plots(res, adj, sal):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = EXP / "plots"
    for pair in res.pair.unique():
        r = res[(res.pair == pair) & (res.fee_mode == "ZERO_FEE_BOUND") & (res.split.isin(["validation", "holdout"])) & (res["null"] == "none")]
        if r.empty:
            continue
        r = r.assign(label=r.policy + "/" + r.feat_set + "/h" + r.hidden.astype(str) + "/λ" + r["lambda"].astype(str))
        piv = r.pivot_table(index="label", columns="split", values="pnl_per_window_cents", aggfunc="first").fillna(0)
        fig, ax = plt.subplots(figsize=(7.5, max(3, 0.3 * len(piv))))
        y = np.arange(len(piv))
        ax.barh(y - 0.2, piv.get("validation", 0), height=0.4, color="#777", label="validation")
        ax.barh(y + 0.2, piv.get("holdout", 0), height=0.4, color="#222", label="holdout")
        ax.set_yticks(y)
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.axvline(0, color="#999", lw=0.7)
        ax.set_xlabel("¢ per window (zero fee, exact replay, NOT_REAL_FILL)")
        ax.set_title(f"Policy P&L · {pair}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(P / f"policy_pnl_{pair}.png", dpi=110)
        plt.close(fig)
    if len(adj):
        for pair, g in adj.groupby("pair"):
            w0 = g.window_id.iloc[0]
            gg = g[g.window_id == w0]
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(gg.phase, 100 * gg.lambda_t, lw=1, c="#222", label="λ_t = ∂J/∂q_t (¢)")
            ax.plot(gg.phase, gg.spread_cents / 2, lw=0.8, c="#999", label="half spread ¢")
            ax.plot(gg.phase, -gg.spread_cents / 2, lw=0.8, c="#999")
            ax.set_xlabel("seconds in window")
            ax.set_title(f"Adjoint shadow price of inventory · {pair} {w0}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(P / f"adjoint_{pair}.png", dpi=110)
            plt.close(fig)
    if len(sal):
        cols = [c for c in sal.columns if c not in ("pair", "window_id", "split")]
        for pair, g in sal.groupby("pair"):
            m = g[cols].mean().sort_values()
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(range(len(m)), m.values, color="#444")
            ax.set_yticks(range(len(m)))
            ax.set_yticklabels(m.index, fontsize=8)
            ax.set_xlabel("mean |∂J/∂x| (validation+holdout)")
            ax.set_title(f"Input saliency · {pair}")
            fig.tight_layout()
            fig.savefig(P / f"saliency_{pair}.png", dpi=110)
            plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
