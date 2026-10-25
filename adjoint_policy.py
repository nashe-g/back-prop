"""Sprint 09B — adjoint (BPTT) policy on cost-aware P&L + exact discrete replay. No live orders."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

NS = 1_000_000_000
OWN = ["dB_open", "rB5", "spreadB", "imbB", "phase_frac", "T_left_frac"]
PARTNER = ["dA_open", "rA5", "spreadA"]
GEOM = ["E_gauss", "grad_zA_gauss", "grad_zB_gauss", "E_resid"]
NO_ENTRY_FINAL = 60.0


def illustrative_fee(p):
    if isinstance(p, torch.Tensor):
        pc = torch.clamp(p, 0.01, 0.99)
        return 0.07 * pc * (1 - pc)
    pc = min(max(p, 0.01), 0.99)
    return math.ceil(0.07 * pc * (1 - pc) * 100) / 100


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase_frac"] = out["phase"] / 900.0
    out["T_left_frac"] = out["T_left"] / 900.0
    out = out.sort_values(["pair", "window_id", "t_ns"]).reset_index(drop=True)
    out["dm_next"] = out.groupby(["pair", "window_id"])["pB"].shift(-1) - out["pB"]
    out["ok_next"] = out.groupby(["pair", "window_id"])["level_ok"].shift(-1).fillna(False).astype(bool) & out["level_ok"].astype(bool)
    for c in OWN + PARTNER + GEOM:
        if c not in out.columns:
            out[c] = 0.0
    return out


def fit_scales(train: pd.DataFrame, cols: list[str]) -> dict:
    sc = {}
    for c in cols:
        x = train[c].to_numpy(float)
        x = x[np.isfinite(x)]
        mu = float(np.median(x)) if len(x) else 0.0
        sd = float(np.median(np.abs(x - mu))) * 1.4826 if len(x) else 1.0
        sc[c] = (mu, max(sd, 1e-3))
    return sc


def to_windows(df: pd.DataFrame, cols: list[str], sc: dict) -> list[dict]:
    wins = []
    for (pair, w), g in df.groupby(["pair", "window_id"], sort=False):
        g = g.sort_values("t_ns").reset_index(drop=True)
        X = np.column_stack([(g[c].fillna(0).to_numpy(float) - sc[c][0]) / sc[c][1] for c in cols])
        X = np.clip(X, -6, 6)
        wins.append(
            {
                "pair": pair,
                "window_id": w,
                "split": g.split.iloc[0],
                "date": g.date.iloc[0] if "date" in g.columns else "",
                "X": torch.tensor(X, dtype=torch.float32),
                "dm": torch.tensor(np.nan_to_num(g.dm_next.to_numpy(float)), dtype=torch.float32),
                "ok": torch.tensor(g.ok_next.to_numpy(bool)),
                "spread": torch.tensor(np.nan_to_num(g.spreadB.to_numpy(float), nan=0.02), dtype=torch.float32),
                "mid": torch.tensor(np.nan_to_num(g.pB.to_numpy(float), nan=0.5), dtype=torch.float32),
                "T_left": torch.tensor(g.T_left.to_numpy(float), dtype=torch.float32),
                "df": g,
            }
        )
    return wins


def batch(wins: list[dict]) -> dict:
    """Pad windows to equal length. Padded steps: ok=False, T_left=0 (forces flat), dm=0."""
    T = max(len(w["dm"]) for w in wins)
    B = len(wins)
    F = wins[0]["X"].shape[1]
    X = torch.zeros(B, T, F)
    dm = torch.zeros(B, T)
    ok = torch.zeros(B, T, dtype=torch.bool)
    sp = torch.full((B, T), 0.02)
    mid = torch.full((B, T), 0.5)
    tl = torch.zeros(B, T)
    for i, w in enumerate(wins):
        n = len(w["dm"])
        X[i, :n] = w["X"]
        dm[i, :n] = w["dm"]
        ok[i, :n] = w["ok"]
        sp[i, :n] = w["spread"]
        mid[i, :n] = w["mid"]
        tl[i, :n] = w["T_left"]
    return {"X": X, "dm": dm, "ok": ok, "spread": sp, "mid": mid, "T_left": tl}


class Policy(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int = 0):
        super().__init__()
        self.hidden = hidden
        if hidden > 0:
            self.l1 = torch.nn.Linear(n_in + 1, hidden)
            self.l2 = torch.nn.Linear(hidden, 1)
        else:
            self.l1 = torch.nn.Linear(n_in + 1, 1)
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(0.1)
            self.l1.bias.zero_()

    def head(self, z: torch.Tensor) -> torch.Tensor:
        if self.hidden > 0:
            return torch.tanh(self.l2(torch.tanh(self.l1(z)))).squeeze(-1)
        return torch.tanh(self.l1(z)).squeeze(-1)

    def rollout_batch(self, X: torch.Tensor, ok: torch.Tensor, T_left: torch.Tensor) -> torch.Tensor:
        """X (B,T,F) → q (B,T). Decision only at valid samples outside the final 60s; flat in final 60s."""
        B, T, _ = X.shape
        q_prev = torch.zeros(B)
        qs = []
        for t in range(T):
            q_new = self.head(torch.cat([X[:, t, :], q_prev.unsqueeze(1)], dim=1))
            act = ok[:, t] & (T_left[:, t] > NO_ENTRY_FINAL)
            flat = T_left[:, t] <= NO_ENTRY_FINAL
            q = torch.where(act, q_new, torch.where(flat, torch.zeros_like(q_prev), q_prev))
            qs.append(q)
            q_prev = q
        return torch.stack(qs, dim=1)

    def rollout(self, X: torch.Tensor, ok: torch.Tensor, T_left: torch.Tensor) -> torch.Tensor:
        return self.rollout_batch(X.unsqueeze(0), ok.unsqueeze(0), T_left.unsqueeze(0))[0]


def surrogate_batch(q: torch.Tensor, b: dict, fee_mode: str, lam: float) -> torch.Tensor:
    """Per-window J (B,). J = Σ q_t dm_{t+1} − Σ (s_t/2 + fee_t)|q_t − q_{t−1}| − λ Σ q_t²  (+ forced flatten at end)."""
    half = 0.5 * b["spread"]
    fee = illustrative_fee(b["mid"]) if fee_mode == "ILLUSTRATIVE_FEE_STRESS" else torch.zeros_like(half)
    q_prev = torch.cat([torch.zeros(q.shape[0], 1), q[:, :-1]], dim=1)
    gain = (q * b["dm"] * b["ok"].float()).sum(1)
    cost = ((half + fee) * (q - q_prev).abs()).sum(1) + (half[:, -1] + fee[:, -1]) * q[:, -1].abs()
    return gain - cost - lam * (q * q).sum(1)


def surrogate_pnl(q: torch.Tensor, w: dict, fee_mode: str, lam: float) -> torch.Tensor:
    return surrogate_batch(q.unsqueeze(0), batch([w]), fee_mode, lam)[0]


def train_policy(train_w, val_w, n_in, hidden, fee_mode, lam, epochs=150, lr=0.01, seed=7, shuffle_labels=False):
    torch.manual_seed(seed)
    pol = Policy(n_in, hidden)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    tb = batch(train_w)
    vb = batch(val_w) if val_w else None
    if shuffle_labels:
        g = torch.Generator().manual_seed(seed)
        dm = tb["dm"].clone()
        for i in range(dm.shape[0]):
            dm[i] = dm[i][torch.randperm(dm.shape[1], generator=g)]
        tb["dm"] = dm
    best, best_state, best_ep, hist = -1e9, None, -1, []
    for ep in range(epochs):
        opt.zero_grad()
        J = surrogate_batch(pol.rollout_batch(tb["X"], tb["ok"], tb["T_left"]), tb, fee_mode, lam).mean()
        (-J).backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            v = float(surrogate_batch(pol.rollout_batch(vb["X"], vb["ok"], vb["T_left"]), vb, fee_mode, lam).mean()) if vb else float(J)
        hist.append({"epoch": ep, "train_surrogate_cents": 100 * float(J.detach()), "val_surrogate_cents": 100 * v})
        if v > best:
            best, best_ep = v, ep
            best_state = {k: t.detach().clone() for k, t in pol.state_dict().items()}
    if best_state is not None:
        pol.load_state_dict(best_state)
    return pol, pd.DataFrame(hist), best_ep


def adjoint_path(pol: Policy, w: dict, fee_mode: str, lam: float) -> np.ndarray:
    """λ_t = ∂J/∂q_t along the realized rollout: shadow price of one unit of inventory at t."""
    q = pol.rollout(w["X"], w["ok"], w["T_left"]).detach().requires_grad_(True)
    (g,) = torch.autograd.grad(surrogate_pnl(q, w, fee_mode, lam), q)
    return g.numpy()


def input_saliency(pol: Policy, w: dict, fee_mode: str, lam: float) -> np.ndarray:
    X = w["X"].clone().requires_grad_(True)
    (g,) = torch.autograd.grad(surrogate_pnl(pol.rollout(X, w["ok"], w["T_left"]), w, fee_mode, lam), X)
    return g.numpy()


def _arrays(w: dict):
    g = w["df"]
    return (
        g.level_ok.to_numpy(bool),
        g.bB.to_numpy(float),
        g.aB.to_numpy(float),
        g.T_left.to_numpy(float),
        np.nan_to_num(g.rB5.to_numpy(float)),
    )


def _cycle(w, fee_mode, side, entry_i, i, entry_ask, bid, forced=False):
    f = (lambda p: 0.0) if fee_mode == "ZERO_FEE_BOUND" else illustrative_fee
    return {
        "pair": w["pair"],
        "window_id": w["window_id"],
        "split": w["split"],
        "date": w["date"],
        "side": side,
        "entry_i": int(entry_i),
        "exit_i": int(i),
        "entry_ask": float(entry_ask),
        "exit_bid": float(bid),
        "pnl_cents": 100.0 * (bid - entry_ask - f(entry_ask) - f(bid)),
        "fee_mode": fee_mode,
        "fill_label": "NOT_REAL_FILL",
        "forced_close": forced,
    }


def _replay(w: dict, fee_mode: str, decide, max_cycles: int, max_hold: int | None = None) -> list[dict]:
    """Shared exact discrete YES/NO replay. decide(i, pos, side) → 'YES'|'NO'|'EXIT'|None.
    Decision at valid sample i executes at the NEXT valid sample (ask to enter, bid to exit). NOT_REAL_FILL."""
    ok, bB, aB, tl, _ = _arrays(w)
    n = len(ok)
    pos, side, entry_ask, entry_i, cycles, pending, done = "FLAT", "", np.nan, -1, [], None, 0
    for i in range(n):
        if not ok[i]:
            continue
        if pending is not None:
            if pending[0] == "ENTER" and pos == "FLAT":
                s = pending[1]
                entry_ask = aB[i] if s == "YES" else 1.0 - bB[i]
                pos, side, entry_i = "LONG", s, i
            elif pending[0] == "EXIT" and pos == "LONG":
                bid = bB[i] if side == "YES" else 1.0 - aB[i]
                cycles.append(_cycle(w, fee_mode, side, entry_i, i, entry_ask, bid))
                pos, side, done = "FLAT", "", done + 1
            pending = None
        if pos == "LONG":
            if tl[i] <= NO_ENTRY_FINAL / 2 or (max_hold is not None and i - entry_i >= max_hold) or decide(i, pos, side) == "EXIT":
                pending = ("EXIT",)
        elif done < max_cycles and tl[i] > NO_ENTRY_FINAL:
            s = decide(i, pos, side)
            if s in ("YES", "NO"):
                pending = ("ENTER", s)
    if pos == "LONG":
        last = int(np.flatnonzero(ok)[-1])
        bid = bB[last] if side == "YES" else 1.0 - aB[last]
        cycles.append(_cycle(w, fee_mode, side, entry_i, last, entry_ask, bid, forced=True))
    return cycles


def exact_replay(pol: Policy, w: dict, fee_mode: str, band_on=0.5, band_off=0.2, max_cycles=5) -> list[dict]:
    with torch.no_grad():
        q = pol.rollout(w["X"], w["ok"], w["T_left"]).numpy()

    def decide(i, pos, side):
        qi = float(q[i])
        if pos == "LONG":
            if (side == "YES" and qi < band_off) or (side == "NO" and qi > -band_off):
                return "EXIT"
            return None
        if qi > band_on:
            return "YES"
        if qi < -band_on:
            return "NO"
        return None

    return _replay(w, fee_mode, decide, max_cycles)


def fixed_rule_replay(w: dict, fee_mode: str, branch: str, thr=0.01, max_cycles=3, max_hold=3) -> list[dict]:
    r = _arrays(w)[4]

    def decide(i, pos, side):
        if pos == "LONG":
            return None
        if branch == "momentum":
            return "YES" if r[i] >= thr else ("NO" if r[i] <= -thr else None)
        return "NO" if r[i] >= thr else ("YES" if r[i] <= -thr else None)

    return _replay(w, fee_mode, decide, max_cycles, max_hold=max_hold)
