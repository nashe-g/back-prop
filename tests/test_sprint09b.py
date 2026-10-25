"""Sprint 09B tests. From the repo root: python -m unittest tests.test_sprint09b"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adjoint_policy import (  # noqa: E402
    OWN,
    Policy,
    adjoint_path,
    exact_replay,
    fit_scales,
    fixed_rule_replay,
    illustrative_fee,
    prepare,
    surrogate_pnl,
    to_windows,
    train_policy,
)


def _synthetic(n=60, trend=0.0, seed=1):
    rng = np.random.default_rng(seed)
    pB = 0.5 + np.cumsum(rng.normal(0, 0.002, n) + trend)
    t = np.arange(n) * 5
    df = pd.DataFrame(
        {
            "pair": "X-Y",
            "window_id": "w1",
            "split": "discovery",
            "date": "d",
            "domain": "test",
            "t_ns": t * 10**9,
            "phase": t,
            "T_left": 900 - t,
            "level_ok": True,
            "pB": pB,
            "bB": pB - 0.005,
            "aB": pB + 0.005,
            "spreadB": 0.01,
            "dB_open": pB - pB[0],
            "rB5": np.r_[0, np.diff(pB)],
            "imbB": 0.0,
            "dA_open": 0.0,
            "rA5": 0.0,
            "spreadA": 0.01,
            "E_gauss": 0.0,
            "grad_zA_gauss": 0.0,
            "grad_zB_gauss": 0.0,
            "E_resid": 0.0,
        }
    )
    return prepare(df)


class TestSurrogate(unittest.TestCase):
    def test_adjoint_matches_finite_difference(self):
        d = _synthetic()
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]
        pol = Policy(len(OWN), 0)
        lam_t = adjoint_path(pol, w, "ZERO_FEE_BOUND", 0.0)
        q0 = pol.rollout(w["X"], w["ok"], w["T_left"]).detach()
        eps = 1e-4
        for i in (5, 20, 40):
            qp, qm = q0.clone(), q0.clone()
            qp[i] += eps
            qm[i] -= eps
            fd = (float(surrogate_pnl(qp, w, "ZERO_FEE_BOUND", 0.0)) - float(surrogate_pnl(qm, w, "ZERO_FEE_BOUND", 0.0))) / (2 * eps)
            self.assertAlmostEqual(fd, float(lam_t[i]), places=3)

    def test_zero_position_zero_pnl(self):
        d = _synthetic()
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]
        q = torch.zeros(len(w["dm"]))
        self.assertEqual(float(surrogate_pnl(q, w, "ILLUSTRATIVE_FEE_STRESS", 0.001)), 0.0)

    def test_cost_term_penalizes_turnover(self):
        d = _synthetic()
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]
        q_flip = torch.tensor([1.0, -1.0] * (len(w["dm"]) // 2))
        q_hold = torch.ones(len(w["dm"]))
        self.assertLess(float(surrogate_pnl(q_flip, w, "ZERO_FEE_BOUND", 0.0)), float(surrogate_pnl(q_hold, w, "ZERO_FEE_BOUND", 0.0)) + 1.0)
        self.assertLess(float(surrogate_pnl(q_flip, w, "ZERO_FEE_BOUND", 0.0)), -0.2)

    def test_fee_formula(self):
        self.assertAlmostEqual(illustrative_fee(0.5), 0.02)
        self.assertAlmostEqual(illustrative_fee(0.9), 0.01)


class TestTraining(unittest.TestCase):
    def test_learns_to_hold_on_trending_window(self):
        d = _synthetic(trend=0.003, seed=3)
        w = to_windows(d, OWN, fit_scales(d, OWN))
        pol, hist, _ = train_policy(w, w, len(OWN), 0, "ZERO_FEE_BOUND", 0.0, epochs=60, lr=0.05)
        q = pol.rollout(w[0]["X"], w[0]["ok"], w[0]["T_left"]).detach().numpy()
        self.assertGreater(q[5:-12].mean(), 0.5)

    def test_abstains_when_cost_dominates(self):
        d = _synthetic(seed=5)
        d["spreadB"] = 0.5
        w = to_windows(d, OWN, fit_scales(d, OWN))
        pol, hist, _ = train_policy(w, w, len(OWN), 0, "ZERO_FEE_BOUND", 0.0, epochs=60, lr=0.05)
        q = pol.rollout(w[0]["X"], w[0]["ok"], w[0]["T_left"]).detach().numpy()
        self.assertLess(np.abs(q).mean(), 0.15)

    def test_training_history_has_val(self):
        d = _synthetic()
        w = to_windows(d, OWN, fit_scales(d, OWN))
        _, hist, best = train_policy(w, w, len(OWN), 8, "ZERO_FEE_BOUND", 0.0, epochs=5)
        self.assertEqual(len(hist), 5)
        self.assertIn("val_surrogate_cents", hist.columns)
        self.assertGreaterEqual(best, 0)


class TestExactReplay(unittest.TestCase):
    def test_replay_round_trip_accounting(self):
        d = _synthetic(trend=0.003, seed=3)
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]

        class Const(Policy):
            def rollout(self, X, ok, T_left):
                q = torch.ones(X.shape[0])
                q[T_left <= 60] = 0.0
                return q

        pol = Const(len(OWN), 0)
        cyc = exact_replay(pol, w, "ZERO_FEE_BOUND")
        self.assertEqual(len(cyc), 1)
        c = cyc[0]
        self.assertEqual(c["side"], "YES")
        self.assertEqual(c["fill_label"], "NOT_REAL_FILL")
        self.assertAlmostEqual(c["pnl_cents"], 100 * (c["exit_bid"] - c["entry_ask"]), places=9)
        self.assertEqual(c["entry_i"], 1)

    def test_replay_respects_no_entry_final(self):
        d = _synthetic(seed=2)
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]

        class Late(Policy):
            def rollout(self, X, ok, T_left):
                q = torch.zeros(X.shape[0])
                q[T_left <= 60] = 1.0
                return q

        self.assertEqual(exact_replay(Late(len(OWN), 0), w, "ZERO_FEE_BOUND"), [])

    def test_fixed_rule_pays_spread(self):
        d = _synthetic(seed=4)
        d["rB5"] = 0.02
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]
        cyc = fixed_rule_replay(w, "ZERO_FEE_BOUND", "momentum", thr=0.01, max_cycles=1)
        self.assertEqual(len(cyc), 1)
        self.assertLess(cyc[0]["pnl_cents"], 0.5)

    def test_fee_stress_lower_than_zero_fee(self):
        d = _synthetic(trend=0.003, seed=3)
        d["rB5"] = 0.02
        w = to_windows(d, OWN, fit_scales(d, OWN))[0]
        a = fixed_rule_replay(w, "ZERO_FEE_BOUND", "momentum", max_cycles=1)
        b = fixed_rule_replay(w, "ILLUSTRATIVE_FEE_STRESS", "momentum", max_cycles=1)
        self.assertLess(b[0]["pnl_cents"], a[0]["pnl_cents"])


if __name__ == "__main__":
    unittest.main()
