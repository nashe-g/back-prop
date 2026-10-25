# Sprint 09B test report

Command (from /Users/nashe/Kalshi): `crypto/data_download/.venv/bin/python -m unittest tests.test_sprint09b`

Result: Ran 11 tests — OK.

| test | checks |
|---|---|
| test_adjoint_matches_finite_difference | λ_t = ∂J/∂q_t from autograd equals central finite difference at three steps (3 dp) |
| test_zero_position_zero_pnl | q ≡ 0 gives J = 0 under fee stress and λ > 0 |
| test_cost_term_penalizes_turnover | alternating ±1 inventory is penalized relative to holding |
| test_fee_formula | illustrative fee rounds up to the cent (0.5 → 2¢, 0.9 → 1¢) |
| test_learns_to_hold_on_trending_window | BPTT policy on a trending synthetic reaches mean q > 0.5 |
| test_abstains_when_cost_dominates | with a 50¢ spread the trained policy has mean |q| < 0.15 |
| test_training_history_has_val | history length equals epochs; validation column present; best epoch recorded |
| test_replay_round_trip_accounting | constant-long policy → exactly one YES cycle, entry at next valid sample, pnl = bid − ask, NOT_REAL_FILL |
| test_replay_respects_no_entry_final | signals only inside the last 60 s produce no trades |
| test_fixed_rule_pays_spread | P1 momentum round trip on a flat synthetic costs the spread |
| test_fee_stress_lower_than_zero_fee | fee-stress P&L is below zero-fee P&L for the same cycle |

Bug found by the tests and fixed before the main run: a position still open at the last valid sample was silently dropped by the replayer; both replayers now force-close it at the last valid bid and tag the cycle `forced_close=True`.

Runtime: full run 617 s (80 policy trainings + replays), diagnostics 236 s, forecast baseline 3 s.
