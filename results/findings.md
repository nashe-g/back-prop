# Sprint 09B — Backpropagation through the decision (adjoint policy learning)

Status: PILOT_ONLY_REUSED_ARCHIVE · FEE_UNVERIFIED · NOT_REAL_FILL · TRADEABILITY_NOT_ESTABLISHED · no orders placed.
Plan frozen: `analysis_plan.yaml` sha256 `0df3f8846a6e6d57b826decc3e185c8fb1adc0db4e331f96412847bcd442bc93` (see `plan_freeze.json`).
Inputs: Sprint 09 `asof_states.parquet` (5 s grid; BTC-ETH, BTC-DOGE on 2026-10-13/14/15 train, 10-18 validation, 10-19 holdout; GOLD-SILVER, GOLD-WTI on 2026-10-22, 15/4/4 windows). Torch 2.14.1, seed 7.

## 1. Why this experiment exists

Sprint 09 did not back-propagate anything. Its "gradient" ∇E = Σ⁻¹(z−μ) was a fixed linear re-labeling of the state; no parameter was ever updated by a gradient of a trading objective; the differentiable policy (P5) was gated off; policies were hand thresholds. This sprint does the one thing the idea actually requires: **a policy whose parameters are set by the gradient of cost-aware P&L, propagated backward through time via the inventory recurrence**, then frozen and judged on an exact discrete YES/NO replay it never saw during training.

## 2. What was built

- Policy: q_t = tanh(W·[x_t, q_{t−1}] + b), q ∈ (−1, 1) (linear, 10–14 parameters) and a tanh-8 variant (≈100 parameters). q_{t−1} as an input gives the chain rule a path through time (BPTT) and lets the policy learn hysteresis.
- Surrogate objective per window: J = Σ q_t Δm_{t+1} − Σ (s_t/2 + fee_t)|q_t − q_{t−1}| − λ Σ q_t², plus a forced flatten at the end and q ≡ 0 in the final 60 s. Adam, lr 0.01, 300 epochs, early stopping on validation surrogate only.
- Exact replay (never differentiated): q crosses ±0.5 → enter at the **ask** of that side at the **next valid 5 s sample**; q falls inside ±0.2 → exit at the **bid** at the next valid sample; ≤ 5 round trips per window; no entry in the final 60 s; positions open at the last valid sample are closed there (`forced_close`). Fees: ZERO_FEE_BOUND and ILLUSTRATIVE_FEE_STRESS (0.07·P·(1−P), unverified).
- Adjoint λ_t = ∂J/∂q_t and input saliency ∂J/∂x_t via autograd; finite-difference check in tests.
- Baselines on identical data: P0; fixed P1 momentum/fade (rB5 ≥ 1¢, 15 s hold); forecast-then-threshold ridge on identical features (ŷ(15 s) vs half-spread); label-shuffled training; sign-flip training (each training window's Δm multiplied by an independent ±1; three seeds).
- Ablations: feature sets own / own+partner / own+partner+geom (E_gauss, ∇E, E_resid); hidden 0/8; λ ∈ {0, 0.001}.
- Tests: `tests/test_sprint09b.py` (11 OK) — adjoint equals finite differences, cost term penalizes turnover, policy learns to hold on a trending synthetic, policy abstains when cost dominates, exact replay accounting, forced close, no-entry window, fee stress ordering.

## 3. Results

### 3.1 Under cost, the validation-selected policy is abstention (H_ABSTAIN: supported, with a precise mechanism)
Under ILLUSTRATIVE_FEE_STRESS the frozen (validation-selected) policy is q ≈ 0 in 28 of 36 configurations: BTC-ETH 9/9 (mean |q| 0.0003–0.03; every weight |w| < 0.03 in the own+partner linear model), BTC-DOGE 8/9, GOLD-SILVER 8/9, GOLD-WTI 3/9. The fee-stress configurations that trade are GOLD-WTI linear policies (§3.3) and one BTC-DOGE geometry model that loses 11.5¢/window on holdout.

The training curves (`training_curve_summary.csv`) show how that abstention arises, and the mechanism has two distinct cases:
- **Own-leg features only, crypto (6 configs):** the training surrogate itself ends at ≈ 0 (train_last −0.46 to 0.03¢/window). The gradient finds nothing to fit in-sample and converges to zero exposure on its own.
- **Partner and geometry features (the other 30):** the training surrogate ends positive in 69% of fee-stress configs (median +4.1¢/window; BTC-ETH own+partner+geom +13.1, GOLD-SILVER geom h8 +33.8) while the validation surrogate ends negative in 94% (median −2.7; BTC-ETH geom −15.4, GOLD-SILVER −7 to −39). The validation curve rises to ≈ −0.4¢ around epochs 30–70, where weights are still near zero, then falls as the policy fits training-day patterns. Early stopping on validation selects the near-zero epoch (median val_best −0.34¢/window ≈ P0).

So the one-run reading is: train the cost-aware policy and watch the pair (train_last, val_last). Own-leg crypto features produce (≈0, ≈0) — nothing there. Partner and geometry features produce (+, −) — in-sample mirages that validation rejects. Either way the selected policy is P0, and the second pattern names which features manufacture the mirage: the same coupled-geometry inputs that 08H-R, 08H.1 and 09 found to carry no incremental signal.

### 3.2 Zero-fee crypto: validation gains do not transfer (H_ADJ: not supported)
BTC-ETH, exact replay, ¢ per window [cluster-bootstrap 95% CI over windows, seed 7]:

| config | validation (64 w) | holdout (7 w) |
|---|---|---|
| own h0 | +0.6 [−6.9, 8.1] | −13.3 [−35.4, 8.4] |
| own h8 | +2.9 [−4.5, 10.3] | −15.6 [−41.0, 9.4] |
| own+partner h0 | +0.9 [−7.3, 8.5] | −10.8 [−34.5, 12.8] |
| own+partner h8 | +2.4 [−6.6, 10.9] | −1.3 [−32.7, 31.6] |
| own+partner+geom h0 | −0.7 [−11.5, 9.4] | −18.2 [−41.6, 5.9] |
| own+partner+geom h8 | abstains | abstains |
| P1 momentum | −2.4 | −2.0 |
| P1 fade | −3.8 | −4.0 |
| forecast-threshold ridge (×1) | −0.3 [−3.4, 2.6] | −10.9 [−19.1, −0.7] |

BTC-DOGE: every validation CI spans zero; holdout (7 windows) spans zero in all configurations. No crypto configuration clears zero on both halves.

What the BTC-ETH linear policy learned (own+partner, zero fee): w(imbB) = +0.63, w(dB_open) = −0.78, w(dA_open) = +0.50, w(q_prev) = +1.10, bias ≈ 0 — follow the book imbalance, fade the move from open, lean with the partner's move, with hysteresis. The ridge forecaster on the same features learned the same signs (imbB +0.0071, dB_open −0.0052, dA_open +0.0041). Decision loss and forecast loss found the same direction here; neither survived the holdout day.

### 3.3 GOLD-WTI: holdout positives that fail the pre-registered rule and lose to the null (NOT_A_CANDIDATE)
Several GOLD-WTI linear policies are positive on the 4 holdout windows with CIs above zero (own h0 zero fee +14.7 [5.9, 23.3]; own+partner h0 fee stress +14.2 [7.3, 22.0]; own+partner h8 zero fee +24.0 [13.4, 38.5]). Three facts rule them out:
1. Validation CIs span zero in every case (e.g. +12.0 [−11.3, 33.1]). The rule requires both halves clear of zero after fee.
2. The sign-flip null (training labels destroyed) with seed 7 produces +32.5¢/window on the same holdout windows zero fee and +17.2 under fee stress — larger than the real policy. Two other null seeds abstain. A policy trained on nonsense matching or beating the real policy on this holdout means the holdout number carries no information about the features.
3. What the policy is: w(dB_open) = +1.40, w(q_prev) = +2.02, bias = +0.53, all other weights < 0.07. It is "go with WTI's move from the open and stay there" — median hold 83–136 five-second steps (7–11 min), 29% of cycles closed by the end-of-window force. On 8 windows of one afternoon this is a bet on four quarter-hour drifts. Removing the bias leaves holdout at +112.9¢ over 5 cycles (still the same drift bet).

### 3.4 Geometry features in the policy (H_GEOM_IN_POLICY: adding them does not help)
Adding E_gauss, ∇E, E_resid to the policy input: BTC-ETH linear validation +0.9 → −0.7, holdout −10.8 → −18.2; tanh-8 → abstains; GOLD-WTI linear holdout +12.0 → −0.5; tanh-8 → abstains. The saliency table shows the policy does use them (GOLD-WTI |∂J/∂E_gauss| is the second-largest input gradient), and that use does not transfer. This matches Sprint 09's A4−A2 and B3−B2 result from the opposite direction.

### 3.5 Decision loss vs forecast loss (H_DECISION_LOSS: same direction, different sizing, same outcome)
The forecast-then-threshold ridge trades when |ŷ(15 s)| > half-spread. On BTC-ETH it fires on 6,306 of 11,520 validation samples at ×1 and loses −0.3¢/window; at ×2 it loses −1.9; under fee stress it loses −5.0 and −0.8. The P&L-trained policy on the same features keeps a position for 7–10 steps instead of 3 and loses less per cycle, and under fee it switches itself off while the forecaster at ×1 keeps firing 348 times and loses −5.0¢/window. The gradient of P&L sizes and times the same linear signal better than a threshold on a forecast, and the signal itself is too small for either to profit.

### 3.6 Adjoint and saliency
λ_t = ∂J/∂q_t along the realized path has median |λ| ≈ 1–2¢ and exceeds half-spread (0.5¢) on 77–82% of steps. This is a hindsight quantity (it contains Δm_{t+1}); it says the shadow price of one unit of inventory is large almost everywhere, which is exactly why the clairvoyant bound (88–92¢/window in Sprint 09) is large. The policy cannot collect it because the sign of λ_t is not predictable from x_t: the zero-fee policies' causal P&L is ≈ 0 ± 8¢/window while |λ| averages 2¢ per 5 s step.

Saliency (mean |∂J/∂x| over validation+holdout windows, own+partner+geom linear): BTC-ETH and BTC-DOGE load on imbB and dA_open/dB_open; GOLD-WTI and GOLD-SILVER load on dB_open, E_gauss, E_resid, ∇E_B. None of these loadings produced transfer.

### 3.7 Training dynamics and the surrogate–simulator gap
Best validation epoch is often late (173–298 of 300) for zero-fee configs and early (15–97) where the policy overfits. The surrogate overstates the exact replay: BTC-ETH own h0 best validation surrogate +8.7¢/window vs exact replay +0.6¢/window. The gap comes from fractional continuous positions, same-step execution at mid ± s/2, and no hysteresis band in the surrogate. Any future version should include the band and the next-sample delay inside the differentiable loss (straight-through estimator) so that validation selection is done on the quantity that is actually scored.

### 3.8 A flaw in one pre-registered null, documented
The label-shuffle null permutes Δm within each training window, which preserves each window's mean drift; features like dB_open therefore still predict the remaining drift in shuffled data. That null traded heavily (mean |q| 0.85–0.94) and scored +6.5¢/window on BTC-ETH validation, more than any real policy. It is not a clean null; the sign-flip null added in `analyze.py` is the one used for the ledger. The plan file is frozen and unchanged; this deviation is recorded here.

## 4. Verdict

- NO_CANDIDATE. No configuration clears zero on both validation and holdout after fee; the only holdout positives (GOLD-WTI, 4 windows, one afternoon) fail validation and are matched by a destroyed-label null.
- ABSTENTION_LEARNED is the main empirical result: under an illustrative fee, the validation-selected P&L-trained policy is P0 in 28/36 configurations, and for own-leg crypto features the training gradient itself converges to zero exposure. This is the hidden thing in backprop that is usable now: **training a cost-aware policy and reading (train_last, val_last, mean |q|) is a one-run test of a feature set.** (≈0, ≈0) means nothing is there; (+, −) means the features manufacture in-sample mirages; only (+, +) with exact-replay transfer earns a threshold study. In this sprint no feature set reached the third case.
- GEOMETRY_IN_POLICY_NO_GAIN: coupled-geometry features reduce out-of-sample P&L or drive abstention.
- The adjoint λ_t is large in hindsight and unpredictable in foresight; the opportunity exists (clairvoyant bound) and the features do not reach it.

## 5. What would move this forward (suggestions, not done)

1. Data: the forward collector on fresh multi-day commodity and crypto tape (user will start it when markets open). Commodities currently have 23 windows of one day; nothing about GOLD-WTI can be settled on that.
2. Put the exact simulator inside the loss (straight-through band, next-sample execution) so validation selection and scoring agree.
3. Features at the 1 s or event level rather than the 5 s grid; the order-book deltas (13M, in memory, not archived) would be the natural input if the collector keeps them.
4. Train one policy across pairs with pair embeddings rather than one per pair, to raise effective sample size.
5. Keep the abstention test as the first step of every future feature set: train the cost-aware policy, read mean |q|; proceed to threshold studies only when it trades.

## 6. Files

`analysis_plan.yaml`, `plan_freeze.json`, `policy_scores.csv`, `paper_trades.parquet`, `training_history.csv`, `training_curve_summary.csv`, `bootstrap_by_window.csv`, `learned_weights.json`, `sign_flip_null.csv`, `forecast_threshold_baseline.csv`, `input_saliency.csv`, `saliency_summary.csv`, `adjoint_paths.csv`, `adjoint_summary.csv`, `candidate_ledger.csv`, `plots/` (policy_pnl_*, adjoint_*, saliency_*). Code: `sprint09b/adjoint_policy.py`, `sprint09b/run_sprint09b.py`, `sprint09b/analyze.py`, `sprint09b/forecast_baseline.py`; tests `tests/test_sprint09b.py`.
