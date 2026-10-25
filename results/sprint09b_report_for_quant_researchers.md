# Backpropagation through the trading decision on Kalshi 15-minute Up/Down markets

**Sprint 09B · research report for external review · 25 October 2026**

Status labels that apply to everything below: PILOT_ONLY_REUSED_ARCHIVE · FEE_UNVERIFIED · NOT_REAL_FILL · TRADEABILITY_NOT_ESTABLISHED. No orders were placed. Nothing here is a claim of profitability.

---

## 0. Summary for the impatient

- We trained a 10–14-parameter recurrent policy by back-propagation through time on cost-aware P&L (Moody–Saffell / deep-hedging style), froze it on validation, and replayed it on an exact discrete YES/NO simulator on later data. 80 trainings across BTC-ETH, BTC-DOGE, GOLD-SILVER, GOLD-WTI.
- Under an illustrative taker fee, the validation-selected policy is **zero exposure in 28 of 36 configurations**. For own-leg crypto features the training gradient itself converges to zero. For partner-leg and geometry features, training ends positive in-sample (median +4.1¢/window) and validation ends negative (94% of configs): the gradient finds in-sample mirages and the validation day rejects them.
- At zero fee the policies trade; validation gains sit inside their bootstrap intervals and the crypto holdout day is negative.
- **Four strategy families were positive on both held-out halves of the archive and are listed in §6 with exact specifications and an external-validation protocol.** None clears our pre-registered bar (both halves clear zero after fee, with a destroyed-label null below the real policy). The GOLD-WTI family in particular rests on 8 windows of one afternoon and is matched by a sign-flip null. We are sending them out because the specifications are frozen and cheap to test on fresh tape, not because we believe them.
- The reusable methodological result: training a cost-aware policy and reading (train surrogate, validation surrogate, mean |q|) is a one-run test of a feature set that replaces a threshold scan and names which inputs manufacture in-sample edge.

---

## 1. Market and data

**Instrument.** Kalshi 15-minute Up/Down binary contracts (series per asset: BTC, ETH, DOGE, GOLD, SILVER, WTI, …). A window opens at a quarter-hour, settles 15 minutes later at 0 or 1 against the reference price. YES ask = 1 − best NO bid; NO ask = 1 − best YES bid. Prices in cents; typical top-of-book spread 1–2¢.

**Archive used (all previously collected by us; no new data were purchased or downloaded for this sprint).**

| domain | pairs | dates | windows | split |
|---|---|---|---|---|
| crypto | BTC-ETH, BTC-DOGE | 2026-10-13, 14, 15 / 10-18 / 10-19 | 48 / 64 / 7 | train / validation / holdout by date |
| commodities | GOLD-SILVER, GOLD-WTI | 2026-10-22 only | 15 / 4 / 4 | train / validation / holdout by time of day |

States are an as-of join of both legs' top-of-book on a 5 s grid (both quotes ≤ 2 s old, two-sided), inherited from Sprint 09's `asof_states.parquet`. Crypto rows: 21,420 per pair (19,807 eligible). Commodity rows: ≈4,000 per pair. The commodity book archive consists of WebSocket snapshots plus 1 s interval dumps; the ~13M intra-second deltas were applied in memory and are not archived, so commodity quote timestamps are dump times (exchange update age unknown). Crypto rows carry `received_ms` and sequence numbers.

**Fees.** Kalshi's fee page was unreachable behind a challenge during this work. All "fee stress" numbers use an illustrative taker formula, round-up(0.07 · P · (1−P)) per contract side, taken from a public July 2026 schedule. Treat it as a stress test, not as the true fee. Zero-fee numbers are an upper bound.

**Fills.** Every paper trade is marked NOT_REAL_FILL. Entries are at the displayed ask of the side being bought, exits at the displayed bid, both taken at the next valid 5 s sample after the decision. No queue, no size, no impact.

---

## 2. Why this sprint exists: what the predecessor got wrong

Sprints 08H-R, 08H.1 and 09 built a coupled-geometry picture of the gold–silver (and BTC–ETH) pairs: MAD-standardized joint moves z, a Gaussian "tension" energy E = ½(z−μ)ᵀΣ⁻¹(z−μ), its gradient ∇E = Σ⁻¹(z−μ), a cost-weighted "repair" δ*, and a clairvoyant greedy round-trip selector as an opportunity ceiling. Verdicts were consistent: no incremental directional or capacity signal from the geometry beyond ordinary own-leg and partner-leg state; every causal hand-threshold policy net negative even at zero fee; clairvoyant ceiling ≈ 88–92¢/window at N = 3 round trips.

The original research idea was "backpropagation as a trading principle": (1) back-propagate geometric inconsistency, (2) back-propagate realized cost-aware P&L through a differentiable policy, (3) back-propagate missed opportunity through a hindsight value function. Sprint 09 executed only a shadow of (1): ∇E of a *fixed* Gaussian is a linear re-labeling of the state, so the pre-registered null (NO_INFORMATION_BEYOND_STATE) was true by construction. No parameter was ever updated by the gradient of a trading objective; the differentiable policy was gated off; the "DP" was greedy. Sprint 09B does (2) properly, with (3) retained as the adjoint diagnostic.

---

## 3. Method

### 3.1 Policy
q_t ∈ (−1, 1) is the target inventory in the B leg (silver, WTI, ETH, DOGE; the "A" leg is the partner used only as a feature).

q_t = tanh( W · [x_t, q_{t−1}] + b )  (linear, 10–14 parameters)
q_t = tanh( w₂ · tanh( W₁ · [x_t, q_{t−1}] + b₁ ) + b₂ )  (tanh-8, ≈ 100 parameters)

q_{t−1} as an input gives the chain rule a path through time (BPTT) and lets the policy learn hysteresis (a no-trade band) by itself. q is forced to 0 in the final 60 s of the window and held at its previous value through invalid samples.

Features (train-split robust standardization, clipped ±6):
- own: dB_open (mid change since window open), rB5 (5 s return), spreadB, imbB (top-of-book imbalance), phase/900, T_left/900
- partner: dA_open, rA5, spreadA
- geometry: E_gauss, ∂E/∂z_A, ∂E/∂z_B, E_resid (from Sprint 09, parameters fit on the training split)

### 3.2 Objective (surrogate, differentiable)
Per window,

J = Σ_t q_t Δm_{t+1} − Σ_t (s_t/2 + fee_t) |q_t − q_{t−1}| − λ Σ_t q_t²,  plus a forced flatten at the last sample.

Δm is the next-step mid change of leg B, s_t its spread, fee_t the illustrative fee at the current mid (zero in ZERO_FEE_BOUND). Adam, lr 0.01, 300 epochs, gradient-norm clip 5, seed 7. Early stopping on the validation surrogate only; the holdout split is scored once with the frozen policy.

### 3.3 Exact replay (never differentiated)
Hysteresis band on the frozen q: cross +0.5 → buy YES at the YES ask at the next valid sample; cross −0.5 → buy NO at the NO ask; fall inside ±0.2 → sell at the bid at the next valid sample. One position per contract, ≤ 5 round trips per window, no entries in the final 60 s, a position still open at the last valid sample is closed there (`forced_close`). Fee charged per side under stress.

### 3.4 Baselines and nulls on identical data
- P0 (no trades).
- P1 fixed rules: momentum / fade on rB5 ≥ 1¢, hold 3 steps, ≤ 3 round trips (Sprint 09's P1).
- Forecast-then-threshold: ridge (α = 1) on the same own+partner features predicting the 15 s mid change, trade when |ŷ| > half-spread (+fee), ×1 and ×2 multipliers.
- Label-shuffle null: Δm permuted within each training window. **Flawed** (preserves each window's mean drift, so dB_open still predicts remaining drift); reported for transparency, not used in the ledger.
- Sign-flip null: each training window's Δm multiplied by an independent ±1; three seeds. This destroys feature→drift relations while preserving dynamics and is the null used in the ledger.

### 3.5 Diagnostics
Adjoint λ_t = ∂J/∂q_t along the realized rollout (shadow price of inventory), input saliency ∂J/∂x_t, training curves (train and validation surrogate per epoch), learned weights, bias-removed replay.

### 3.6 Statistics
Cluster bootstrap over windows (quarter-hours), 2,000 resamples, seed 7, 95% percentile intervals on ¢ per window. Pre-registered acceptance rule (unchanged from earlier sprints): a candidate must clear zero on both validation and holdout after fee, with the destroyed-label null below it.

### 3.7 Tests
`tests/test_sprint09b.py`, 11 tests, all pass: autograd adjoint equals central finite differences; q ≡ 0 gives J = 0; turnover is penalized; the policy learns to hold on a trending synthetic and abstains when the spread is 50¢; replay accounting (one YES cycle, entry at next valid sample, pnl = bid − ask); no entries in the final 60 s; fixed rule pays the spread; fee stress ordering. The tests caught one real bug before the main run: positions still open at the last valid sample were silently dropped; both replayers now force-close them.

---

## 4. Results

### 4.1 Abstention under cost — the headline

Fee-stress configurations whose frozen policy makes zero trades on validation and holdout:

| pair | configs | abstaining | mean \|q\| | train surrogate at epoch 300 (¢/w) | validation surrogate at epoch 300 (¢/w) |
|---|---|---|---|---|---|
| BTC-ETH | 9 | 9 | 0.0003–0.03 | −0.5 to +13.1 | −0.0 to −15.4 |
| BTC-DOGE | 9 | 8 | ≤ 0.03 | −0.2 to +9.0 | −0.0 to −3.1 |
| GOLD-SILVER | 9 | 8 | 0.01–0.25 | −0.0 to +33.8 | −2.0 to −38.9 |
| GOLD-WTI | 9 | 3 | 0.56–0.96 when trading | −13.5 to +23.4 | −19.3 to +8.5 |

Two mechanisms produce the abstention:

1. **Own-leg features, crypto (6 configs).** The training surrogate itself ends at ≈ 0 (−0.46 to +0.03¢/window). The gradient finds nothing to fit, in-sample or out.
2. **Partner and geometry features (30 configs).** Training ends positive in 69% of configs (median +4.1¢/window; BTC-ETH own+partner+geom +13.1; GOLD-SILVER geom tanh-8 +33.8) while validation ends negative in 94% (median −2.7; BTC-ETH geom −15.4). Validation peaks near −0.4¢ around epochs 30–70 where weights are still ≈ 0, then falls as training-day patterns are fit. Early stopping selects the near-zero epoch (median best validation −0.34¢/window ≈ P0).

For BTC-ETH own+partner under fee the frozen linear weights are all |w| < 0.03; the policy is P0 to three decimals.

### 4.2 Zero fee, crypto — validation gains stay inside their intervals; holdout day negative

BTC-ETH, exact replay, ¢ per window [95% cluster-bootstrap CI]:

| policy | validation (64 w) | holdout (7 w) |
|---|---|---|
| ADJOINT own, linear | +0.6 [−6.9, 8.1] | −13.3 [−35.4, 8.4] |
| ADJOINT own, tanh-8 | +2.9 [−4.5, 10.3] | −15.6 [−41.0, 9.4] |
| ADJOINT own+partner, linear | +0.9 [−7.3, 8.5] | −10.8 [−34.5, 12.8] |
| ADJOINT own+partner, tanh-8 | +2.4 [−6.6, 10.9] | −1.3 [−32.7, 31.6] |
| ADJOINT own+partner+geom, linear | −0.7 [−11.5, 9.4] | −18.2 [−41.6, 5.9] |
| ADJOINT own+partner+geom, tanh-8 | abstains | abstains |
| P1 momentum | −2.4 | −2.0 |
| P1 fade | −3.8 | −4.0 |
| forecast→threshold ridge ×1 | −0.3 [−3.4, 2.6] | −10.9 [−19.1, −0.7] |

What the BTC-ETH own+partner linear policy learned (standardized inputs): w(imbB) = +0.63, w(dB_open) = −0.78, w(dA_open) = +0.50, w(spreadA) = +0.31, w(q_prev) = +1.10, bias ≈ 0. Read: follow the book imbalance, fade the move from the open, lean with the partner's move, with hysteresis. The ridge forecaster on the same features learned the same signs (imbB +0.0071, dB_open −0.0052, dA_open +0.0041). Decision loss and forecast loss agreed on direction; the P&L-trained policy held 7–10 steps instead of 3 and lost less per cycle, and under fee it switched itself off while the forecaster at ×1 kept firing 348 times for −5.0¢/window. The signal is too small for either.

BTC-DOGE zero fee: every validation CI spans zero; holdout (7 windows) CIs span zero; three configs are positive on both halves and are listed in §6.

### 4.3 GOLD-WTI — holdout positives that fail validation and lose to the null

| config | validation (4 w) | holdout (4 w) | sign-flip null, holdout (best of 3 seeds) |
|---|---|---|---|
| own linear, zero fee | +12.0 [−11.3, 33.1] | +14.7 [5.9, 23.3] | +32.5 |
| own+partner linear, fee stress | +8.5 [−13.9, 31.0] | +14.2 [7.3, 22.0] | +39.5 |
| own+partner tanh-8, zero fee | +12.5 [−9.4, 34.5] | +24.0 [13.4, 38.5] | +32.5 |
| own+partner linear λ = 0.001, zero fee | +22.3 [−2.6, 42.3] | +12.2 [1.9, 28.0] | +32.5 |
| own+partner+geom linear, zero fee | +13.5 [−5.4, 32.5] | −0.5 [−13.1, 12.2] | +32.5 |
| P1 momentum, zero fee | −6.3 | −1.8 | — |
| forecast→threshold ridge ×1, zero fee | −5.8 [−17.3, 5.0] | +9.0 [6.3, 11.3] | — |

Learned weights (own linear, zero fee): w(dB_open) = +1.40, w(q_prev) = +2.02, bias = +0.53, all other |w| < 0.07. The realized trade list (own+partner linear, fee stress) makes the mechanism explicit:

```
window            side  entry_step exit_step  ask    bid    pnl¢
19:00 (val)       YES     1   167   0.53  0.810  +24.0
19:15 (val)       YES     2   128   0.49  0.320  −21.0
19:15 (val)       NO    130   168   0.82  0.986  +13.6
19:30 (val)       YES     2   143   0.63  0.270  −40.0
19:30 (val)       NO    144   162   0.73  0.956  +19.6  forced_close
19:45 (val)       YES     2   167   0.58  0.989  +37.9
20:00 (holdout)   YES     1    85   0.60  0.450  −19.0
20:00 (holdout)   NO     86   164   0.61  0.988  +34.8  forced_close
20:15 (holdout)   YES     3    96   0.49  0.470   −6.0
20:15 (holdout)   NO     98   161   0.65  0.997  +31.7  forced_close
20:30 (holdout)   YES     2    67   0.49  0.330  −20.0
20:30 (holdout)   NO     69   167   0.67  0.944  +24.4
20:45 (holdout)   YES     2   167   0.56  0.710  +11.0
```

The policy buys YES within 10 s of every open (the bias), holds through hysteresis, and when that leg has lost enough it flips to NO at 0.61–0.82 and rides the contract into settlement at 0.94–0.997. Eight windows, one afternoon; the sign-flip null earns more on the same holdout windows. The §6 entry states the hypothesis this actually encodes.

### 4.4 Geometry features inside the policy
Adding E_gauss, ∇E, E_resid: BTC-ETH linear validation +0.9 → −0.7, holdout −10.8 → −18.2; tanh-8 → abstains. GOLD-WTI linear holdout +12.0 → −0.5; tanh-8 → abstains. Saliency shows the policy does use them (GOLD-WTI |∂J/∂E_gauss| is the second-largest input gradient) and that use does not transfer. This matches Sprints 08H-R/08H.1/09 from the opposite direction and, via §4.1, identifies the geometry inputs as the main source of in-sample mirage.

### 4.5 Adjoint and saliency
λ_t = ∂J/∂q_t has median |λ| 1–2¢ and exceeds half-spread (0.5¢) on 77–82% of 5 s steps across all four pairs. λ_t is a hindsight quantity (it contains Δm_{t+1}); its size is the same fact the clairvoyant ceiling expresses, and its sign is what the features fail to predict: the zero-fee policies' causal P&L is ≈ 0 ± 8¢/window while |λ| averages 2¢ per step.

### 4.6 Surrogate–simulator gap
The surrogate overstates exact replay (BTC-ETH own linear: best validation surrogate +8.7¢/window vs exact replay +0.6). Sources: fractional continuous positions, same-step execution at mid ± s/2, no hysteresis band. Any continuation should put the band and the next-sample delay inside the differentiable loss (straight-through estimator) so that validation selection is done on the scored quantity.

---

## 5. Verdicts

- NO_CANDIDATE under the pre-registered rule.
- ABSTENTION_LEARNED: under cost the validation-selected policy is P0 in 28/36 configurations; own-leg crypto features converge to zero exposure in-sample as well.
- GEOMETRY_IN_POLICY_NO_GAIN: coupled-geometry features lower out-of-sample P&L or drive abstention, and are the main source of in-sample over-fit.
- Decision loss and forecast loss agree on direction on BTC-ETH; neither transfers.
- Hindsight shadow price of inventory is large everywhere and causally unpredictable from these features at the 5 s grid.

---

## 6. Strategies that were positive on both held-out halves and now need external data

Everything in this section failed at least one leg of the pre-registered rule. Specifications are frozen as written; parameters come from `learned_weights.json` and `policy_scores.csv`. For each we state the mechanism, the exact rule, what was observed, why it is unproven, and the specific test that would settle it.

### 6.1 "Open-side YES, flip to NO and ride into settlement" — GOLD-WTI (WTI leg)

**Observed.** Positive on both halves in 7 of 10 WTI configurations, zero fee and fee stress: validation +5.8 to +22.3¢/window (all CIs span zero), holdout +0.2 to +24.0¢/window (four CIs above zero). 13 round trips total.

**Mechanism encoded (from the trade list).** Enter YES at the open at ~0.49–0.63. If the mid falls far enough that q crosses the band, exit and buy NO at 0.61–0.82, then hold to settlement. The second leg is what pays: it buys the ~65–80% side mid-window and collects 0.94–0.997. The hypothesis this tests is therefore **"the favored side of a WTI 15-minute contract is underpriced in the second half of the window"** (price-to-settlement convergence is faster than the market prices). Under fair pricing a 0.70 NO contract pays 0.70 in expectation; the archive showed 5/5 such legs settling in the money. The opening YES leg is a coin flip with a cost and contributed −33¢ net over 8 legs; the 5 NO legs contributed +124¢.

**Frozen rule for external test (linear own+partner, fee-stress weights).** Standardize features with the saved train-split scales; q_t = tanh(0.973·dB_open − 0.016·rB5 + 0.001·spreadB + 0.083·imbB − 0.221·phase + 0.200·T_left + 0.289·dA_open − 0.002·rA5 − 0.060·spreadA + 1.780·q_{t−1} + 0.186); band ±0.5 on, ±0.2 off; no entry in last 60 s; next-sample execution at displayed ask/bid.

**A simpler two-parameter version to test alongside (not run here, derived from the weights).** Ignore the opening leg. At each 5 s sample with T_left ∈ [120 s, 450 s]: if the mid of either side is in [0.62, 0.85] and dB_open has the same sign for ≥ 60 s, buy that side at the ask and hold to settlement. Report win rate vs. entry price (calibration), P&L per contract after real fees, and settlement-price slippage.

**Why unproven.** 8 windows of one afternoon (22 Oct 2026, 19:00–21:00 UTC); a sign-flip-trained null earned +32.5¢/window on the same holdout windows; validation CIs span zero; a single trending afternoon in WTI explains everything observed.

**External validation.** ≥ 20 trading days of WTI 15-minute windows (≈ 500+ windows), plus GOLD and SILVER as controls. Pre-register: calibration curve of settlement vs. mid at T_left = 300 s binned by 0.05; the test statistic is realized frequency minus mid in the 0.62–0.85 bin after fee, cluster-bootstrapped by day. Edge exists only if the lower CI is above the round-trip cost. Expect zero: this is the market being well calibrated mid-window, and the 5/5 result is consistent with p ≈ 0.7 by chance (0.7⁵ ≈ 0.17).

### 6.2 "Imbalance-follow with partner lean, inventory hysteresis" — BTC-DOGE (DOGE leg), zero fee

**Observed.** Three configurations positive on both halves at zero fee: own+partner linear +3.9 [−6.1, 14.8] / +19.4 [−2.6, 49.3]; own+partner tanh-8 +3.7 [−4.9, 13.3] / +7.1 [−8.3, 22.0]; own+partner+geom linear +0.8 [−5.9, 8.1] / +25.4 [−1.0, 60.1]. 273–299 round trips on validation, 35 on holdout. **All abstain under fee stress** (the mechanism does not cover a 1–2¢ fee per side).

**Frozen rule (own+partner linear, zero fee).** q_t = tanh(−1.695·dB_open − 0.064·rB5 + 0.034·spreadB + 1.269·imbB + 0.589·phase − 0.610·T_left − 1.118·dA_open + 0.276·rA5 + 0.829·spreadA + 1.761·q_{t−1} + 1.060); band ±0.5/±0.2; median hold 7–9 steps (35–45 s). The bias of +1.06 is a standing long-YES tilt; with the bias zeroed, validation goes from +250¢ to −70¢ and holdout from +136¢ to +38¢, so the tilt carries most of the result.

**Why unproven.** Validation CIs span zero; holdout is 7 windows of one day; a sign-flip null seed earned +5.9¢/window on the same holdout; the strategy only exists at zero fee and we have not verified Kalshi's fee or any maker rebate.

**External validation.** Only worth running if there is a maker path (resting orders at the bid) that removes the spread crossing; then re-fit on ≥ 10 days of DOGE tape and score ≥ 5 later days. The cheap first check is a fee-inclusive replay against the actual fee schedule: if the policy abstains, stop.

### 6.3 Fixed momentum rule on SILVER (Sprint 09 P1 form), zero fee only

**Observed.** P1 momentum (enter with the sign of the 5 s return when |rB5| ≥ 1¢, hold 3 steps, ≤ 3 round trips): validation +3.0¢/window (12 cycles), holdout +6.0¢/window (12 cycles). Under fee stress −8.5 / −6.0. The fade branch is −9.5 / −13.3.

**Why unproven.** 24 cycles on 8 windows of one day; net negative in Sprints 08H.1 and 09 on the same tape with slightly different delays and cycle caps (which is exactly the kind of neighbouring-cell comparison our rules forbid as evidence); loses under any realistic fee. Listed because it is the simplest rule in the set and costs nothing to re-score.

**External validation.** Score the frozen rule unchanged on every SILVER window of ≥ 10 days with real fees. Any positive result would then have to beat the sign-flip null built the same way.

### 6.4 Forecast-then-threshold ridge on WTI, zero fee, ×1 threshold

**Observed.** Validation −5.8 [−17.3, 5.0], holdout +9.0 [6.3, 11.3]. Positive on one half only; included because it is the forecast-loss counterpart of §6.1 on the same features and the same afternoon, and the two should be validated together. Fee stress: −2.7 / −2.1.

**Status.** Not a candidate. If §6.1's calibration test comes back positive, re-fit this ridge on the external data as the baseline it must beat.

### What is *not* on this list
Every BTC-ETH configuration (holdout negative at zero fee, abstains under fee). Every GOLD-SILVER adjoint configuration (holdout negative or abstaining). Every fee-stress crypto configuration except the one BTC-DOGE geometry model, which loses −11.5¢/window on holdout. All geometry-feature policies.

---

## 7. What we think is the durable finding

A policy whose only parameters are set by ∂(P&L after cost)/∂θ, with the chain rule running through its own inventory, gives a one-run answer about a feature set that a threshold scan gives only after hundreds of cells and a multiple-comparisons problem:

- (train ≈ 0, validation ≈ 0, mean |q| ≈ 0): nothing to find. Own-leg crypto features at 5 s.
- (train > 0, validation < 0): the features manufacture in-sample edge. Partner-leg and coupled-geometry features.
- (train > 0, validation > 0, exact-replay transfer on a later period): earns a threshold study and a null. Nothing reached this case.

The adjoint λ_t adds the complementary fact: the hindsight value of inventory is 2–4× the half-spread almost everywhere, so the opportunity ceiling is real and the failure is predictive, not structural. Whatever predicts the sign of Δm over 5–15 s at Kalshi's top of book would be worth a great deal; nothing in own-leg, partner-leg, or coupled-geometry state at the 5 s grid does.

---

## 8. Caveats colleagues should weigh

1. Commodities: one day, 23 windows per pair, split by time of day. Nothing about GOLD-WTI or GOLD-SILVER is settled by this archive.
2. Crypto holdout: 7 windows (one day). Validation (64 windows) was used for early stopping across 300 epochs, so validation numbers carry selection bias; holdout numbers do not, and are negative or inside their intervals.
3. Fees unverified; fills not real; sizes ignored; no latency model beyond "next 5 s sample"; commodity quote ages unknown.
4. The label-shuffle null in the frozen plan is flawed (preserves window drift); we replaced it with the sign-flip null in the ledger and left the plan file unchanged.
5. The surrogate overstates the exact replay (§4.6).
6. The forecast-threshold baseline fires on >50% of samples at ×1 because the ridge's predicted magnitude often exceeds the 0.5¢ half-spread; its P&L is the right comparison but its signal count is not a measure of confidence.

---

## 9. Reproduction

Environment: `/Users/nashe/Kalshi/crypto/data_download/.venv` (torch 2.14.1, pandas 3.1, numpy, pyarrow, matplotlib). Seed 7 throughout.

```
cd /Users/nashe/Kalshi
crypto/data_download/.venv/bin/python -m unittest tests.test_sprint09b          # 11 OK
MPLCONFIGDIR=/tmp/mpl crypto/data_download/.venv/bin/python sprint09b/run_sprint09b.py      # ~10 min
crypto/data_download/.venv/bin/python sprint09b/analyze.py                      # ~4 min
crypto/data_download/.venv/bin/python sprint09b/forecast_baseline.py
```

Plan: `configs/sprint09b_analysis_plan.yaml`, frozen sha256 `0df3f8846a6e6d57b826decc3e185c8fb1adc0db4e331f96412847bcd442bc93`. Outputs in `experiments/sprint09b_adjoint_policy_20261025/`: `policy_scores.csv`, `bootstrap_by_window.csv`, `paper_trades.parquet`, `learned_weights.json`, `sign_flip_null.csv`, `forecast_threshold_baseline.csv`, `training_history.csv`, `training_curve_summary.csv`, `input_saliency.csv`, `saliency_summary.csv`, `adjoint_paths.csv`, `adjoint_summary.csv`, `candidate_ledger.csv`, `findings.md`, `plots/`.

Lineage (same archive, same rules): 08H-R retrospective coupled geometry; 08H.1 distributional outcomes and multi-cycle harvesting (clairvoyant 18/49/70¢ at 1/3/5 cycles zero fee; all causal policies negative); 09 reverse sensitivity and opportunity back-prop (gradient checks pass to 1e−11; A4−A2 and B3−B2 ≈ 0 or negative; clairvoyant N = 3 ≈ 88–92¢/window).

---

## 10. Suggested next steps

1. Collect fresh multi-day tape for WTI, GOLD, SILVER, BTC, ETH, DOGE (the forward collector exists; it is stopped until markets open).
2. Run the §6.1 calibration test first; it is the only item with a mechanism we can state in one sentence and it costs one afternoon.
3. Obtain the actual fee schedule and maker/taker treatment before any zero-fee result is discussed further.
4. Put the exact simulator inside the differentiable loss; train one policy across pairs with a pair embedding to raise effective sample size.
5. Keep the cost-aware policy training as the first gate on any new feature family; proceed to threshold studies only when it trades out of sample.
