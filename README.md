# Sprint 09B — backpropagation through the trading decision

A tiny recurrent policy trained by back-propagation through time on cost-aware P&L, frozen on a validation split, then replayed on an exact discrete YES/NO simulator. No orders are placed. No market data is included in this repository.

The code is the Sprint 09B experiment (Kalshi 15-minute Up/Down). Point `--states` at a parquet you build from your own tape.

The run on the original archive is in `results/`. Start with `results/sprint09b_report_for_quant_researchers.md`. That folder is output, not input; do not pass it to `--states`.

## What it does

Policy, inventory in (−1, 1):

```
q_t = tanh(W · [x_t, q_{t-1}] + b)          # linear, 10–14 parameters
q_t = tanh(w2 · tanh(W1 · [x_t, q_{t-1}] + b1) + b2)   # tanh hidden size 8
```

`q_{t-1}` is an input, so the chain rule runs through time and the policy can learn a no-trade band.

Surrogate, per window (the thing that is differentiated):

```
J = Σ q_t · Δm_{t+1}
  − Σ (spread_t/2 + fee_t) · |q_t − q_{t-1}|
  − λ · Σ q_t²
```

plus a forced flatten at the last sample, and `q = 0` in the final 60 seconds. Adam, learning rate 0.01, 300 epochs, seed 7. The checkpoint is the epoch with the best validation surrogate. The holdout split is scored once.

Exact replay (never differentiated): `q` crosses ±0.5 to enter at the ask of that side on the next valid row, falls inside ±0.2 to exit at the bid on the next valid row, at most 5 round trips per window, no entry in the final 60 seconds. Fills are labeled `NOT_REAL_FILL`. Fee modes are `ZERO_FEE_BOUND` and `ILLUSTRATIVE_FEE_STRESS` (`ceil(0.07 · P · (1−P))` per side). The fee is a stress test, not a verified Kalshi schedule.

Baselines on the same file: do nothing; fixed momentum and fade on the 5-second return; a ridge forecast of the 15-second mid change traded when `|ŷ|` exceeds half the spread. Diagnostics: cluster bootstrap over windows, learned weights, a sign-flip training null, input saliency `∂J/∂x`, and the adjoint `λ_t = ∂J/∂q_t`.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest tests.test_sprint09b
```

Tests use synthetic windows and do not need the parquet.

## Run

```
python run_sprint09b.py --states /path/to/states.parquet --out outputs
python analyze.py       --states /path/to/states.parquet --out outputs
python forecast_baseline.py --states /path/to/states.parquet --out outputs
```

`--states` and `--out` can also be set with `S09B_STATES` and `S09B_OUT`. `analyze.py` reads `policy_scores.csv`, `paper_trades.parquet`, and `training_history.csv` from `--out`, so run it after `run_sprint09b.py`.

Every pair present in the file is trained. A pair with no `discovery` rows is skipped.

Outputs land in `--out`: `policy_scores.csv`, `paper_trades.parquet`, `training_history.csv`, `bootstrap_by_window.csv`, `learned_weights.json`, `sign_flip_null.csv`, `forecast_threshold_baseline.csv`, adjoint and saliency tables, and `plots/`.

## Data you supply

One parquet (or anything pandas reads as parquet) of **decision states**, one row per pair per window per time step. The archive this experiment was written against was a 5-second grid; the code treats consecutive rows inside a window as one step, so the row spacing is the decision horizon. `Δm` is the next row's mid minus this row's mid.

### Required columns

| column | meaning |
|---|---|
| `pair` | string id of the pair, e.g. `BTC-ETH`. Leg B is the traded leg. |
| `window_id` | id of the 15-minute contract window. Rows of one window stay together. |
| `t_ns` | time of the row, nanoseconds, increasing within a window. |
| `split` | `discovery` (train), `validation` (early stopping), or `holdout` (scored once). |
| `phase` | seconds since the window opened. The code divides by 900, so a 15-minute window is the intended scale. |
| `T_left` | seconds until the window settles. No new entries when this is 60 or below. |
| `level_ok` | true when both legs have a usable two-sided quote. Invalid rows are not traded. |
| `pB` | mid of the traded leg, in dollars (0–1). |
| `bB`, `aB` | best bid and best ask of the traded leg's YES book, in dollars. NO ask is `1 − bB`, NO bid is `1 − aB`. |
| `spreadB` | `aB − bB`. |
| `dB_open` | `pB` minus the mid at the open of this window. |
| `rB5` | own-leg return over the previous step (named for a 5-second step). |
| `imbB` | top-of-book imbalance of the traded leg, about −1 to 1. |

### Optional columns

Filled with 0 when absent. Partner columns are leg A (the other asset in the pair). Geometry columns are whatever coupled-state features you want the ablation to see; the original run used a Gaussian energy and its gradient.

| column | meaning in the original run |
|---|---|
| `dA_open`, `rA5`, `spreadA` | partner leg, same definitions as leg B |
| `E_gauss` | Gaussian tension `½(z−μ)ᵀΣ⁻¹(z−μ)` of the joint standardized move |
| `grad_zA_gauss`, `grad_zB_gauss` | gradient of that energy in each leg |
| `E_resid` | residual-regression energy |
| `domain` | label copied into the score table, e.g. `crypto` or `commodities` |
| `date` | calendar date, copied onto each paper trade |
| `pA` | partner mid; not required by the trainer |

Feature scales (median and MAD) are fit on `discovery` rows with `level_ok` true, per pair. Do not put holdout information into those rows.

### What the original archive contained

Not shipped. It was an as-of join of Kalshi top-of-book for 15-minute Up/Down contracts:

- Crypto pairs `BTC-ETH` and `BTC-DOGE`, dates 2026-10-13, 14, 15 (discovery), 2026-10-18 (validation), 2026-10-19 (holdout). 48 / 64 / 7 windows. About 21,420 rows per pair on a 5-second grid, both quotes at most 2 seconds old.
- Commodity pairs `GOLD-SILVER` and `GOLD-WTI`, 2026-10-22 only, 15 / 4 / 4 windows split by time of day.

Build the same columns from whatever tape you have. Keep `discovery` strictly earlier than `validation`, and both strictly earlier than `holdout`.

## Notes

- `analysis_plan.yaml` is the frozen experiment plan (hypotheses, fee modes, bands, seeds). `run_sprint09b.py` copies it into `--out` and records its hash.
- Seed is 7. Cluster bootstrap over windows uses seed 7 and 2,000 resamples.
- This repository does not place orders and does not read API keys.
