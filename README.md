[中文](README.zh.md) | **English**

# Small-Cap Strategy Replication

A faithful replication of Debon Securities' 2022-04-20 report *Quant Small-Cap Series Part 1: A First Look at Small-Cap Strategies* (《金工小市值专题之一：小市值策略初探》). The report is the baseline; out-of-sample extensions and variants are layered on top (in progress, see [`PROJECT.md`](PROJECT.md)).

> All internal docs (`PROJECT.md`, `project_enhance.md`, `grill.md`, `grill_enhance.md`) are in Chinese.

**Monthly rebalanced, 100 smallest stocks by market cap, equal-weighted, 2012-12-03 ~ 2022-03-31: annualized 43.00% (report: 43.1%) — assumptions fixed a priori, run once, no tuning.**

> ⚠️ **Two execution conventions, both labeled.** Across all 24 pages, the report never states the price at which trades are executed (verified page by page). So the "this project" side comes in two versions, and every table below states which one it uses:
> - **Old · trade at T-day close**: rank on the T-day close and trade at that same close — **look-ahead bias**, removed from the engine. It reproduces the report's number to within **0.01pp**.
> - **New · trade at T+1 open**: executable convention, current default, matches to within **0.1pp**.
>
> Every "this project vs report" gap therefore contains a **convention gap**, not pure replication error. The 43.1% itself cannot reveal the convention — both versions round to 43.1% at monthly frequency. Rationale in [`grill.md`](grill.md) Q19.

## The strategy (in one sentence)

From all A-shares, take the 100 smallest by **total market cap**, equal-weight, rebalance monthly; exclude ST/\*ST, Beijing Stock Exchange stocks, and new listings with fewer than 20 trading days; on the signal day also exclude stocks at limit-up or suspended. Round-trip cost 0.3%, benchmark CSI 1000. All parameters are in [`config.yaml`](config.yaml).

## Replication results: this project vs report

**Full-period core metrics** (2012-12-03 ~ 2022-03-31; new-version numbers from the current CSVs in `output/`)

| Metric | Old · T close | New · T+1 open | Report |
|---|---:|---:|---:|
| Annualized return | 43.09% | **43.00%** | 43.1% |
| Annualized volatility | 32.4% | 32.4% | 32.1% |
| Sharpe ratio (rf=2%) | 1.269 | 1.267 | 1.28 |
| Information ratio | 2.644 | 2.637 | 2.651 |
| Calmar ratio | 0.776 | 0.775 | 0.788 |
| Max drawdown | 55.6% | 55.5% | 54.7% |
| Max drawdown window | 2015-06-12 → 07-08 | same | matches day by day |
| Benchmark annualized (CSI 1000) | 9.73% | 9.73% | 9.7% |
| Excess annualized | 33.4% | 33.3% | 33.4% |
| Total removals from portfolio | 2368 | 2368 | 2401 |

**Calendar-year returns** (%, new convention)

| | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| This project | 60.0 | 80.5 | 272.1 | 21.1 | −24.1 | −15.6 | 53.4 | 14.3 | 42.5 | **43.0** |
| Report | 60.9 | 76.7 | 267.0 | 22.2 | −22.5 | −17.1 | 52.7 | 16.4 | 45.1 | 43.1 |
| Diff | −0.9 | +3.8 | +5.1 | −1.1 | −1.6 | +1.5 | +0.7 | −2.1 | −2.6 | −0.1 |

> **Don't read this as precision.** The full-period figure lands at 43% thanks to yearly errors cancelling out: new-version max deviations are **+5.1pp (2015) / −2.6pp (2021)**, old-version **+4.2pp (2015) / −3.0pp (2020)**. What supports "the replication holds" is the path and cross-sectional evidence below, not this single endpoint.

## Evidence stronger than the headline number

Four free parameters (fees, rebalance date, execution price, filters) can fit an endpoint; they cannot fit the fine structure of the whole NAV path or the cross-section.

**Calendar effect (Fig. 28) — the strongest piece.** Daily returns over 2,267 trading days, averaged by weekday:

| Mean daily return % | Mon | Tue | Wed | Thu | Fri |
|---|---:|---:|---:|---:|---:|
| 2012.12–2016  this / report | −0.05 / −0.07 | +0.10 / +0.11 | +0.15 / +0.15 | −0.22 / −0.21 | +0.01 / +0.03 |
| 2017–2018  this / report | −0.10 / −0.11 | +0.32 / +0.32 | +0.03 / +0.04 | −0.15 / −0.14 | −0.09 / −0.10 |
| 2019–2022.3  this / report | +0.16 / +0.16 | +0.04 / +0.04 | +0.07 / +0.08 | −0.22 / −0.22 | −0.06 / −0.05 |

**15 numbers, max deviation 0.02pp.** It is also nearly immune to the execution convention (at monthly frequency it shifts holdings by only one trading day), so it validates "the right stocks were held", which is a separate matter from "trades were priced right".

**Structural and cross-sectional diagnostics (new convention)**

| Report's qualitative claim (figure) | Report | This project |
|---|---|---|
| Smaller cap → higher return · decile extremes (Fig. 6) | monotonic, steeper at small end | 6.6% → 36.8% |
| Cap buckets monotonically decreasing · extremes (Tab. 3/Fig. 30) | 43.1% → 9.7% | 43.0% → 12.5% |
| Returns driven by a few big winners · Gini winners > losers (Figs. 8/9) | winners more unequal | 0.445 > 0.349 |
| Cross-sectional cap percentiles almost all positive (Fig. 10) | almost all positive | 97.3% positive |
| Systematic buy-low/sell-high · exit − entry percentile (Figs. 12–14) | exit higher | +0.13 ~ +0.14 across three periods |
| Industry concentration in machinery · top-3 shares (Fig. 15, citics_2019 + free float) | 20 / 11 / 9% | 17.1 / 10.1 / 9.3% |
| Correlation decreasing CSI 1000 / 500 / 300 (Fig. 27) | 1000 > 500 > 300 | .94 / .90 / .68 |
| Removal attribution · cap rise / ST-tagged / delisted (Fig. 26) | 97.2 / 2.8 / 0.08% | 97.9 / 2.2 / 0.00% |

Differences under the old convention are tiny (Gini .446/.336, correlation .938, deciles 5.97%→36.56%); **no conclusion flips**. Item-by-item comparison in [`grill.md`](grill.md), "Re-run after Q19".

## A directional finding: the report's frequency conclusion flips

Section 4.2 of the report states that "daily rebalancing significantly underperforms" and treats rebalance lag itself as a source of return. Under the executable T+1 open convention, the conclusion flips — daily goes from worst to best:

| Rebalance frequency | Old · T close | New · T+1 open |
|---|---:|---:|
| Monthly | 43.09% | 43.00% |
| Weekly | 45.70% | 49.12% |
| Daily | **38.02% (worst)** | **51.89% (best)** |

The report's daily-frequency result reproduces only under the **old (T close)** convention — this is the only hard evidence for the inference that "the report used T-day close or an equivalent immediate overnight exposure". Mechanism: newly selected small caps drop especially hard the next morning (mean of 1st subsequent overnight −0.159% vs 2nd −0.123%), and daily rebalancing picks up this microstructure effect 2,267 times.

**The actual conclusion: at daily frequency, the single choice "buy at close vs. buy at next open" is worth 14pp, more than frequency itself — so the report's frequency conclusion is not robust in either direction, and 51.89% is equally untrustworthy.** At low frequency (monthly/weekly) the choice is worth only 0.09–3.4pp, and comparisons there are credible. Mechanism in [`grill.md`](grill.md) Q19.

## Making the number trustworthy

The headline 43% is the easiest number to get right by luck and wrong by luck; the real effort went into making it trustworthy:

- **No survivorship bias**: the universe is the union over the interval (not the end-of-block list); all 75 stocks delisted in 2013–2022 are in the database, each with data ending exactly on its delisting date.
- **Conventions pinned down by the benchmark first**: annualization uses **calendar days** (not 252 trading days, which would lift 43.1% to 45.0%); calendar-year returns start from the prior year-end, and risk is computed within the year only — both verified by back-solving the report's own CSI 1000 benchmark row, and locked into `tests/test_benchmark.py`.
- **Tests as guardrails**: 180 tests run in 1 second, all expected values hand-computable or taken from published report values. Purpose: "if one day it stops matching 43.1%, first rule out formula/data/engine errors".

## Known deviations and convention clarifications

| Item | Description | Nature |
|---|---|---|
| Suspended holdings "sold at frozen price" | The old convention removes holdings suspended on the day at their frozen price (unsellable in reality); a `suspended` switch now allows holding until resumption. Measured effect is **two-sided** — 2015 **−33.3pp**, 2014 **+8.0pp**, full period only **−0.29pp**; the report's headline also carries this bias (2015 +267% and the 54.7% drawdown only match the baseline) | Switch implemented, used for sensitivity; **not the "largest deviation source"** (frequent but small net magnitude); baseline remains the default for replicating the report |
| Fig. 26 delisting attribution = 0 | The three counts measure different things: status on the removal day (this repo) **0**, held-then-later-delisted **14 (0.59%)**, report **2 (0.08%)**. Those 14 near-delisting stocks are always suspended/ST-tagged first, filtered out on average **496 days** in advance, none removed on its delisting day; the report's 2 sits in between (more like "removed shortly before/after delisting", or Wind holding 2 more). No window cherry-picked to hit 2 | Different definitions, not a deviation |
| Fig. 16 CSI 300 bank weight | Two problems stacked: the report's note says "total market cap" but it is actually **free-float published weights**, and the classification must use **citics_2019** (the old `zx` source misclassifies CATL as autos). With both fixed, all 28 CSI 300 industries match the report (max 0.62pp; published weights aggregated by citics_2019 match the report **exactly** row by row). Banks 21.5% → **12.3%** (report 12.77). This also fixed Fig. 18 CSI 1000 and overturned the earlier "Fig. 15 overstated" conclusion | Located · fixed (convention + classification source) |
| Fig. 24 turnover level | Measured: the `turnover_rate` denominator is **tradable A-shares** (ratio 0.997) → old convention was low at 2.55%; switching the denominator to **free-float market cap** raises it to **3.66%**, close to the report's eyeballed 4–6%. Cost: small change in cross-group ranking (CSI 1000 slightly overtakes Small-cap 100) | Located · convention changed |

## Part 2 *Small-Cap Enhanced Strategies* replication (in-sample 2010-2022.5)

On the same engine, this replicates Section 3 of Debon's 2022-06-23 Part 2 report (enhancement family + flagship). **Benchmark, period and metrics all differ from Part 1**: baseline Small-cap 100 is **26.7%**, not 43.1%; stock selection requires at least **1 year** since listing and **non-registration-system** listings; metrics switch to annualized return / turnover / **win rate** / **strategy and benchmark average percentile**. The flagship is **timed biweekly small-cap low-vol 50 (in cash in Jan/Apr/Jun)**, report 50.9%.

**Table 26 summary (after-fee annualized, this repo vs report)**

| Strategy | This | Report | | Strategy | This | Report |
|---|--:|--:|---|---|--:|--:|
| Small-cap 100 baseline | 29.7¹ | 26.7 | | Biweekly + penalty, timed 100 | 42.7 | 44.3 |
| Small-cap 50 | 33.4 | 33.3 | | Bimonthly + penalty, timed 100 | 36.7 | 33.6 |
| Small-cap low-vol 50 | 32.7 | 32.8 | | Quarterly + penalty, timed 100 | 31.2 | 32.2 |
| Timed small-cap 100 | 39.4 | 39.2 | | **★Flagship biweekly low-vol 50** | **44.2** | **50.9** |
| Timed small-cap 50 | 42.4 | 42.1 | | Fee test (1%) | 37.8 | 44.4 |
| Timed low-vol 50 | 39.7 | 39.5 | | Keep June | 42.9 | 48.5 |
| Low-vol 50, month-end rebalance | 39.0 | 37.6 | | Capacity test 0.5bn / 1bn | 37.3 / 30.9 | 29.9 / 22.8 |
| Limit-down penalty, timed 100 | 39.1 | 36.4 | | Analyst coverage | 18.7² | 39.7 |

¹ No limit-down penalty, after fees; report brackets 26.7 (with penalty) / 29.6 (before fees). ² The consensus target-price proxy has sparser coverage than Wind; direction is right (coverage → large cap → low return) but magnitude overshoots.

**Monthly distribution, Tables 19/20, match cell by cell**: the months with negative absolute monthly return in this repo are **Jan, Apr, Jun** (same as the report), with January's excess the most negative — this is the basis for the "cash in Jan/Apr" timing.

**Global convention notes (differences come from these fixed choices, not pure replication error; item by item in [`grill_enhance.md`](grill_enhance.md))**

- **Execution at T+1 open** (inherits grill.md Q19) — the report never states its execution price.
- **Volatility window = 250 trading days** — the report never defines the window for "low volatility"; the prior choice of 60 was too short and systematically low by ~3pp; measured 250 (≈1 year) matches both Low-vol 50 (32.7/32.8) and Timed low-vol 50 (39.7/39.5).
- **Limit-down penalty** implemented faithfully (freeze the limit-down removal slot, sell at close on the first non-limit-down day); under the executable convention it costs only about **−0.4pp** (report −2.8pp) — measurement shows limit-down removal slots fall a further 5.16% on average at the next open; the report's −2.8pp is mostly its "sell at limit-down close" execution convention, which this project's T+1 open already absorbs.
- **The biweekly high-frequency family carries the Q19 frequency convention gap**: Q19 measured this execution-convention gap at 0.09pp monthly and 3.4pp weekly, growing with frequency; hence the biweekly flagship is **44.2%** here vs 50.9% in the report. Monthly/lower-frequency strategies don't carry this gap and replicate within ±1pp.
- **Strategy average percentile** is systematically ~11pp lower (63 vs 74, Wind percentile methodology); **benchmark average percentile** matches (55 vs 56). Acceptance therefore rests on annualized return + win rate + Tables 19/20.
- **Capacity test** decay shape and volatility/Sharpe signature match (simple Sharpe 2.56 ≈ report 2.55, volatility decreasing 19→17 with capital); the absolute decay rate is shallower.

### Out-of-sample extension (2022-06 ~ 2026-08, parameters unchanged verbatim)

The enhancement family's parameters are kept verbatim; only the period is extended to the end of the data (`scripts/07_enhance_oos.py`, sharing the same construction path `smallcap/enhance.py` as in-sample `06`). This goes beyond the report's sample, so only **in-sample vs out-of-sample** is shown (no report column); convention notes below.

**ladder-6 · in-sample vs out-of-sample annualized (%)**

| Strategy | In-sample | OOS | | Strategy | In-sample | OOS |
|---|--:|--:|---|---|--:|--:|
| Small-cap 100 baseline | 29.7 | 29.6 | | Timed low-vol 50 | 39.7 | 26.0 |
| Small-cap 50 | 33.4 | 33.3 | | ★Flagship biweekly low-vol 50 | 44.2 | 18.6 |
| Small-cap low-vol 50 | 32.7 | 26.9 | | Keep June | 42.9 | 18.0 |

**2024-01 micro-cap stampede · timed vs untimed (end-to-end %, three views)**

Timed strategies sit **fully in cash for the whole month** every January/April (flagship also June), and the 2024-01 micro-cap stampede fell exactly in January. Three windows side by side:

| Window | Small-cap low-vol 50 (untimed) | Timed low-vol 50 (ex Jan, Apr) | CSI 1000 | CSI 2000 |
|---|--:|--:|--:|--:|
| January 2024-01-02~01-31 | −22.8 | −0.4 | −18.3 | −21.1 |
| February re-entry 02-01~02-29 | −7.1 | −6.9 | +12.5 | +7.7 |
| Peak → recovery arc 2023-12~06-30 | −36.2 | −8.8 | −20.1 | −25.7 |

The timed strategy's NAV is flat in January (in cash) and re-enters on 2024-02-01 per the calendar-month convention, while the micro-cap trough was on Feb 5–8.

**Global convention notes (out-of-sample)**

- **Only the period changes**: the 250-day volatility window, limit-down penalty, calendar-month timing, T+1 open and other in-sample locked values are kept verbatim; no parameter is refit on out-of-sample data (grill.md Q14). An in-sample slice self-check reproduces 06 (29.7 / 44.2), proving `_ext` does not contaminate the in-sample results.
- **Secondary benchmark CSI 2000** (`932000.INDX`, back-calculated from 2013-12-31) is closer to micro-caps: out-of-sample CSI 1000 +4.6%/yr, CSI 2000 +8.6%/yr — excess measured against CSI 1000 is about 4pp higher than against CSI 2000. Flagship out-of-sample max drawdown 34% < bare baseline 48%.
- **Comparison with Part 1 out-of-sample**: the bare Small-cap 100 returns −44% out-of-sample in the 05_oos window (2024-01-02~02-08); `05_oos.py` has also gained the CSI 2000 secondary benchmark.

Cell-by-cell numbers in `output/enhance_oos_*.csv`, figures in `output/figures/enh_oos_{ladder,crash}.png`; mechanism discussion in [`grill_enhance.md`](grill_enhance.md), "样本外" (out-of-sample).

New code: `smallcap/factors.py` (volatility), `smallcap/enhance.py` (shared enhancement-family construction: ROSTER + calendar-month timing + `run_strategy`, same path for 06/07), `universe.cascade`/`not_registration`, multi-frequency + timed cash + limit-down penalty + `run_with_capacity` in `backtest`, win rate/percentile in `metrics`; drivers `scripts/06_enhance.py` (in-sample) and `scripts/07_enhance_oos.py` (out-of-sample); tests `tests/test_factors.py`, `test_enhance.py`.

## Dependencies

Python 3.11.

| Package | Version | Purpose |
|---|---|---|
| pandas / numpy | 2.3.3 / 1.26.4 | Data structures and vectorized backtesting |
| pyarrow | 25.0.0 | Parquet I/O |
| pyyaml | 6.0.3 | Reading `config.yaml` |
| pytest | 9.1.1 | Tests |
| matplotlib | 3.11.1 | Plotting; Chinese fonts must be set explicitly (tested: `PingFang SC`) |
| scipy | 1.10.1 | Kernel density estimation for Figs. 10/12–14 |
| rqdatac | 3.6.1 | **Only needed for fetching data**; backtests and tests run fully offline |

**Data is not in the repo**: `data/` is a 713 MB local Parquet cache (6,445 trading days × 5,543 stocks, 2000–2026), including back-adjusted prices from 2000 and full histories of delisted stocks. rqdatac is a trial license with a hard expiry; after it expires `01_fetch.py` can no longer pull this data, so it **must be backed up separately outside git**. Without `data/`, data-dependent tests are skipped automatically; pure formula tests still run.

## Usage

```bash
# Backtest and acceptance — fully offline, no rqdatac quota used
python scripts/02_backtest.py           # Table 1 comparison + six structural checks, ~6 s
python scripts/02_backtest.py --full    # adds frequency comparison, cap buckets, Q14 one-variable sensitivity

# Diagnostic figures — Figs. 6, 8–28, written to output/figures/
python scripts/03_analytics.py          # Figs. 8–28, ~1 min
python scripts/03_analytics.py --deciles  # adds Fig. 6 deciles (2007–2021)

# Part 2 enhancement family — Table 26 summary + Tables 19/20 + capacity Table 22 + analyst Table 24 + neutral figure set
python scripts/06_enhance.py            # 15-strategy comparison + structural checks, ~30 s
python scripts/06_enhance.py --vol-sweep  # adds E4 volatility-window sensitivity
python scripts/07_enhance_oos.py        # out-of-sample extension 2022-06~2026-08 (in-sample self-check + three crash views), ~1-2 min

# Out-of-sample (Part 1) — extended to 2026, 2024-01 micro-cap stampede breakdown
python scripts/05_oos.py                # ~10 s

# Tests — 200 tests, 1 s
python -m pytest tests/ -q

# Fetch data — only when rebuilding data/; run --smoke first to validate the API
python scripts/01_fetch.py --smoke
python scripts/01_fetch.py              # real run, resumable, cached chunks are skipped
```

## Documentation map

| File | What it records |
|---|---|
| `README.md` · [`README.zh.md`](README.zh.md) | What this is, results, how to run (English / Chinese) |
| [`PROJECT.md`](PROJECT.md) · [`project_enhance.md`](project_enhance.md) | **Where things stand** — progress, directory tree, program descriptions, to-dos, pitfalls (Part 1 / Part 2) |
| [`grill.md`](grill.md) · [`grill_enhance.md`](grill_enhance.md) | **Why** — design decisions and rationale, acceptance, sensitivity (Part 1 Q series / Part 2 E series) |

Read the relevant `grill*.md` before touching any structural decision: it records several decisions whose **original premises were later disproved**, and the reasons they were overturned.

## Version History

- 0.1 — Initial Push
- 0.2 - Aug 10 Push, added outputs like graphs and datasets
- 0.3 - Aug 11 Push, updated the trading logic to T + 1 Open
- 0.4 - Aug 13 Push, Fig. 16 industry weights switched to free-float convention (what the report's note got wrong was the convention), corroborated by index_weights; the "Fig. 15 overstated" conclusion was corrected accordingly
- 0.5 - Aug 17 Push, Part 2 out-of-sample extension (`07_enhance_oos.py`): all 15 strategies in-sample vs out-of-sample, three views of the 2024-01 crash (E3 hypothesis test); fetched CSI 2000 secondary benchmark (`index_csi2000`) and added it to 07 and 05_oos; ROSTER/timing construction moved into `smallcap/enhance.py` (shared by 06/07)
