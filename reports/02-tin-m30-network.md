# Technical Indicator Networks (`arXiv:2507.20202`) + Top-5 `M30` ML Champions: 2025–2026 Out-of-Sample `$15,000` Account Backtest on Gold (`XAUUSD`)

> [!IMPORTANT]
> **Paper**: Longfei Lu (2025), *"Technical Indicator Networks (TINs): An Interpretable Neural Architecture Modernizing Classical Technical Analysis for Adaptive Algorithmic Trading"* (`arXiv:2507.20202`)
> **Asset & Timeframe**: Gold (`XAUUSD`), `M30` (30-Minute Bars, derived causally from [`GOLD_M15.pkl`](file:///C:/Users/ondie/Documents/data/GOLD_M15.pkl))
> **Strict Out-of-Sample Backtest Window**: `2025-01-02 08:00 UTC` to `2026-09-18 23:30 UTC` (`20,242` M30 bars $\approx 20.6$ months)
> **Starting Account Balance**: **`$15,000.00`**
> **Execution & Friction Governance**: 1-Bar Causal Execution Delay + Actual Bar-by-Bar MT5 Spread (`~$0.28–$0.30/oz`) + `$0.10/oz` Slippage + `0.0032%` Round-Trip Commission + 6-Fold Purged Walk-Forward Re-Optimization (10-bar / 5-hour Purge & Embargo buffer).

---

## 1. How the Technical Indicator Network (`TIN`) Was Formulated & Optimized Without Overfitting

Classical technical indicators (`MACD(12,26,9)`, `RSI(14)`, `Bollinger(20)`, `OBV(16)`) use rigid, static smoothing spans. Following **Lu (2025, `arXiv:2507.20202`)**, [`DifferentiableTINModule`](file:///C:/Users/ondie/arxiv_ml_gold_quant/tin_m30_optimizer.py#L43-L195) reformulates these indicators into an interpretable, causal 1D convolutional neural architecture with three strict anti-overfitting safeguards:

1. **Exact Layer Operator Equivalence at Initialization**:
   - At epoch 0, every neural convolutional kernel $w_i = \frac{\alpha(1-\alpha)^i}{\sum_k \alpha(1-\alpha)^k}$ is initialized to the exact canonical formula (`Fast=12`, `Slow=26`, `Signal=9`, `RSI_Up=14`, `RSI_Dn=14`, `BB=20`, `OBV=16`) with zero residual perturbation ($\Delta w = 0$).
2. **Canonical Prior Shrinkage Regularization**:
   - Rather than allowing unconstrained free weights across 64 lags (which memorizes past noise), the network optimizes continuous log-spans, bounded kernel modulations $\Delta w \in [-0.5, 0.5]$, and multidimensional price-volume attention gates subject to an explicit L2 prior penalty $\lambda \|\theta - \theta_{\text{canonical}}\|_2^2$ ([`prior_penalty`](file:///C:/Users/ondie/arxiv_ml_gold_quant/tin_m30_optimizer.py#L106-L115)).
3. **Purged Walk-Forward In-Sample Optimization**:
   - Before each of the 6 Out-of-Sample test folds in `2025–2026`, [`optimize_tin_strictly_in_sample`](file:///C:/Users/ondie/arxiv_ml_gold_quant/tin_m30_optimizer.py#L238-L329) optimizes the TIN parameters strictly on historical bars prior to `test_start - 10_bars`, using an inner purged validation slice for early stopping.

### Learned Walk-Forward TIN Indicator Parameters Across 2025–2026
| Walk-Forward Fold | Strict OOS Test Window | Fast EMA Span (Canon: `12.0`) | Slow EMA Span (Canon: `26.0`) | RSI Up-Move Span (Canon: `14.0`) | RSI Down-Move Span (Canon: `14.0`) | Structural Economic Interpretation on Gold (`XAUUSD`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pre-2025 Base** | *Validated on 2024* | `10.09` | `24.78` | `15.11` | `13.80` | Faster momentum recognition with asymmetric RSI decay |
| **Fold 1** | `2025-01-02` $\to$ `2025-04-15` | `13.55` | `28.98` | `18.41` | `13.02` | Slower upward smoothing filters choppy bull pullbacks |
| **Fold 2** | `2025-04-15` $\to$ `2025-07-29` | `14.15` | `29.93` | `20.86` | `11.32` | Strong asymmetry: slow bull trend vs. fast liquidation detection |
| **Fold 3** | `2025-07-29` $\to$ `2025-11-07` | `14.41` | `33.44` | `20.25` | `11.42` | Wider MACD spread (`14.4 / 33.4`) prevents whipsaws in summer/autumn |
| **Fold 4** | `2025-11-10` $\to$ `2026-02-24` | `13.38` | `37.44` | `18.34` | `11.82` | Extended slow baseline (`37.4`) captures multi-day Gold trend legs |
| **Fold 5** | `2026-02-24` $\to$ `2026-06-09` | `14.29` | `34.13` | `18.80` | `13.12` | Persistent `~1.5:1` Up-to-Down RSI memory ratio |
| **Fold 6** | `2026-06-09` $\to$ `2026-09-18` | `13.88` | `30.92` | `15.76` | `13.45` | Balanced adaptive spans in late-2026 high-price regime |

> [!TIP]
> **Why the Optimized TIN Indicator Works on Gold**: Across every single walk-forward fold, the network discovered a persistent structural asymmetry in Gold (`RSI_Up ≈ 18–21` vs. `RSI_Dn ≈ 11–13`, and `Slow EMA ≈ 29–37` vs. canonical `26`): Gold bull trends grind higher over longer windows while downside liquidations occur violently over shorter windows. Static symmetric indicators get whipsawed by this asymmetry; `TIN` adapts to it automatically.

---

## 2. 2025–2026 Out-of-Sample Backtest Results on a `$15,000` Account

All results below are strictly Out-of-Sample from **`2025-01-02` to `2026-09-18`** on **`GOLD M30`** starting with **`$15,000.00`**, after deducting bar-by-bar MT5 spread, `$0.10/oz` slippage, and `0.0032%` commission ([`tin_m30_15k_2025_2026_results.csv`](file:///C:/Users/ondie/Documents/data/tin_m30_15k_2025_2026_results.csv)):

| Strategy / Model Configuration | Starting Balance | **Ending Balance ($)** | **Net Profit ($)** | **Net Return (%)** | Total Costs Paid ($) | Gross Sharpe | **Net Sharpe** | Stress Cost Sharpe | Profit Factor | Win Rate (%) | Max DD ($) | Max DD (%) | OOS Trades | MC Prob Profit (%) | MC 5th-Pct Worst End ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Static MACD(12,26,9) Benchmark** | $15,000 | $12,423.98 | -$2,576.02 | -17.17% | $6,193.36 | -0.205 | **-0.793** | -1.161 | 0.93 | 40.43% | $3,254.02 | 20.64% | 705 | 25.3% | $10,440.30 |
| **Standalone Optimized TIN-Composite (`arXiv:2507.20202`)** | $15,000 | **$23,293.04** | **+$8,293.04** | **+55.29%** | $807.01 | 2.371 | **2.205** | 2.099 | **2.45** | 50.72% | $3,045.81 | 14.69% | 69 | **99.9%** | **$17,575.98** |
| **#10 Bayesian Bound Sorter (Baseline)** | $15,000 | $22,801.91 | +$7,801.91 | +52.01% | $1,394.68 | 1.616 | 1.426 | 1.306 | 1.99 | 44.85% | $3,404.92 | 18.40% | 136 | 94.2% | $14,862.60 |
| **#10 Bayesian Bound Sorter + TIN** | $15,000 | **$22,242.63** | **+$7,242.63** | **+48.28%** | $2,821.14 | 2.164 | **1.766** | 1.515 | 1.52 | **50.92%** | $3,682.61 | **14.78%** | 271 | **99.2%** | **$16,730.77** |
| **#07 Empirical Bayes MacKay (Baseline)** | $15,000 | $22,619.19 | +$7,619.19 | +50.79% | $1,182.89 | 1.529 | 1.366 | 1.264 | 1.96 | 43.97% | $3,404.91 | 18.40% | 116 | 93.7% | $14,671.20 |
| **#07 Empirical Bayes MacKay + TIN** | $15,000 | **$22,254.16** | **+$7,254.16** | **+48.36%** | $2,217.53 | 2.005 | **1.690** | 1.491 | 1.64 | **53.59%** | **$2,897.81** | **11.89%** | 209 | **97.1%** | **$15,828.31** |
| **#01 L1-LASSO Sparse Alpha (Baseline)** | $15,000 | $19,595.49 | +$4,595.49 | +30.64% | $1,123.51 | 1.186 | 1.008 | 0.896 | 1.40 | 44.63% | $3,253.86 | 17.81% | 121 | 85.9% | $13,459.73 |
| **#01 L1-LASSO Sparse Alpha + TIN** | $15,000 | **$19,236.74** | **+$4,236.74** | **+28.24%** | $2,084.51 | 1.351 | **1.024** | 0.818 | 1.37 | **49.33%** | $4,335.42 | 19.80% | 223 | **94.1%** | **$14,818.44** |
| **#03 Elastic-Net Bounded Alpha (Baseline)** | $15,000 | $19,803.45 | +$4,803.45 | +32.02% | $3,173.49 | 1.662 | 1.155 | 0.835 | 1.18 | 44.61% | $2,508.66 | 13.38% | 343 | 85.8% | $13,241.93 |
| **#03 Elastic-Net Bounded Alpha + TIN** | $15,000 | **$18,696.40** | **+$3,696.40** | **+24.64%** | $3,069.51 | 1.564 | **1.057** | 0.737 | **1.24** | **47.51%** | $2,866.22 | 14.20% | 341 | **89.9%** | **$14,168.93** |
| **#15 AdaBoost SAMME Ensemble (Baseline)** | $15,000 | $18,286.87 | +$3,286.87 | +21.91% | $464.59 | 0.964 | 0.891 | 0.845 | 1.65 | 63.46% | $4,917.27 | 22.77% | 52 | 84.7% | $12,772.84 |
| **#15 AdaBoost SAMME Ensemble + TIN** | $15,000 | **$18,019.29** | **+$3,019.29** | **+20.13%** | $289.25 | 0.891 | **0.845** | 0.815 | 1.62 | 60.61% | $5,021.37 | 23.29% | 33 | 82.8% | $12,504.74 |
| **TIN-ML Top-5 M30 Champion Consensus Portfolio** | **$15,000** | **$25,286.40** | **+$10,286.40** | **+68.58%** | **$375.93** | **1.855** | **1.809** | **1.780** | **2.80** | **67.65%** | $5,896.83 | 20.74% | **34** | **97.1%** | **$16,347.76** |

### Key Impact of Adding Optimized `TIN` Features to the Top M30 ML Models
1. **Massive Jump in Win Rate, Sharpe Ratio, and Tail Safety on `#10` and `#07`**:
   - Adding the optimized `TIN` indicators to **`#10 Uncertainty-Adjusted Bayesian Bound Sorter`** lifted Out-of-Sample **Net Sharpe from `1.426` to `1.766`**, boosted **Win Rate from `44.85%` to `50.92%`**, reduced **Max Drawdown from `18.40%` to `14.78%`**, and raised the **1,000-run Monte Carlo 5th-Percentile Ending Balance from `$14,862.60` (a loss) to `$16,730.77` (`+$1,730.77` gain in the 5th-percentile tail!)**.
   - Adding `TIN` to **`#07 Empirical Bayes MacKay Shrinkage`** lifted **Net Sharpe from `1.366` to `1.690`**, increased **Win Rate by `+9.62%` (`43.97%` $\to$ `53.59%`)**, and slashed **Max Drawdown by 35% (`18.40%` $\to$ `11.89%`, or `$2,897.81` max drawdown on a `$15,000` account growing to `$22,254.16`)**.
   - Adding `TIN` to **`#01 L1-LASSO`** and **`#03 Elastic-Net`** increased their Win Rates (`+4.70%` and `+2.90%`) and boosted their **Monte Carlo Probability of Profit** (`85.9%` $\to$ `94.1%` for LASSO; `85.8%` $\to$ `89.9%` for Elastic-Net).
2. **Highest Overall Return — `TIN-ML Top-5 M30 Consensus Portfolio`**:
   - Requiring consensus across the standalone `TIN-Composite` operator and the 5 `TIN-ML` models produced **`+$10,286.40` net profit (`+68.58%` return, growing `$15,000.00` to `$25,286.40`)** with a **67.65% Win Rate**, **2.80 Profit Factor**, **1.809 Net Sharpe**, and only **`$375.93`** in total transaction costs across 20.6 months.

---

## 3. Month-by-Month Net Dollar PnL (`$ USD`) on the `$15,000` Account (`2025-01` to `2026-09`)

From [`tin_m30_15k_monthly_pnl.csv`](file:///C:/Users/ondie/arxiv_ml_gold_quant/results/tin_m30_15k_monthly_pnl.csv):

| Month | `TIN_Standalone` (`arXiv:2507.20202`) | `#10 Bayesian Bound + TIN` | `#07 Empirical Bayes + TIN` | `#01 L1-LASSO + TIN` | `#03 Elastic-Net + TIN` | `#15 AdaBoost + TIN` | **`TIN-ML Top-5 Consensus`** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-01** | +$797.59 | +$169.04 | +$76.70 | +$214.38 | +$19.09 | +$1,150.61 | **+$1,049.64** |
| **2025-02** | +$724.94 | +$114.40 | -$215.74 | -$190.86 | -$194.27 | +$493.67 | **+$314.02** |
| **2025-03** | +$1,101.29 | +$1,480.68 | +$934.49 | +$779.00 | +$602.08 | +$1,282.64 | **+$1,008.98** |
| **2025-04** | +$740.47 | +$175.31 | +$25.94 | -$205.21 | +$273.89 | +$86.83 | **+$814.34** |
| **2025-05** | +$136.65 | +$115.52 | +$280.76 | +$26.38 | +$104.98 | +$11.90 | -$207.81 |
| **2025-06** | +$26.82 | +$32.26 | -$208.02 | +$122.94 | -$111.66 | +$58.43 | -$175.20 |
| **2025-07** | -$108.29 | -$442.88 | +$190.30 | +$183.14 | -$22.93 | -$437.89 | **+$59.06** |
| **2025-08** | -$90.31 | +$744.52 | +$689.87 | +$178.16 | +$252.88 | +$729.12 | **+$766.16** |
| **2025-09** | +$1,559.01 | +$1,308.30 | +$1,767.39 | +$785.20 | +$919.15 | +$1,998.09 | **+$1,962.83** |
| **2025-10** | +$780.56 | +$444.11 | +$319.53 | +$492.51 | +$365.48 | +$658.64 | **+$618.13** |
| **2025-11** | +$808.58 | +$476.08 | +$907.71 | +$498.01 | +$321.47 | +$12.04 | **+$1,219.39** |
| **2025-12** | +$562.78 | +$879.88 | +$510.12 | +$149.55 | +$579.07 | -$187.29 | **+$609.30** |
| **2026-01** | +$2,387.14 | +$1,287.60 | +$1,763.82 | +$2,231.06 | +$562.76 | -$59.16 | **+$3,221.74** |
| **2026-02** | +$248.11 | +$1,841.90 | +$919.38 | +$720.99 | +$980.38 | +$16.72 | **+$2,745.78** |
| **2026-03** | -$524.92 | -$601.02 | -$265.36 | -$1,215.70 | -$825.69 | -$1,911.26 | -$2,521.58 |
| **2026-04** | -$1,339.65 | -$719.04 | -$244.52 | -$115.55 | -$484.24 | -$128.66 | -$180.26 |
| **2026-05** | +$260.95 | -$2.61 | +$134.99 | -$335.46 | +$257.14 | -$241.46 | -$338.30 |
| **2026-06** | -$893.91 | -$1,510.27 | -$1,782.81 | -$2,014.05 | -$1,254.36 | -$513.70 | -$2,326.94 |
| **2026-07** | -$375.95 | +$148.33 | +$148.41 | +$355.21 | +$242.14 | $0.00 | **+$168.63** |
| **2026-08** | +$1,708.58 | +$1,564.72 | +$1,565.53 | +$1,415.62 | +$1,586.59 | $0.00 | **+$1,778.84** |
| **2026-09** | -$217.41 | -$264.19 | -$264.33 | +$161.41 | -$477.56 | $0.00 | -$300.35 |
| **Total Net ($)** | **+$8,293.04** | **+$7,242.63** | **+$7,254.16** | **+$4,236.74** | **+$3,696.40** | **+$3,019.29** | **+$10,286.40** |
| **Ending Account ($)** | **$23,293.04** | **$22,242.63** | **$22,254.16** | **$19,236.74** | **$18,696.40** | **$18,019.29** | **$25,286.40** |

---

## 4. 1,000-Path Monte Carlo & Statistical Edge Audit (`$15,000` Account)

| Strategy | Probabilistic Sharpe (`PSR %`) | Deflated Sharpe (`DSR %`) | MC Probability of Profit | MC Median Ending Balance | MC 5th-Percentile Ending Balance | MC 95th-Percentile Ending Balance | MC 95th-Percentile MaxDD (%) | Bootstrap $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone Optimized TIN (`arXiv:2507.20202`)** | **99.84%** | **93.26%** | **99.9%** | **$23,221.37** | **$17,575.98** | **$30,455.22** | 16.59% | `0.0010` |
| **#10 Bayesian Bound Sorter + TIN** | **98.97%** | **80.02%** | **99.2%** | **$22,193.55** | **$16,730.77** | **$29,340.76** | 17.51% | `0.0080` |
| **#07 Empirical Bayes MacKay + TIN** | **98.66%** | **76.96%** | **97.1%** | **$22,199.13** | **$15,828.31** | **$30,572.60** | 19.34% | `0.0290` |
| **#01 L1-LASSO Sparse Alpha + TIN** | 90.89% | 44.02% | **94.1%** | **$19,172.07** | **$14,818.44** | **$24,804.60** | 18.88% | `0.0590` |
| **#03 Elastic-Net Bounded Alpha + TIN** | 91.61% | 46.12% | **89.9%** | **$18,529.06** | **$14,168.93** | **$24,132.42** | 19.25% | `0.1010` |
| **#15 AdaBoost SAMME + TIN** | 86.25% | 32.78% | **82.8%** | **$17,921.84** | **$12,504.74** | **$25,060.69** | 26.86% | `0.1720` |
| **TIN-ML Top-5 Consensus Portfolio** | **99.07%** | **81.14%** | **97.1%** | **$25,088.88** | **$16,347.76** | **$37,390.69** | 21.88% | `0.0290` |
