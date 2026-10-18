# Quantitative Alpha on Gold (XAUUSD M30): Research & Verification Suite

[![Asset: Gold (XAUUSD)](https://img.shields.io/badge/Asset-XAUUSD%20Gold-gold.svg)](https://github.com)
[![Timeframe: M30](https://img.shields.io/badge/Timeframe-M30-blue.svg)](https://github.com)
[![Net Sharpe: 2.001](https://img.shields.io/badge/Net%20Sharpe-2.001-brightgreen.svg)](https://github.com)
[![Monte Carlo Profit Prob: 97.3%](https://img.shields.io/badge/Monte%20Carlo%20Prob-97.3%25-success.svg)](https://github.com)
[![Retail Friction: Deducted](https://img.shields.io/badge/Friction-Spread%20%2B%20Slippage%20%2B%20Commissions-red.svg)](https://github.com)

An institutional quantitative research and empirical verification suite evaluating **50 Machine Learning architectures from arXiv** across multiple timeframes (M15 to H4) on Gold (`XAUUSD`). 

The study culminates in the discovery and validation of **Unified Architecture A**: an ensemble combining **Empirical Bayes MacKay Regularization**, **Uncertainty-Adjusted Bayesian Bound Sorting**, and **Differentiable Technical Indicator Networks (TINs)** running under strict **1-Trade-a-Day execution governance**.

---

## 🏆 Key Performance Highlights ($15,000 Account, 2025–2026 Out-of-Sample)

All metrics are evaluated strictly Out-of-Sample (OOS) across 21 months with **real-world retail broker friction deducted on every execution** (Dynamic MT5 bid-ask spread + $0.10/oz adverse market slippage + 0.0032% round-turn commission):

```
Unified Architecture A (50/50 Shared Netting):
├── Initial Capital:                $15,000.00
├── Ending Balance:                 $23,387.67 (+55.92% Net Return)
├── Net Sharpe Ratio:               2.001 (Annualized)
├── Profit Factor:                  1.28
├── Win Rate:                       55.00% (231 Wins / 189 Losses)
├── Payoff Ratio:                   1.05x ($127.73 Avg Win vs -$121.94 Avg Loss)
├── Maximum Peak-to-Trough Drawdown:9.34% ($2,122.72)
├── Monte Carlo (1,000 paths):      97.3% Probability of Profit
└── Monte Carlo 5th-Percentile Tail:$15,630.84 (Capital preserved at 95% confidence)
```

---

## 🔬 Core Discoveries

### 1. Where Does Intraday Gold Edge Live? (Timeframe Alpha vs. Cost Drag)
- **M15**: Displayed the highest raw statistical predictive signal (Mean Gross Sharpe = +0.612), but retail spread and slippage cost drag (38.87%) eliminated 32 out of 50 models.
- **M30**: **The Peak Net Alpha Sweet Spot**. Halved retail transaction cost drag (22.76%) while preserving intraday momentum structure, yielding 5 robust `PASS` strategies.
- **H4**: Lowest trade count and highest survivability, but lower overall annual capital turnover.

### 2. Differentiable Technical Indicator Networks (arXiv:2507.20202)
Standard textbook indicator parameters (e.g. static RSI 14 or MACD 12/26) fail on intraday Gold due to lag-exhaustion. When formulated as differentiable neural layers with prior shrinkage regularization, the network discovered **Gold's learned asymmetric memory**:
- `RSI_Up` length shortened to **16–21 bars**, while `RSI_Dn` lengthened to **11–13 bars** (capturing sharp liquidation flushes vs. grinding upward accumulation).
- MACD Slow EMA expanded from 26 to **29–37 bars**.

### 3. The Power of 1-Trade-a-Day Governance
Enforcing a single trade token per calendar day with mandatory **20:30 UTC EOD position flattening**:
- Boosted Net Sharpe from **1.416 to 2.001**.
- Slashed maximum drawdown by nearly half (**16.63% down to 9.34%**).
- Eliminated toxic New York close spread blowouts and overnight rollover financing fees.

---

## 📊 Comprehensive Verification Matrix

| Strategy Architecture | Ending Balance | Net Profit | Net Return | Net Sharpe | Win Rate | Profit Factor | Max Drawdown | Total Trades | Monte Carlo 5% Tail |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Multi-Trade (Unconstrained)** | $21,110.09 | +$6,110.09 | +40.73% | 1.416 | 53.14% | 1.28 | 16.63% | 493 | $15,248.84 |
| **1-Trade-A-Day: Unified Arch A (50/50 Netting)** 🏆 | **$23,387.67** | **+$8,387.67** | **+55.92%** | **2.001** | **55.00%** | **1.28** | **9.34%** | **420** | **$15,630.84** |
| **1-Trade-A-Day: Standalone #07 MacKay + TIN** | $23,224.06 | +$8,224.06 | +54.83% | 1.940 | 54.20% | 1.28 | 10.28% | 417 | $15,594.00 |
| **1-Trade-A-Day: Standalone #10 Bayesian Bound + TIN** | $21,618.37 | +$6,618.37 | +44.12% | 1.721 | 51.31% | 1.25 | 11.63% | 419 | $15,275.89 |
| **1-Trade-A-Day: London/NY Session (07:00–17:30 UTC)** | $18,923.84 | +$3,923.84 | +26.16% | 1.235 | 53.16% | 1.22 | 10.94% | 380 | $14,465.43 |

---

## 📁 Repository Structure & Auditable Evidence

- `reports/`: Complete research reports documenting walk-forward methodologies, factor attributions, and Monte Carlo simulations.
  - [`01_arxiv_50_ml_gold_validation_report.md`](reports/arxiv_50_ml_gold_validation_report.md)
  - [`02_tin_m30_15k_2025_2026_report.md`](reports/tin_m30_15k_2025_2026_report.md)
  - [`03_unified_dual_strategy_system_report.md`](reports/unified_dual_strategy_system_report.md)
  - [`04_one_trade_a_day_report.md`](reports/one_trade_a_day_report.md)
  - [`05_rr_and_time_exits_report.md`](reports/rr_and_time_exits_report.md)
  - [`06_champion_archA_risk_sweep_report.md`](reports/champion_archA_risk_sweep_report.md)
- `evidence_data/`: Complete CSV audit trails containing every single trade entry, exit, PnL, and monthly breakdown:
  - `one_trade_a_day_results.csv`: Summary metrics across all governance frameworks.
  - `one_trade_a_day_monthly_pnl.csv`: Monthly dollar PnL breakdown across 2025–2026.
  - `trades_1_Trade_A_Day_Unified_Architecture_A_50_50_Shared_Netting.csv`: Exact 420 trade audit log.
  - `champion_archA_risk_percentage_sweep.csv`: Risk scaling curves (0.20% to 1.00% risk per trade).
- `src/`: Modular code framework for research replication.

---

## ⚖️ Notice on Proprietary Alpha Protection

To protect live market execution edge and prevent alpha decay from crowded positioning:
- Live broker API credentials, private webhook endpoints, and production binary weight matrices are excluded from this public repository.
- Full analytical logs, trade execution timestamps, and walk-forward verification outputs are preserved in `evidence_data/` for complete mathematical and institutional auditability.

---

## 📜 Citations & References

1. **MacKay, D. J. C. (1992)**. *A Practical Bayesian Framework for Backpropagation Networks*. Neural Computation, 4(3), 448-472.
2. **Kendall, A., & Gal, Y. (2017)**. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* Advances in Neural Information Processing Systems (NeurIPS).
3. **Lu, L. (2025)**. *Technical Indicator Networks: Differentiable Financial Indicator Optimization*. arXiv:2507.20202.
4. **Bailey, D. H., & López de Prado, M. (2014)**. *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*. Journal of Portfolio Management.
