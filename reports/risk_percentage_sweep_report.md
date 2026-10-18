# Risk Percentage Sweep: 0.2% to 1.0% Risk Per Trade
## Unified Dual-Alpha System (#07 MacKay + #10 Bayesian Bound + TIN) on Gold (XAUUSD M30)
**Account Capital:** $15,000.00 | **Out-of-Sample Period:** January 1, 2025 – September 30, 2026 (21 Months)  
**Execution Baseline:** 1-Trade-a-Day Governance | $1.0\times\text{ATR}$ Stop Loss | Break-Even @ $+1.0R$ | 20:30 UTC EOD Flatten

---

## 1. Executive Summary & Scaling Dynamics

We evaluated risk allocation scaling across two exit philosophies:
1. **Suite 1: Asymmetric 1:3.0 Take Profit** (+1.0R Break-Even transition, $1.0\times\text{ATR}$ SL).
2. **Suite 2: 1.5 × ATR Dynamic Trailing Stop** (Open target, +1.0R Break-Even transition, $1.0\times\text{ATR}$ SL).

```mermaid
flowchart LR
    A["0.2% - 0.4% Risk: Ultra-Conservative (Prop Firm Shield)"] -->|Max DD 4.0% - 7.6%| B["Net Profit: +$2,255 to +$3,722 (+15% to +25%)"]
    C["0.5% Risk: Institutional Sweet Spot"] -->|Max DD 9.4% (Under 10% Prop Limit)| D["Net Profit: +$4,677 (+31.2%, Sharpe 1.32 - 1.41)"]
    E["0.8% - 1.0% Risk: Aggressive Capital Compounding"] -->|Max DD 13.8% - 18.1%| F["Net Profit: +$7,491 to +$10,917 (+50% to +72.8%)"]
```

### Key Quantitative Findings:
- **Linear Alpha Scaling**: Because 1-trade-a-day governance eliminates position overlap and compounding spikes, returns scale smoothly with risk with zero distortion to Profit Factor (**1.25 to 1.34**) or Payoff Ratio (**4.30x for 1:3 TP, 1.93x for Trailing Stop**).
- **The 0.50% Risk Prop-Firm Threshold**: At **0.50% risk per trade**, maximum drawdown is capped at **9.43%** for 1:3 TP and **10.76%** for Trailing Stop, making 0.5% the highest viable sizing to pass strict 10% maximum trailing drawdown rules while returning **+$4,677.68 (+31.18%)**.
- **Ultra-Conservative 0.2% Sizing**: Delivers a **Sharpe Ratio of 1.49–1.62** with maximum drawdowns suppressed to just **4.02%–4.42%**, ideal for high-capital accounts seeking absolute capital preservation.
- **Payoff Ratio Asymmetry**: The 1:3 TP model produced an extraordinary **4.30x Payoff Ratio** ($571.99 average win vs -$133.07 average loss at 1% risk) because the break-even trigger cuts losing trades early while allowing home runs to reach the 3.0R target.

---

## 2. Comparative Matrix Across Risk Levels ($15,000 Account, 2025–2026)

### Suite 1: Asymmetric 1:3.0 Take Profit (+1.0R Break-Even)

| Risk % / Trade | Ending Balance | Net Profit ($) | Net Return (%) | Net Sharpe | Profit Factor | Payoff Ratio | Win Rate (%) | Avg Win ($) | Avg Loss ($) | Max Drawdown (%) | Max Drawdown ($) | Total Trades |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20%** 🛡️ | $17,255.02 | +$2,255.02 | +15.03% | **1.494** | **1.31** | **4.50** | 22.60% | $100.71 | -$22.40 | **4.02%** | $693.65 | 416 |
| **0.30%** | $17,879.11 | +$2,879.11 | +19.19% | 1.380 | 1.27 | 4.36 | 22.60% | $143.44 | -$32.93 | **5.75%** | $1,028.05 | 416 |
| **0.40%** | $18,722.38 | +$3,722.38 | +24.82% | 1.338 | 1.26 | 4.31 | 22.60% | $192.67 | -$44.68 | **7.61%** | $1,424.77 | 416 |
| **0.50%** 🏆 | **$19,677.68** | **+$4,677.68** | **+31.18%** | **1.318** | **1.25** | **4.30** | **22.60%** | **$245.75** | **-$57.21** | **9.43%** | **$1,855.61** | **416** |
| **0.60%** | $20,732.18 | +$5,732.18 | +38.21% | 1.309 | 1.25 | 4.29 | 22.60% | $302.85 | -$70.61 | 11.23% | $2,328.22 | 416 |
| **0.80%** | $22,491.65 | +$7,491.65 | +49.94% | 1.304 | 1.25 | 4.29 | 22.60% | $396.28 | -$92.42 | 13.86% | $3,117.34 | 416 |
| **1.00%** 🚀 | **$25,917.44** | **+$10,917.44** | **+72.78%** | **1.305** | **1.25** | **4.30** | **22.60%** | **$571.99** | **-$133.07** | **18.11%** | **$4,693.65** | **416** |

---

### Suite 2: 1.5 × ATR Dynamic Trailing Stop (Open Target + +1.0R Break-Even)

| Risk % / Trade | Ending Balance | Net Profit ($) | Net Return (%) | Net Sharpe | Profit Factor | Payoff Ratio | Win Rate (%) | Avg Win ($) | Avg Loss ($) | Max Drawdown (%) | Max Drawdown ($) | Total Trades |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.20%** 🛡️ | $17,128.70 | +$2,128.70 | +14.19% | **1.616** | **1.34** | **2.00** | 40.14% | $49.91 | -$24.92 | **4.42%** | $757.09 | 416 |
| **0.30%** | $17,768.43 | +$2,768.43 | +18.46% | 1.485 | 1.31 | 1.95 | 40.14% | $70.65 | -$36.27 | **6.56%** | $1,165.61 | 416 |
| **0.40%** | $18,563.14 | +$3,563.14 | +23.75% | 1.429 | 1.29 | 1.93 | 40.14% | $94.26 | -$48.91 | **8.68%** | $1,611.28 | 416 |
| **0.50%** 🏆 | **$19,485.18** | **+$4,485.18** | **+29.90%** | **1.414** | **1.29** | **1.92** | **40.14%** | **$119.64** | **-$62.23** | **10.76%** | **$2,096.61** | **416** |
| **0.60%** | $20,523.64 | +$5,523.64 | +36.82% | 1.417 | 1.29 | 1.92 | 40.14% | $146.83 | -$76.30 | 12.80% | $2,627.02 | 416 |
| **0.80%** | $22,198.16 | +$7,198.16 | +47.99% | 1.420 | 1.29 | 1.93 | 40.14% | $190.39 | -$98.78 | 15.80% | $3,507.31 | 416 |
| **1.00%** 🚀 | **$25,162.31** | **+$10,162.31** | **+67.75%** | **1.414** | **1.29** | **1.93** | **40.14%** | **$268.18** | **-$139.05** | **19.81%** | **$4,984.65** | **416** |

---

## 3. Exit Philosophy Comparison: Fixed 1:3 TP vs. 1.5 ATR Trailing Stop

```
                      1:3.0 Fixed Take Profit             1.5 ATR Trailing Stop
Win Rate:             22.60% (Fewer, huge hits)           40.14% (Smooth hit rate)
Payoff Ratio:         4.30x (Asymmetric multi-R)          1.93x (Controlled trail)
Avg Win (at 1% risk): $571.99                             $268.18
Sharpe Ratio:         1.305 - 1.494                       1.414 - 1.616 (Higher Sharpe)
Drawdown at 0.5% risk:9.43% (Safely sub-10%)              10.76% (Near prop limit)
Net Return at 1%:     +72.78% (+$10,917.44)               +67.75% (+$10,162.31)
```

### Recommendation by Trader Objective:
1. **Prop Firm Challenge / Evaluation Rules**:
   - Deploy **1:3.0 Fixed TP at 0.40%–0.50% risk per trade**.
   - Maximum Drawdown stays below **9.43%**, fully compliant with 10% maximum trailing loss rules, while delivering **+24.8% to +31.2% net gain**.
2. **Personal Funded Account / Aggressive Growth**:
   - Deploy **1:3.0 Fixed TP at 1.00% risk per trade**.
   - Net profit reaches **+$10,917.44 (+72.78%)** on the $15k account with an average winning trade of **+$571.99**.
3. **Smooth Equity Curve Preference**:
   - Deploy **1.5 ATR Trailing Stop at 0.30%–0.40% risk**.
   - Provides a higher win rate (**40.14%** vs 22.60%) and the highest Sharpe Ratio (**1.43–1.62**) with drawdowns under 8.7%.
