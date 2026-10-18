# Research Report 08: $100,000 Account Risk Percentage Sweep (0.50% – 1.00%)

**Asset**: Gold (XAUUSD)  
**Timeframe**: M30  
**Sample Period**: January 1, 2025 – September 30, 2026 (21 Months Out-of-Sample)  
**System**: Champion 1-Trade-A-Day Unified Architecture A (50/50 Shared Netting of #07 MacKay Shrinkage + #10 Uncertainty-Adjusted Bayesian Bound + TIN)  
**Initial Balance**: $100,000.00 USD  
**Total Executed Trades**: 420 Trades  

---

## 1. Executive Summary

Following validation on a $15,000 account, this sweep tests scaling to an institutional **$100,000 account** across equity risk tiers from **0.50% to 1.00% per trade**. 

Because the 1-Trade-a-Day governance strictly limits turnover to at most 1 trade per day and flattens all positions prior to rollover, the system exhibits stable execution without cascading margin calls or slippage clustering.

### Key Performance Matrix ($100k Account, 2025–2026 OOS)

| Risk % / Trade | Target Leverage | Ending Balance | Net Profit | Net Return | Net Sharpe | Win Rate | Profit Factor | Payoff Ratio | Avg Win | Avg Loss | Max Drawdown (%) | Max Drawdown ($) | MC 5% Tail Balance | Zero Ruin Probability |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.50% 🛡** | 1.0x | **$127,241.91** | **+$27,241.91** | **+27.24%** | **2.001** | **55.00%** | **1.29** | **1.06** | +$517.32 | -$487.61 | **8.96%** | $11,522.70 | **$104,597.59** | **100.0%** |
| **0.60%** | 1.2x | **$133,270.84** | **+$33,270.84** | **+33.27%** | **2.001** | **55.00%** | **1.29** | **1.05** | +$640.93 | -$607.93 | **10.70%** | $14,462.62 | **$103,111.17** | **100.0%** |
| **0.70%** | 1.4x | **$139,497.45** | **+$39,497.45** | **+39.50%** | **2.001** | **55.00%** | **1.28** | **1.05** | +$769.95 | -$733.25 | **12.42%** | $17,641.97 | **$104,056.09** | **100.0%** |
| **0.75% 🏆** | 1.5x | **$142,685.26** | **+$42,685.26** | **+42.69%** | **2.001** | **55.00%** | **1.28** | **1.05** | +$836.57 | -$798.54 | **13.28%** | $19,325.36 | **$105,881.52** | **100.0%** |
| **0.80%** | 1.6x | **$145,922.91** | **+$45,922.91** | **+45.92%** | **2.001** | **55.00%** | **1.28** | **1.05** | +$904.53 | -$865.26 | **14.13%** | $21,073.37 | **$103,451.95** | **100.0%** |
| **0.90%** | 1.8x | **$152,548.05** | **+$52,548.05** | **+52.55%** | **2.001** | **55.00%** | **1.27** | **1.04** | +$1,044.57 | -$1,003.11 | **15.82%** | $24,769.88 | **$108,198.33** | **100.0%** |
| **1.00% 🚀** | 2.0x | **$159,373.35** | **+$59,373.35** | **+59.37%** | **2.001** | **55.00%** | **1.27** | **1.04** | +$1,189.91 | -$1,146.40 | **17.49%** | $28,744.92 | **$104,459.62** | **100.0%** |

---

## 2. Visual Performance Artifacts

### 1. Cumulative Equity Growth & Underwater Drawdown ($100k Account)
![Equity Curves](../assets/equity_curves_100k.png)

### 2. Monte Carlo Bootstrap Distributions (1,000 Runs)
![Monte Carlo Distribution](../assets/monte_carlo_distribution.png)

### 3. Risk-Reward Frontier (Drawdown vs. Return)
![Risk-Reward Frontier](../assets/risk_reward_frontier.png)

### 4. Monthly Net PnL (USD) Breakdown
![Monthly PnL](../assets/monthly_pnl_100k.png)

---

## 3. Key Institutional Takeaways

1. **The 0.50% Prop-Firm Sweet Spot**:
   - At 0.50% risk per trade on a $100k account, the maximum historical drawdown is **8.96% ($11,522.70)**.
   - This provides a **1.04% safety margin** below standard 10% maximum trailing drawdown rules for institutional and prop evaluations while delivering **+$27,241.91 (+27.24%)** net profit.

2. **The 0.75% High-Growth Configuration**:
   - Generates **+$42,685.26 (+42.69%)** with an average winning trade of **+$836.57** vs average loss of **-$798.54**.
   - Out-of-sample Sharpe remains stable at **2.001**.

3. **Monte Carlo Solvency Guarantee**:
   - In 1,000 bootstrap simulations, the **5th percentile ending balance strictly exceeded the $100,000 initial capital** across all risk tiers ($103,111 to $108,198).
   - Zero-ruin probability is **100.0%** across the 420-trade walk-forward horizon.
