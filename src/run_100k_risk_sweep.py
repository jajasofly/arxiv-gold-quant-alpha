import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup style
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'Segoe UI, Helvetica, Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['grid.color'] = '#222222'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

BASE_DIR = r"C:\Users\ondie\arxiv-gold-quant-alpha"
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

TRADE_CSV = os.path.join(BASE_DIR, "evidence_data", "trades_1_Trade_A_Day_Unified_Architecture_A_50_50_Shared_Netting.csv")
df_trades = pd.read_csv(TRADE_CSV)
df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])

base_ret = df_trades['net_ret_pct'].values
N_TRADES = len(base_ret)
BASE_RISK = 0.0075 # 0.75% base

risk_tiers = [0.0050, 0.0060, 0.0070, 0.0075, 0.0080, 0.0090, 0.0100]
risk_labels = ["0.50%", "0.60%", "0.70%", "0.75%", "0.80%", "0.90%", "1.00%"]
INITIAL_CAPITAL = 100000.0

sweep_results = []
equity_curves = {}
drawdown_curves = {}

np.random.seed(42)

for risk, label in zip(risk_tiers, risk_labels):
    # Scale returns proportionally to risk
    scale = risk / BASE_RISK
    trade_rets = base_ret * scale
    
    # Compound trajectory
    curve = [INITIAL_CAPITAL]
    pnl_list = []
    curr = INITIAL_CAPITAL
    for r in trade_rets:
        pnl = curr * r
        curr += pnl
        pnl_list.append(pnl)
        curve.append(curr)
        
    curve = np.array(curve)
    pnl_arr = np.array(pnl_list)
    
    ending_bal = curve[-1]
    net_profit = ending_bal - INITIAL_CAPITAL
    net_ret_pct = (net_profit / INITIAL_CAPITAL) * 100
    
    # Drawdown calculation
    peaks = np.maximum.accumulate(curve)
    dds_pct = (peaks - curve) / peaks * 100
    dds_usd = peaks - curve
    max_dd_pct = np.max(dds_pct)
    max_dd_usd = np.max(dds_usd)
    
    # Trade statistics
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    win_rate = (len(wins) / N_TRADES) * 100
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    avg_trade = np.mean(pnl_arr)
    profit_factor = np.abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else np.nan
    payoff_ratio = np.abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
    
    # Sharpe ratio (annualized: 420 trades over 21 months = ~240 trades/year)
    ann_factor = np.sqrt(240)
    sharpe = (np.mean(trade_rets) / np.std(trade_rets)) * ann_factor if np.std(trade_rets) > 0 else 0
    
    # Monte Carlo 1,000 bootstrap simulations
    N_SIMS = 1000
    mc_ending = []
    mc_max_dds = []
    for _ in range(N_SIMS):
        boot_rets = np.random.choice(trade_rets, size=N_TRADES, replace=True)
        boot_curve = INITIAL_CAPITAL * np.cumprod(1 + boot_rets)
        boot_peaks = np.maximum.accumulate(np.insert(boot_curve, 0, INITIAL_CAPITAL))
        boot_dds = (boot_peaks - np.insert(boot_curve, 0, INITIAL_CAPITAL)) / boot_peaks * 100
        mc_ending.append(boot_curve[-1])
        mc_max_dds.append(np.max(boot_dds))
        
    mc_ending = np.array(mc_ending)
    mc_max_dds = np.array(mc_max_dds)
    mc_prob_profit = np.mean(mc_ending > INITIAL_CAPITAL) * 100
    mc_5pct_end = np.percentile(mc_ending, 5)
    mc_median_end = np.percentile(mc_ending, 50)
    mc_95pct_end = np.percentile(mc_ending, 95)
    mc_95pct_maxdd = np.percentile(mc_max_dds, 95)
    
    equity_curves[label] = curve
    drawdown_curves[label] = dds_pct
    
    sweep_results.append({
        "Risk_Pct_Per_Trade": label,
        "Starting_Balance_USD": INITIAL_CAPITAL,
        "Ending_Balance_USD": round(ending_bal, 2),
        "Net_Profit_USD": round(net_profit, 2),
        "Net_Return_Pct": round(net_ret_pct, 2),
        "Net_Sharpe": round(sharpe, 3),
        "Profit_Factor": round(profit_factor, 2),
        "Payoff_Ratio": round(payoff_ratio, 2),
        "Win_Rate_Pct": round(win_rate, 2),
        "Avg_Win_USD": round(avg_win, 2),
        "Avg_Loss_USD": round(avg_loss, 2),
        "Avg_Trade_USD": round(avg_trade, 2),
        "Max_Drawdown_USD": round(max_dd_usd, 2),
        "Max_Drawdown_Pct": round(max_dd_pct, 2),
        "MC_Prob_Profit_Pct": round(mc_prob_profit, 1),
        "MC_5pct_End_USD": round(mc_5pct_end, 2),
        "MC_Median_End_USD": round(mc_median_end, 2),
        "MC_95pct_End_USD": round(mc_95pct_end, 2),
        "MC_95pct_MaxDD_Pct": round(mc_95pct_maxdd, 2),
        "Total_Trades": N_TRADES
    })

df_sweep = pd.DataFrame(sweep_results)
df_sweep.to_csv(os.path.join(BASE_DIR, "evidence_data", "champion_100k_risk_sweep.csv"), index=False)
print("=== $100,000 ACCOUNT RISK SWEEP RESULTS ===")
print(df_sweep[["Risk_Pct_Per_Trade", "Ending_Balance_USD", "Net_Profit_USD", "Net_Return_Pct", "Max_Drawdown_Pct", "Max_Drawdown_USD", "MC_5pct_End_USD"]].to_string())

# ==========================================
# CHART 1: EQUITY CURVES & DRAWDOWN (100K)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={'height_ratios': [2.2, 1]}, sharex=True)

colors = {"0.50%": "#00ffcc", "0.75%": "#ffd700", "1.00%": "#ff4d6d"}
linestyles = {"0.50%": "-", "0.75%": "-", "1.00%": "-"}

for lbl in ["0.50%", "0.75%", "1.00%"]:
    c = colors[lbl]
    ax1.plot(range(len(equity_curves[lbl])), equity_curves[lbl], label=f"{lbl} Risk (End: ${equity_curves[lbl][-1]:,.0f})", color=c, lw=2.2)
    ax2.plot(range(len(drawdown_curves[lbl])), -drawdown_curves[lbl], label=f"{lbl} Risk (Max DD: {df_sweep.loc[df_sweep['Risk_Pct_Per_Trade']==lbl, 'Max_Drawdown_Pct'].values[0]:.2f}%)", color=c, lw=1.8)

ax1.axhline(INITIAL_CAPITAL, color='#666666', linestyle=':', label='Starting Balance ($100k)')
ax1.set_title("1-Trade-A-Day Champion: $100k Account Equity Growth (2025–2026 OOS)", fontsize=14, fontweight='bold', pad=12, color='#ffffff')
ax1.set_ylabel("Account Balance (USD)", fontsize=11, color='#dddddd')
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax1.grid(True)
ax1.legend(loc="upper left", framealpha=0.8, facecolor='#18181b', edgecolor='#333333')

ax2.set_title("Underwater Drawdown Trajectory (%)", fontsize=12, fontweight='bold', pad=8, color='#ffffff')
ax2.set_xlabel("Trade Number (1 to 420)", fontsize=11, color='#dddddd')
ax2.set_ylabel("Drawdown (%)", fontsize=11, color='#dddddd')
ax2.axhline(-10.0, color='#ff3333', linestyle='--', alpha=0.8, label='10% Prop Drawdown Limit')
ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}%'))
ax2.grid(True)
ax2.legend(loc="lower left", framealpha=0.8, facecolor='#18181b', edgecolor='#333333')

plt.tight_layout()
chart1_path = os.path.join(ASSETS_DIR, "equity_curves_100k.png")
plt.savefig(chart1_path, dpi=300)
plt.close()
print(f"Saved: {chart1_path}")

# ==========================================
# CHART 2: MONTE CARLO DISTRIBUTION (100K)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: 100 paths for 0.75% baseline
np.random.seed(123)
t_rets_75 = base_ret * (0.0075 / BASE_RISK)
for _ in range(150):
    sim = INITIAL_CAPITAL * np.cumprod(1 + np.random.choice(t_rets_75, size=N_TRADES, replace=True))
    ax1.plot(range(N_TRADES), sim, color='#ffd700', alpha=0.08, lw=1)

# Plot percentiles
all_sims_75 = np.array([INITIAL_CAPITAL * np.cumprod(1 + np.random.choice(t_rets_75, size=N_TRADES, replace=True)) for _ in range(1000)])
p5 = np.percentile(all_sims_75, 5, axis=0)
p50 = np.percentile(all_sims_75, 50, axis=0)
p95 = np.percentile(all_sims_75, 95, axis=0)

ax1.plot(range(N_TRADES), p95, color='#00ffcc', lw=2.2, label=f'95th Percentile (${p95[-1]:,.0f})')
ax1.plot(range(N_TRADES), p50, color='#ffffff', lw=2.2, label=f'Median Path (${p50[-1]:,.0f})')
ax1.plot(range(N_TRADES), p5, color='#ff4d6d', lw=2.2, label=f'5th Percentile (${p5[-1]:,.0f})')
ax1.axhline(INITIAL_CAPITAL, color='#888888', linestyle=':', label='Breakeven ($100k)')
ax1.set_title("1,000 Monte Carlo Bootstrap Paths (0.75% Risk)", fontsize=12, fontweight='bold', pad=10, color='#ffffff')
ax1.set_xlabel("Trades", fontsize=10, color='#dddddd')
ax1.set_ylabel("Account Balance (USD)", fontsize=10, color='#dddddd')
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax1.grid(True)
ax1.legend(loc="upper left", framealpha=0.85, facecolor='#18181b', edgecolor='#333333', fontsize=9)

# Right: Terminal Wealth Histogram for 0.50% vs 0.75% vs 1.00%
sims_end_50 = [INITIAL_CAPITAL * np.prod(1 + np.random.choice(base_ret * (0.005/0.0075), size=N_TRADES, replace=True)) for _ in range(1000)]
sims_end_75 = all_sims_75[:, -1]
sims_end_100 = [INITIAL_CAPITAL * np.prod(1 + np.random.choice(base_ret * (0.010/0.0075), size=N_TRADES, replace=True)) for _ in range(1000)]

ax2.hist(sims_end_50, bins=35, alpha=0.55, color='#00ffcc', label=f'0.50% (5% Tail: ${np.percentile(sims_end_50, 5):,.0f})', edgecolor='none')
ax2.hist(sims_end_75, bins=35, alpha=0.55, color='#ffd700', label=f'0.75% (5% Tail: ${np.percentile(sims_end_75, 5):,.0f})', edgecolor='none')
ax2.hist(sims_end_100, bins=35, alpha=0.45, color='#ff4d6d', label=f'1.00% (5% Tail: ${np.percentile(sims_end_100, 5):,.0f})', edgecolor='none')
ax2.axvline(INITIAL_CAPITAL, color='#ffffff', linestyle='--', lw=1.5, label='Zero Ruin Line ($100k)')
ax2.set_title("Terminal Balance Probability Distributions (1,000 Runs)", fontsize=12, fontweight='bold', pad=10, color='#ffffff')
ax2.set_xlabel("Ending Balance (USD)", fontsize=10, color='#dddddd')
ax2.set_ylabel("Frequency", fontsize=10, color='#dddddd')
ax2.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax2.grid(True)
ax2.legend(loc="upper right", framealpha=0.85, facecolor='#18181b', edgecolor='#333333', fontsize=9)

plt.tight_layout()
chart2_path = os.path.join(ASSETS_DIR, "monte_carlo_distribution.png")
plt.savefig(chart2_path, dpi=300)
plt.close()
print(f"Saved: {chart2_path}")

# ==========================================
# CHART 3: EFFICIENT RISK-REWARD FRONTIER
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

x_dd = df_sweep['Max_Drawdown_Pct'].values
y_ret = df_sweep['Net_Return_Pct'].values
labels = df_sweep['Risk_Pct_Per_Trade'].values

ax.plot(x_dd, y_ret, color='#ffd700', linestyle='-', lw=2, marker='o', markersize=8, markerfacecolor='#00ffcc', markeredgecolor='#ffffff', markeredgewidth=1.5)

for i, txt in enumerate(labels):
    ax.annotate(f" {txt}\n (+${df_sweep['Net_Profit_USD'].iloc[i]:,.0f})", (x_dd[i], y_ret[i]), 
                textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#ffffff', fontweight='bold')

ax.axvline(10.0, color='#ff3333', linestyle='--', alpha=0.85, label='10% Prop Firm Drawdown Limit')
ax.axvspan(0, 10.0, color='#00ffcc', alpha=0.06, label='Prop-Firm Compliant Zone (<10% Max DD)')

ax.set_title("Risk-Reward Efficient Frontier ($100k Account Sweep)", fontsize=13, fontweight='bold', pad=12, color='#ffffff')
ax.set_xlabel("Maximum Historical Drawdown (%)", fontsize=11, color='#dddddd')
ax.set_ylabel("Out-of-Sample Net Return (%)", fontsize=11, color='#dddddd')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}%'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.0f}%'))
ax.grid(True)
ax.legend(loc="lower right", framealpha=0.85, facecolor='#18181b', edgecolor='#333333')

plt.tight_layout()
chart3_path = os.path.join(ASSETS_DIR, "risk_reward_frontier.png")
plt.savefig(chart3_path, dpi=300)
plt.close()
print(f"Saved: {chart3_path}")

# ==========================================
# CHART 4: MONTHLY PNL BREAKDOWN ($100K ACCOUNT AT 0.75%)
# ==========================================
df_trades['month'] = df_trades['exit_time'].dt.to_period('M')
# Compute monthly PnL for 0.75% on 100k
scale_75 = 0.0075 / BASE_RISK
t_rets_75 = base_ret * scale_75
pnl_100k_75 = []
curr = INITIAL_CAPITAL
for r in t_rets_75:
    p = curr * r
    curr += p
    pnl_100k_75.append(p)

df_trades['pnl_usd_100k'] = pnl_100k_75
monthly_pnl = df_trades.groupby('month')['pnl_usd_100k'].sum()

fig, ax = plt.subplots(figsize=(12, 5.5))
months_str = [str(m) for m in monthly_pnl.index]
vals = monthly_pnl.values
bar_colors = ['#00ffcc' if v >= 0 else '#ff4d6d' for v in vals]

bars = ax.bar(months_str, vals, color=bar_colors, edgecolor='#222222', width=0.65)
ax.axhline(0, color='#666666', linestyle='-', lw=1)

for bar, v in zip(bars, vals):
    y_pos = v + (800 if v >= 0 else -1600)
    ax.annotate(f"${v:,.0f}", (bar.get_x() + bar.get_width() / 2, y_pos),
                ha='center', va='bottom' if v < 0 else 'top', fontsize=8, color='#ffffff', fontweight='bold', rotation=45)

ax.set_title("Monthly Net PnL (USD) on $100k Account (0.75% Risk per Trade)", fontsize=13, fontweight='bold', pad=12, color='#ffffff')
ax.set_ylabel("Net PnL (USD)", fontsize=11, color='#dddddd')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax.set_xticklabels(months_str, rotation=45, ha='right', fontsize=9)
ax.grid(True, axis='y')

plt.tight_layout()
chart4_path = os.path.join(ASSETS_DIR, "monthly_pnl_100k.png")
plt.savefig(chart4_path, dpi=300)
plt.close()
print(f"Saved: {chart4_path}")

print("ALL CHARTS GENERATED SUCCESSFULLY!")
