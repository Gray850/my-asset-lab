import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 语言文本字典 ──────────────────────────────────────────────

DATA_LANG = {
    "中文": {
        "page_title":        "My Asset Lab",
        "app_title":         "My Asset Lab",
        "sidebar_header":    "参数设置",
        "select_assets":     "选择资产",
        "time_range":        "时间范围",
        "no_assets":         "请在左侧至少选择一个资产",
        "weights_header":    "自定义权重（%）",
        "weight_error":      "权重之和 = {total:.1f}%，需要等于 100%",
        "fee_header":        "交易费率模拟",
        "fee_label":         "单笔调仓费率 (%)",
        "rebalance_label":   "再平衡频率",
        "rebalance_opts":    ["每日再平衡", "买入并持有（无再平衡）"],
        "run_button":        "运行分析",
        "waiting":           "在左侧设置参数后点击「运行分析」",
        "loading":           "下载数据中...",
        "load_error":        "数据下载失败，请检查 ticker 是否正确",
        "rf_caption":        "当前无风险利率（美国3个月国债）：{rf:.2%}  ·  所有夏普比率已扣除",
        "table_header":      "指标对比表",
        "col_ticker":        "Ticker",
        "col_total_ret":     "Total Return (%)",
        "col_max_dd":        "Max Drawdown (%)",
        "col_vol":           "Volatility (%)",
        "col_sharpe":        "Sharpe Ratio",
        "col_recovery":      "解套天数",
        "not_recovered":     ">{days}（未解套）",
        "worst_info":        "**{name}** 在此期间经历了解套耗时最长的考验，共计 **{days}** 个交易日，请注意该仓位的心理耐受度。",
        "drawdown_title":    "回撤走势图",
        "drawdown_ylabel":   "回撤幅度 (%)",
        "portfolio_title":   "组合 vs 单资产走势",
        "portfolio_ylabel":  "归一化净值",
        "label_ew":          "等权组合",
        "label_cp":          "自定义组合",
        "ef_subheader":      "有效前沿（Efficient Frontier）",
        "ef_title":          "有效前沿（5000 个随机组合）",
        "ef_xlabel":         "年化波动率 (%)",
        "ef_ylabel":         "年化收益率 (%)",
        "ef_colorbar":       "夏普比率",
        "ef_max_sharpe":     "最优夏普 ({sharpe:.2f})",
        "ef_min_vol":        "最低波动 ({vol:.1f}%)",
        "ef_cp":             "自定义组合",
        "ef_ms_title":       "**最优夏普权重**",
        "ef_mv_title":       "**最低波动权重**",
        "mc_subheader":      "未来 1 年走势模拟预测（蒙特卡洛）",
        "mc_title":          "自定义组合未来 252 个交易日模拟路径（100 条）",
        "mc_xlabel":         "未来交易日（天）",
        "mc_ylabel":         "预测净值",
        "mc_median":         "中位数路径",
    },
    "English": {
        "page_title":        "My Asset Lab",
        "app_title":         "My Asset Lab",
        "sidebar_header":    "Settings",
        "select_assets":     "Select Assets",
        "time_range":        "Time Range",
        "no_assets":         "Please select at least one asset on the left.",
        "weights_header":    "Custom Weights (%)",
        "weight_error":      "Weights sum = {total:.1f}%, must equal 100%",
        "fee_header":        "Transaction Cost Simulation",
        "fee_label":         "Fee per Rebalance (%)",
        "rebalance_label":   "Rebalancing Frequency",
        "rebalance_opts":    ["Daily Rebalancing", "Buy and Hold (No Rebalancing)"],
        "run_button":        "Run Analysis",
        "waiting":           "Configure parameters on the left, then click Run Analysis.",
        "loading":           "Downloading data...",
        "load_error":        "Failed to download data. Please check ticker symbols.",
        "rf_caption":        "Risk-free rate (US 3-month T-bill): {rf:.2%}  ·  All Sharpe ratios are excess returns.",
        "table_header":      "Metrics Comparison",
        "col_ticker":        "Ticker",
        "col_total_ret":     "Total Return (%)",
        "col_max_dd":        "Max Drawdown (%)",
        "col_vol":           "Volatility (%)",
        "col_sharpe":        "Sharpe Ratio",
        "col_recovery":      "Recovery Days",
        "not_recovered":     ">{days} (not recovered)",
        "worst_info":        "**{name}** had the longest recovery period — **{days}** trading days underwater. Mind your position sizing.",
        "drawdown_title":    "Drawdown Chart",
        "drawdown_ylabel":   "Drawdown (%)",
        "portfolio_title":   "Portfolio vs Individual Assets",
        "portfolio_ylabel":  "Normalized Value",
        "label_ew":          "Equal-weight",
        "label_cp":          "Custom Portfolio",
        "ef_subheader":      "Efficient Frontier",
        "ef_title":          "Efficient Frontier (5,000 random portfolios)",
        "ef_xlabel":         "Annualized Volatility (%)",
        "ef_ylabel":         "Annualized Return (%)",
        "ef_colorbar":       "Sharpe Ratio",
        "ef_max_sharpe":     "Max Sharpe ({sharpe:.2f})",
        "ef_min_vol":        "Min Volatility ({vol:.1f}%)",
        "ef_cp":             "Custom Portfolio",
        "ef_ms_title":       "**Max Sharpe Weights**",
        "ef_mv_title":       "**Min Volatility Weights**",
        "mc_subheader":      "1-Year Monte Carlo Simulation",
        "mc_title":          "Custom Portfolio — 252 Trading Days, 100 Simulated Paths",
        "mc_xlabel":         "Future Trading Days",
        "mc_ylabel":         "Projected Value",
        "mc_median":         "Median Path",
    },
}

# ── 计算函数 ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_prices(tickers_tuple, period):
    raw = yf.download(list(tickers_tuple), period=period, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers_tuple[0])
    return close.ffill().bfill()

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        irx = yf.download("^IRX", period="5d", progress=False)["Close"]
        if hasattr(irx, "columns"):
            irx = irx.iloc[:, 0]
        return float(irx.dropna().iloc[-1]) / 100
    except Exception:
        return 0.0

def calculate_total_return(price):
    if price is None or len(price) == 0:
        return 0.0
    return (price.iloc[-1] / price.iloc[0] - 1) * 100

def calculate_max_drawdown(price):
    if price is None or len(price) == 0:
        return 0.0
    return (price / price.cummax() - 1).min() * 100

def calculate_recovery_days(price):
    if price is None or len(price) == 0:
        return 0, True
    underwater = price < price.cummax()
    max_days = current = 0
    for uw in underwater:
        if uw:
            current += 1
        else:
            max_days = max(max_days, current)
            current = 0
    still_under = current > 0
    max_days = max(max_days, current)
    return max_days, not still_under

def calculate_sharpe_ratio(price, rf_annual=0.0):
    r = price.pct_change().dropna()
    excess = r - rf_annual / 252
    return excess.mean() / excess.std() * (252 ** 0.5)

def calculate_volatility(price):
    return price.pct_change().dropna().std() * (252 ** 0.5) * 100

def portfolio_metrics(daily_returns, weights, rf_annual=0.0, fee_rate=0.0, buy_and_hold=False):
    if daily_returns is None or len(daily_returns) == 0:
        return {
            "Total Return (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Volatility (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "_value": pd.Series(dtype=float),
        }
    if buy_and_hold:
        norm_prices = (1 + daily_returns).cumprod()
        port_value = (norm_prices * weights).sum(axis=1)
        port_daily = port_value.pct_change().dropna()
    else:
        daily_rets = []
        for t in range(len(daily_returns)):
            r = daily_returns.iloc[t].values
            gross = float(weights @ r)
            if fee_rate > 0 and gross > -1:
                w_drift = weights * (1 + r) / (1 + gross)
                turnover = np.abs(w_drift - weights).sum()
                daily_rets.append(gross - fee_rate * turnover)
            else:
                daily_rets.append(gross)
        port_daily = pd.Series(daily_rets, index=daily_returns.index)
        port_value = (1 + port_daily).cumprod()

    excess = port_daily - rf_annual / 252
    return {
        "Total Return (%)": calculate_total_return(port_value),
        "Max Drawdown (%)": calculate_max_drawdown(port_value),
        "Volatility (%)": port_daily.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": excess.mean() / excess.std() * (252 ** 0.5),
        "_value": port_value,
    }

def efficient_frontier(daily_returns, rf_annual=0.0, n=5000):
    n_assets = daily_returns.shape[1]
    mean_returns = daily_returns.mean() * 252
    cov = daily_returns.cov() * 252
    rets, vols, sharpes, all_weights = [], [], [], []
    rng = np.random.default_rng(42)
    for _ in range(n):
        w = rng.random(n_assets)
        w /= w.sum()
        r = float(mean_returns @ w)
        v = float(np.sqrt(w @ cov.values @ w))
        s = (r - rf_annual) / v if v > 0 else 0
        rets.append(r * 100)
        vols.append(v * 100)
        sharpes.append(s)
        all_weights.append(w)
    rets = np.array(rets)
    vols = np.array(vols)
    sharpes = np.array(sharpes)
    all_weights = np.array(all_weights)
    max_sharpe_idx = sharpes.argmax()
    min_vol_idx = vols.argmin()
    return {
        "vols": vols, "rets": rets, "sharpes": sharpes, "weights": all_weights,
        "max_sharpe": {"vol": vols[max_sharpe_idx], "ret": rets[max_sharpe_idx],
                       "sharpe": sharpes[max_sharpe_idx], "weights": all_weights[max_sharpe_idx]},
        "min_vol":    {"vol": vols[min_vol_idx],    "ret": rets[min_vol_idx],
                       "sharpe": sharpes[min_vol_idx],    "weights": all_weights[min_vol_idx]},
    }

# ── 页面配置 & 语言选择 ───────────────────────────────────────

st.set_page_config(page_title="My Asset Lab", layout="wide")

lang = st.sidebar.selectbox("Language / 语言", ["中文", "English"])
T = DATA_LANG[lang]

st.title(T["app_title"])

# ── 侧边栏 ────────────────────────────────────────────────────

st.sidebar.header(T["sidebar_header"])

ASSET_POOL = ["QQQ", "GLD", "AAPL", "IBB", "NVDA", "MSFT", "TSLA", "AMZN"]

tickers = st.sidebar.multiselect(T["select_assets"], options=ASSET_POOL, default=["QQQ", "GLD", "AAPL", "IBB"])
period = st.sidebar.selectbox(T["time_range"], ["6mo", "1y", "2y", "3y", "5y"], index=1)

if not tickers:
    st.warning(T["no_assets"])
    st.stop()

st.sidebar.subheader(T["weights_header"])
weight_inputs = {}
for ticker in tickers:
    default = round(100 / len(tickers), 1)
    weight_inputs[ticker] = st.sidebar.number_input(ticker, min_value=0.0, max_value=100.0, value=default, step=1.0)

total_weight = sum(weight_inputs.values())
if abs(total_weight - 100) > 0.01:
    st.sidebar.error(T["weight_error"].format(total=total_weight))
    st.stop()

st.sidebar.subheader(T["fee_header"])
fee_rate = st.sidebar.slider(T["fee_label"], 0.0, 0.3, 0.05, step=0.01) / 100
rebalance_sel = st.sidebar.selectbox(T["rebalance_label"], T["rebalance_opts"])
buy_and_hold = (rebalance_sel == T["rebalance_opts"][1])

run = st.sidebar.button(T["run_button"], type="primary")

if not run:
    st.info(T["waiting"])
    st.stop()

# ── 下载数据 ──────────────────────────────────────────────────

with st.spinner(T["loading"]):
    prices = load_prices(tuple(tickers), period)
    rf = get_risk_free_rate()

# 去掉全为零的列和全为 NaN 的行，再丢弃最后可能未完全开盘的行
prices = prices.loc[:, (prices != 0).any(axis=0)]
prices = prices.dropna(how="all")
if len(prices) > 1:
    prices = prices.iloc[:-1] if prices.iloc[-1].isna().any() else prices

if prices.empty:
    st.error(T["load_error"])
    st.stop()

# ── 计算指标 ──────────────────────────────────────────────────

daily_returns = prices.pct_change().dropna()
if len(daily_returns) < 2:
    st.error(T["load_error"])
    st.stop()

results = []
for ticker in tickers:
    price = prices[ticker]
    days, rec = calculate_recovery_days(price)
    not_rec_str = T["not_recovered"].format(days=days)
    results.append({
        T["col_ticker"]:    ticker,
        T["col_total_ret"]: calculate_total_return(price),
        T["col_max_dd"]:    calculate_max_drawdown(price),
        T["col_vol"]:       calculate_volatility(price),
        T["col_sharpe"]:    calculate_sharpe_ratio(price, rf),
        T["col_recovery"]:  str(days) if rec else not_rec_str,
        "_recovery_num":    days,
    })

ew_weights = np.array([1 / len(tickers)] * len(tickers))
ew = portfolio_metrics(daily_returns, ew_weights, rf, fee_rate, buy_and_hold)

custom_weights = np.array([weight_inputs[t] / 100 for t in tickers])
cp = portfolio_metrics(daily_returns, custom_weights, rf, fee_rate, buy_and_hold)

for label, m, (days, rec) in [
    (T["label_ew"], ew, calculate_recovery_days(ew["_value"])),
    (T["label_cp"], cp, calculate_recovery_days(cp["_value"])),
]:
    not_rec_str = T["not_recovered"].format(days=days)
    results.append({
        T["col_ticker"]:    label,
        T["col_total_ret"]: m["Total Return (%)"],
        T["col_max_dd"]:    m["Max Drawdown (%)"],
        T["col_vol"]:       m["Volatility (%)"],
        T["col_sharpe"]:    m["Sharpe Ratio"],
        T["col_recovery"]:  str(days) if rec else not_rec_str,
        "_recovery_num":    days,
    })

result_table = pd.DataFrame(results).sort_values(T["col_sharpe"], ascending=False)
ef = efficient_frontier(daily_returns, rf)

# ── 展示结果 ──────────────────────────────────────────────────

st.caption(T["rf_caption"].format(rf=rf))

st.subheader(T["table_header"])
display_table = result_table.drop(columns=["_recovery_num"]).round(2)
st.dataframe(display_table, use_container_width=True, hide_index=True)

worst = result_table.loc[result_table["_recovery_num"].idxmax()]
st.info(T["worst_info"].format(name=worst[T["col_ticker"]], days=worst[T["col_recovery"]]))

# ── 图1、图2：回撤 + 走势 ─────────────────────────────────────

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(len(tickers))]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

for ticker, color in zip(tickers, colors):
    price = prices[ticker]
    drawdown = (price / price.cummax() - 1) * 100
    ax1.plot(drawdown.index, drawdown, label=ticker, color=color, linewidth=1.5)
    ax1.fill_between(drawdown.index, drawdown, 0, alpha=0.12, color=color)

ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_title("Drawdown Chart")
ax1.set_ylabel("Drawdown (%)")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

normalized = prices / prices.iloc[0]
for ticker, color in zip(tickers, colors):
    ax2.plot(normalized.index, normalized[ticker], label=ticker, color=color, linewidth=1.5, alpha=0.75)

ax2.plot(ew["_value"].index, ew["_value"], label="Equal-weight", linewidth=2.5, linestyle="--", color="gray")
ax2.plot(cp["_value"].index, cp["_value"], label="Custom Portfolio", linewidth=3, color="black")

ax2.set_title("Portfolio vs Individual Assets")
ax2.set_ylabel("Normalized Value")
ax2.legend()
ax2.grid(True, alpha=0.25)

plt.tight_layout()
st.pyplot(fig1)

# ── 图3：有效前沿 ─────────────────────────────────────────────

st.subheader(T["ef_subheader"])

ms = ef["max_sharpe"]
mv = ef["min_vol"]

fig2, ax = plt.subplots(figsize=(10, 5))

sc = ax.scatter(ef["vols"], ef["rets"], c=ef["sharpes"], cmap="RdYlGn", s=6, alpha=0.5, linewidths=0)
plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

ax.scatter(ms["vol"], ms["ret"], marker="*", s=400, color="#E74C3C", zorder=5,
           label=f"Max Sharpe ({ms['sharpe']:.2f})")
ax.scatter(mv["vol"], mv["ret"], marker="*", s=400, color="#3498DB", zorder=5,
           label=f"Min Volatility ({mv['vol']:.1f}%)")

cp_ef_ret = (daily_returns @ custom_weights).mean() * 252 * 100
ax.scatter(cp["Volatility (%)"], cp_ef_ret, marker="D", s=120, color="black", zorder=5, label="Custom Portfolio")

ax.set_xlabel("Annualized Volatility (%)")
ax.set_ylabel("Annualized Return (%)")
ax.set_title("Efficient Frontier (5,000 random portfolios)")
ax.legend()
ax.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig2)

col_ms, col_mv = st.columns(2)

with col_ms:
    st.markdown(T["ef_ms_title"])
    for t, w in zip(tickers, ef["max_sharpe"]["weights"]):
        st.write(f"{t}：{w:.1%}")
    st.caption(f"Sharpe {ms['sharpe']:.2f}  ·  Return {ms['ret']:.1f}%  ·  Vol {ms['vol']:.1f}%")

with col_mv:
    st.markdown(T["ef_mv_title"])
    for t, w in zip(tickers, ef["min_vol"]["weights"]):
        st.write(f"{t}：{w:.1%}")
    st.caption(f"Sharpe {mv['sharpe']:.2f}  ·  Return {mv['ret']:.1f}%  ·  Vol {mv['vol']:.1f}%")

# ── 蒙特卡洛未来净值模拟 ──────────────────────────────────────

st.subheader(T["mc_subheader"])

mu = cp_ef_ret / 100          # 年化收益率
sigma = cp["Volatility (%)"] / 100  # 年化波动率
dt = 1 / 252
n_days = 252
n_paths = 100

rng = np.random.default_rng(0)
Z = rng.standard_normal((n_paths, n_days))
daily_factor = np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

paths = np.ones((n_paths, n_days + 1))
paths[:, 1:] = daily_factor.cumprod(axis=1)

median_path = np.median(paths, axis=0)
days_axis = np.arange(n_days + 1)

fig3, ax3 = plt.subplots(figsize=(12, 5))

for i in range(n_paths):
    ax3.plot(days_axis, paths[i], color="#3498DB", alpha=0.15, linewidth=0.8)

ax3.plot(days_axis, median_path, color="black", linewidth=2.5, label="Median Path")

ax3.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
ax3.set_title("Custom Portfolio — 252 Trading Days, 100 Simulated Paths")
ax3.set_xlabel("Future Trading Days")
ax3.set_ylabel("Projected Value")
ax3.legend()
ax3.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig3)