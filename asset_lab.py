import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_close_price(ticker, period="1y"):
    data = yf.download(ticker, period=period, auto_adjust=True)

    close = data["Close"]

    # yfinance 有时候会返回二维数据，这里压成一维
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]

    close = close.dropna()
    return close


def calculate_total_return(price):
    total_return = price.iloc[-1] / price.iloc[0] - 1
    return total_return * 100


def calculate_max_drawdown(price):
    running_max = price.cummax()
    drawdown = price / running_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown * 100

def calculate_volatility(price):
    daily_return = price.pct_change().dropna()
    volatility = daily_return.std() * (252 ** 0.5)
    return volatility * 100

def calculate_sharpe_ratio(price):
    daily_return = price.pct_change().dropna()
    sharpe_ratio = daily_return.mean() / daily_return.std() * (252 ** 0.5)
    return sharpe_ratio

def validate_weights(weight_dict):
    total = sum(weight_dict.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"权重之和必须为 100%，当前为 {total:.2%}")

def portfolio_metrics(daily_returns, weights):
    port_daily = daily_returns @ weights
    port_value = (1 + port_daily).cumprod()
    return {
        "Total Return (%)": calculate_total_return(port_value),
        "Max Drawdown (%)": calculate_max_drawdown(port_value),
        "Volatility (%)": port_daily.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": port_daily.mean() / port_daily.std() * (252 ** 0.5),
        "_value": port_value,
    }


# 自定义权重（修改这里调整仓位）
custom_weight_dict = {
    "QQQ":  0.25,
    "GLD":  0.25,
    "AAPL": 0.25,
    "IBB":  0.25,
}
validate_weights(custom_weight_dict)


tickers = ["QQQ", "GLD", "AAPL", "IBB"]
period = "1y"

# 一次性下载所有资产价格
prices = pd.DataFrame()

for ticker in tickers:
    prices[ticker] = get_close_price(ticker, period=period)

prices = prices.dropna()

results = []

for ticker in tickers:
    price = prices[ticker]

    total_return = calculate_total_return(price)
    max_drawdown = calculate_max_drawdown(price)
    volatility = calculate_volatility(price)
    sharpe_ratio = calculate_sharpe_ratio(price)

    results.append({
        "Ticker": ticker,
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_drawdown,
        "Volatility (%)": volatility,
        "Sharpe Ratio": sharpe_ratio
    })

result_table = pd.DataFrame(results)

# 按 Sharpe Ratio 从高到低排序
result_table = result_table.sort_values(by="Sharpe Ratio", ascending=False)

print(result_table.round(2))

daily_returns = prices.pct_change().dropna()
corr_matrix = daily_returns.corr()

# =========================
# Equal-weight Portfolio
# =========================
ew_weights = np.array([1 / len(tickers)] * len(tickers))
ew = portfolio_metrics(daily_returns, ew_weights)

# =========================
# Custom Portfolio
# =========================
custom_weights = np.array([custom_weight_dict[t] for t in tickers])
cp = portfolio_metrics(daily_returns, custom_weights)

print("\nEqual-weight Portfolio:")
print(f"  Total Return: {ew['Total Return (%)']:.2f}%  |  Max Drawdown: {ew['Max Drawdown (%)']:.2f}%  |  Volatility: {ew['Volatility (%)']:.2f}%  |  Sharpe: {ew['Sharpe Ratio']:.2f}")

print("\nCustom Portfolio:")
print(f"  Total Return: {cp['Total Return (%)']:.2f}%  |  Max Drawdown: {cp['Max Drawdown (%)']:.2f}%  |  Volatility: {cp['Volatility (%)']:.2f}%  |  Sharpe: {cp['Sharpe Ratio']:.2f}")

# 把两个组合追加进表格
for label, m in [("Equal-weight", ew), ("Custom Portfolio", cp)]:
    results.append({
        "Ticker": label,
        "Total Return (%)": m["Total Return (%)"],
        "Max Drawdown (%)": m["Max Drawdown (%)"],
        "Volatility (%)": m["Volatility (%)"],
        "Sharpe Ratio": m["Sharpe Ratio"],
    })

result_table = pd.DataFrame(results).sort_values(by="Sharpe Ratio", ascending=False)

print("\n--- 完整对比表 ---")
print(result_table.round(2).to_string(index=False))

print("\nCorrelation Matrix:")
print(corr_matrix.round(2))

# 保存分析结果
result_table.to_csv("asset_analysis.csv", index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# =========================
# Plot 1: Drawdown Chart
# =========================
colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

for ticker, color in zip(tickers, colors):
    price = prices[ticker]
    running_max = price.cummax()
    drawdown = (price / running_max - 1) * 100

    ax1.plot(drawdown.index, drawdown, label=ticker, color=color, linewidth=1.5)
    ax1.fill_between(drawdown.index, drawdown, 0, alpha=0.12, color=color)

ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_xlabel("Date")
ax1.set_ylabel("Drawdown (%)")
ax1.set_title("Drawdown Chart")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.2)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

# =========================
# Plot 2: Portfolio vs Assets
# =========================
normalized_prices = prices / prices.iloc[0]

for ticker in tickers:
    ax2.plot(
        normalized_prices.index,
        normalized_prices[ticker],
        label=ticker,
        linewidth=1.6,
        alpha=0.75
    )

ax2.plot(ew["_value"].index, ew["_value"], label="Equal-weight", linewidth=2.5, linestyle="--")
ax2.plot(cp["_value"].index, cp["_value"], label="Custom Portfolio", linewidth=3)

ax2.set_title("Portfolio vs Individual Assets")
ax2.set_xlabel("Date")
ax2.set_ylabel("Normalized Value")
ax2.legend()
ax2.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()