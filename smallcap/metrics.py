"""绩效指标，按德邦研报表1 / 表2 / 表3 的口径。

下面四个公式是从表1 公布的数字反推出来的，并在 tests/test_metrics.py 里
对着那些数字逐一断言：

    夏普      = (年化收益 - rf) / 年化波动     (43.1-2)/32.1 = 1.280
    卡玛      = 年化收益 / 最大回撤             43.1/54.7    = 0.788
    超额收益  = 策略年化 - 基准年化             43.1-9.7     = 33.4
    信息比率  = 超额年化 / 超额波动             33.4/12.6    = 2.651

另有两条口径是拿表1、表2 的**基准行**钉死的——基准行是纯指数序列，不掺任何
组合构造，所以能精确对平（见 tests/test_benchmark.py）：

  * 收益按**自然日**折算，不是交易日。A 股一年只有约 243 个交易日，同一条净值
    曲线用 252 口径会把全时段的 9.7% 抬到 10.1%。
  * 分年度那一列以**上年最后一个交易日**的净值为基准，因此包含 1 月首个交易日
    的收益，而不是在 1 月 1 日重新起算。

波动率则相反，仍按 std × sqrt(252) 年化。

研报对「超额」的口径自相矛盾：超额**收益**是年化值的算术差，超额**回撤**却
用比值净值（策略/基准）算。两者都按原样复现，不做调和——对上研报才是目的。
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
DAYS_PER_YEAR = 365.25

ROWS = [
    "策略年化收益率", "基准年化收益率", "超额年化收益率",
    "策略年化波动率", "基准年化波动率", "超额年化波动率",
    "策略夏普比率(rf=2%)", "基准夏普比率(rf=2%)", "信息比率",
    "策略最大回撤", "策略最大回撤起始", "策略最大回撤终止",
    "基准最大回撤", "基准最大回撤起始", "基准最大回撤终止",
    "超额最大回撤", "超额最大回撤起始", "超额最大回撤终止",
    "策略卡玛比率", "基准卡玛比率", "超额卡玛比率",
]


def annualized_return(nav):
    """净值序列的几何年化收益，按**自然日**天数折算。

    自然日而非交易日——这条是拿表1、表2 的基准行钉死的，见模块说明。
    """
    nav = nav.dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return np.nan
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0:
        return np.nan
    return (nav.iloc[-1] / nav.iloc[0]) ** (DAYS_PER_YEAR / days) - 1


def annualized_vol(returns, trading_days=TRADING_DAYS):
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return r.std(ddof=1) * np.sqrt(trading_days)


def max_drawdown(nav):
    """返回 (最大回撤, 峰值日, 谷底日)；回撤取正值。

    峰值日 / 谷底日 对应研报的「最大回撤起始 / 最大回撤终止」。
    """
    nav = nav.dropna()
    if nav.empty:
        return np.nan, None, None
    drawdown = nav / nav.cummax() - 1.0
    trough = drawdown.idxmin()
    peak = nav.loc[:trough].idxmax()
    return -drawdown.loc[trough], peak, trough


def sharpe(annual_return, annual_vol, rf=0.02):
    if not annual_vol:
        return np.nan
    return (annual_return - rf) / annual_vol


def calmar(annual_return, mdd):
    if not mdd:
        return np.nan
    return annual_return / mdd


def information_ratio(excess_annual, excess_vol):
    if not excess_vol:
        return np.nan
    return excess_annual / excess_vol


def to_returns(nav):
    return nav.dropna().pct_change().dropna()


def performance(nav, benchmark_nav, rf=0.02, trading_days=TRADING_DAYS, risk_start=None):
    """给一对已对齐的 (策略, 基准) 净值算出表1 的全部行。

    `risk_start` 把表1 分年度那列实际用的两个窗口分开：**收益**在整段切片上算
    （切片起点是上年最后一个交易日），**波动与回撤**则只从 `risk_start` 往后算
    （也就是当年的第一个交易日）。

    2016 年是判别性证据。基准的 −20.0% 只有以 2015 年末收盘为基准才对得上；
    而它公布的 31.7% 波动率，只有把 2016-01-04（熔断，−8.66%，该日收益属于
    跨年边界）排除在外才对得上——含进去是 33.0%。
    """
    nav, benchmark_nav = nav.dropna().align(benchmark_nav.dropna(), join="inner")
    nav = nav / nav.iloc[0]
    benchmark_nav = benchmark_nav / benchmark_nav.iloc[0]

    risk_nav = nav if risk_start is None else nav.loc[risk_start:]
    risk_bench = benchmark_nav if risk_start is None else benchmark_nav.loc[risk_start:]

    s_ret, b_ret = to_returns(risk_nav), to_returns(risk_bench)
    s_ann = annualized_return(nav)
    b_ann = annualized_return(benchmark_nav)
    e_ann = s_ann - b_ann

    s_vol = annualized_vol(s_ret, trading_days)
    b_vol = annualized_vol(b_ret, trading_days)
    e_vol = annualized_vol(s_ret - b_ret, trading_days)

    s_mdd, s_peak, s_trough = max_drawdown(risk_nav)
    b_mdd, b_peak, b_trough = max_drawdown(risk_bench)
    e_mdd, e_peak, e_trough = max_drawdown(risk_nav / risk_bench)

    return pd.Series(
        [
            s_ann, b_ann, e_ann,
            s_vol, b_vol, e_vol,
            sharpe(s_ann, s_vol, rf), sharpe(b_ann, b_vol, rf),
            information_ratio(e_ann, e_vol),
            s_mdd, s_peak, s_trough,
            b_mdd, b_peak, b_trough,
            e_mdd, e_peak, e_trough,
            calmar(s_ann, s_mdd), calmar(b_ann, b_mdd), calmar(e_ann, e_mdd),
        ],
        index=ROWS,
    )


def performance_table(nav, benchmark_nav, rf=0.02, trading_days=TRADING_DAYS):
    """表1 的版式：每个自然年一列，外加「全时段」一列。

    每年都从上年最后一个交易日的收盘起算，所以 1 月首个交易日的收益算在当年
    ——这正是表1 基准行遵循的口径。表1 本身略去了 2012-12 那个不足一月的零头，
    这里选择保留而不是丢弃；但要注意，一个不足一月的窗口按自然日年化之后跟
    整年不可比，读的时候心里有数。
    """
    nav, benchmark_nav = nav.dropna().align(benchmark_nav.dropna(), join="inner")
    columns = {}
    for year in sorted(set(nav.index.year)):
        in_year = nav.index[nav.index.year == year]
        before = nav.index[nav.index < in_year[0]]
        start = before[-1] if len(before) else in_year[0]
        seg = nav.loc[start:in_year[-1]]
        if len(seg) < 2:
            continue
        columns[str(year)] = performance(
            seg, benchmark_nav.loc[seg.index], rf, trading_days, risk_start=in_year[0]
        )
    columns["全时段"] = performance(nav, benchmark_nav, rf, trading_days)
    return pd.DataFrame(columns)
