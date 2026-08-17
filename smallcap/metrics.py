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


# --------------------------------------------------------------------------- 专题之二指标
# 研报 p.6 §1.1 定义了三个专题之二专用指标：月度胜率、策略平均分位数、基准平均分位数。
# 口径歧义按 grill_enhance.md E5/E9 固定：全市场 = 当期有有效收益的全部 A 股；基准分位数
# 用中证1000 指数自身收益的横截面百分位（非成分股均值，避开 2014 前成分股缺口）；择时
# 空仓期不计入胜率与分位数（无选股可评）。持有期 = 成交日到下一成交日（T+1 开盘口径）。

def _boundary_returns(series, boundaries):
    """把一条日度序列在 `boundaries`（成交日）上采样成逐期简单收益，**按期起点标记**。

    第 i 期收益 = series[boundaries[i+1]] / series[boundaries[i]] − 1，标在 boundaries[i]
    上；最后一个边界没有下一期，为 NaN 后丢弃。
    """
    sampled = series.reindex(pd.DatetimeIndex(boundaries))
    return (sampled.shift(-1) / sampled - 1).iloc[:-1]


def win_rate(strat_nav, benchmark_nav, trade_dates, cash_trade_dates=None):
    """月度/周度胜率：每期策略收益 > 基准收益的比例（研报 p.6）。

    `trade_dates` 是成交日序列（期边界）。`cash_trade_dates` 给出的期起点是择时空仓期，
    按 E9 从分母剔除——没有选股就无从谈「选股跑赢基准」。
    """
    s = _boundary_returns(strat_nav, trade_dates)
    b = _boundary_returns(benchmark_nav, trade_dates)
    s, b = s.align(b, join="inner")
    if cash_trade_dates is not None:
        s = s[~s.index.isin(pd.DatetimeIndex(cash_trade_dates))]
        b = b.reindex(s.index)
    return float((s > b).mean())


def _period_stock_returns(post_close, trade_dates):
    """全市场逐期收益宽表（行 = 期起点成交日，列 = 股票），口径同 `_boundary_returns`。"""
    px = post_close.reindex(pd.DatetimeIndex(trade_dates))
    return (px.shift(-1) / px - 1).iloc[:-1]


def avg_percentile(selection, post_close, schedule):
    """策略平均分位数（研报 p.6，E5）：每期**组合等权收益**在全市场股票收益横截面里的
    百分位，逐（非空仓）期平均。越接近 1.0 越好。

    取「组合收益的百分位」而非「个股百分位均值」，是为了与 `benchmark_percentile`
    对称——两者都把一个组合/指数的单一收益放进横截面排名。**实测残差**：本口径基准
    月频 ≈63%，研报 74.14%，系统性偏低约 11pp；个股百分位均值更低（≈50%）。这个缺口
    是 Wind 分位数方法学与研报文字不足以复原的差异，如实记录（grill_enhance.md E5 /
    「实施中的发现」），不挑口径去凑。

    `selection` 行 = 信号日；`schedule` = bt.trade_dates(...) 的 signal→trade 映射。
    空仓期（当期一只都没选）自然被跳过。
    """
    schedule = schedule.reindex(selection.index).dropna()
    period_ret = _period_stock_returns(post_close, pd.DatetimeIndex(schedule.values))
    vals = []
    for signal, trade in schedule.items():
        held = selection.loc[signal]
        ids = held[held].index
        if trade in period_ret.index and len(ids):
            row = period_ret.loc[trade]
            port = row[row.index.intersection(ids)].mean()
            cross = row.dropna()
            if len(cross) and pd.notna(port):
                vals.append(float((cross < port).mean()))
    return float(pd.Series(vals, dtype=float).mean())


def benchmark_percentile(benchmark_close, post_close, trade_dates):
    """基准平均分位数（研报 p.6，E5）：中证1000 指数每期收益在全市场股票收益横截面里
    的百分位，逐期平均。用指数自身收益（单实体），不用成分股均值——避开成分股 2014
    前缺口，且贴合「基准=指数」的字面。
    """
    tds = pd.DatetimeIndex(trade_dates)
    stock_ret = _period_stock_returns(post_close, tds)
    bench_ret = _boundary_returns(benchmark_close, tds)
    vals = []
    for trade, br in bench_ret.items():
        if trade in stock_ret.index and pd.notna(br):
            row = stock_ret.loc[trade].dropna()
            if len(row):
                vals.append(float((row < br).mean()))
    return float(pd.Series(vals, dtype=float).mean())


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
