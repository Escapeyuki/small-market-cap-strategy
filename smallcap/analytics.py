"""研报 4.3–4.9 节的诊断统计 —— 纯计算，绘图在 plots.py。

每个函数上面标着它复现的是哪张图，不必回去翻研报才知道这段代码在算什么。

三条贯穿全模块的约定：

  * **成员一律用布尔宽表表示**（行 = 日期，列 = order_book_id），与
    `universe.smallest` 的返回值同型。指数成分也先转成这个形状，组合与指数
    才走得上同一条代码路径 —— 图19/20–24 的对比要有意义，两边的算法必须
    逐字相同。
  * **在调仓日上算的**（基尼、分位数漂移、调出归因）取 `selection` 作输入；
    **在每个交易日上算的**（PB、市值中位数、换手率）取 `daily(selection)`。
  * **缺数据就让它缺**。中证1000 成分股最早只到 2014-10-17，返回的序列在那
    之前就是没有这一段，不用 0 或前值填。图上是个缺口，不是一条假的线。

「整体法」在研报里出现两次（PB、换手率），指的都是**先各自加总再相除**，
而不是对个股比值取平均。两处的实现都写在下面，各自注明了分母是怎么来的。
"""
import bisect

import numpy as np
import pandas as pd

from . import universe as u

ENTRY, EXIT = "建仓", "卖出"

# 调出原因，按判定优先级排列。研报 4.7 节只给了三类，前两类与最后一类正是
# 它的「退市 / 戴帽 / 市值上涨」；中间两类是研报没有单列的——它们来自研报
# 自己写明的「调仓日排除涨停、不交易的股票」这条过滤。见 `exit_reasons`。
REASONS = ["退市", "戴帽", "停牌", "涨停", "市值上涨"]

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五"]


# --------------------------------------------------------------------------- 成员

def daily(selection, sessions):
    """调仓日的持仓 → 逐日持仓，两次调仓之间保持不变。

    布尔宽表 reindex 之后会退化成 object dtype，所以中间过一道 float 再比回
    布尔，免得下游的 `where(members)` 拿到 NaN 当真值用。
    """
    filled = selection.astype(float).reindex(pd.DatetimeIndex(sessions)).ffill()
    return filled.fillna(0.0) > 0


def index_members(components, index_id):
    """index_components 长表 → 布尔宽表，与组合持仓同型。

    只覆盖该指数**确实有成分数据**的日期。中证1000 在 2014-10-17 之前直接
    没有行，于是所有下游统计在那段时间自然是空的 —— 这个缺口是真实的
    （指数行情回补到了 2007 年，成分股没有），标注它，不要回补。
    """
    rows = components[components["index_id"] == index_id]
    if rows.empty:
        return pd.DataFrame(dtype=bool)
    return (
        rows.assign(member=True)
        .pivot(index="date", columns="order_book_id", values="member")
        .eq(True)
        .sort_index()
    )


def _aligned(values, members):
    """把一张日期 × 股票的数值表对齐到成员表，并只留下成员那些格子。"""
    return values.reindex(index=members.index, columns=members.columns).where(members)


# --------------------------------------------------------------------------- 图26

def membership_changes(selection):
    """逐个调仓日的建仓（新进组合）与卖出（调出组合）事件。

    研报 4.4 节：「每当有新的股票进入组合时，记一次建仓，每当有股票退出组合
    时，记一次卖出」。首个调仓日的全部持仓都算建仓。
    """
    rows = []
    previous = pd.Series(False, index=selection.columns)
    for date, held in selection.iterrows():
        held = held.astype(bool)
        rows += [(date, sid, ENTRY) for sid in selection.columns[held & ~previous]]
        rows += [(date, sid, EXIT) for sid in selection.columns[previous & ~held]]
        previous = held
    return pd.DataFrame(rows, columns=["date", "order_book_id", "event"])


def exits(selection):
    """只要调出事件。研报 4.7 节统计到的 2401 次说的就是这个。"""
    changes = membership_changes(selection)
    return changes[changes["event"] == EXIT]


def exit_reasons(selection, panel):
    """图26 —— 每只股票被调出组合的原因，返回 (每期计数表, 明细表)。

    分类不是另起一套判据，而是**直接复用把它挤出去的那几条谓词**：逐条问
    「这只股票在调仓日是哪条过滤没通过」，都通过了就只剩市值涨出了前 100 名。
    这样分类天然是穷尽且互斥的，不会出现一个说不清归属的余项。

    研报 4.7 节只列了三类：市值上涨 97.17% / 戴帽 2.75% / 退市 0.08%。而它
    自己 3 节的选股范围里还写着「在每个调仓日，排除当天涨停、不交易的股票」
    —— 那两条同样会把一只在持股票挤出新组合。研报没有把它们单列，说明它的
    统计口径要么不含这两类，要么把它们并进了「市值上涨」。这里如实分五类，
    `collapse_reasons` 再折叠成研报的三类作对照。
    """
    failures = [
        ("退市", ~u.has_price(panel)),          # 已无市值：退市或尚未上市
        ("戴帽", ~u.not_st(panel)),
        ("停牌", ~u.trading(panel)),
        ("涨停", ~u.not_limit_up(panel)),
    ]

    def classify(date, sid):
        for label, failed in failures:
            if sid in failed.columns and bool(failed.loc[date, sid]):
                return label
        return "市值上涨"

    detail = exits(selection).copy()
    detail["reason"] = [classify(d, s) for d, s in
                        zip(detail["date"], detail["order_book_id"])]
    counts = (
        detail.pivot_table(index="date", columns="reason", aggfunc="size", fill_value=0)
        .reindex(columns=REASONS, fill_value=0)
    )
    return counts, detail


def collapse_reasons(counts):
    """折叠成研报 4.7 节的三类。涨停与停牌并入「市值上涨」。

    并入的理由：涨停当天的股票是**涨上去**的，与研报「市值上涨」同因；停牌
    则纯粹是本项目的过滤链造成的，研报既然没有单列，它的 2401 次里必然也
    不含这一类。两者合计占比若很小，这次折叠就不影响结论——占比多大，跑出来
    的数说了算。
    """
    return pd.DataFrame({
        "退市": counts["退市"],
        "戴帽": counts["戴帽"],
        "市值上涨": counts["市值上涨"] + counts["涨停"] + counts["停牌"],
    })


# --------------------------------------------------------------------------- 图8/9

def gini(values):
    """基尼系数，0 = 完全均匀，1 = 全部集中在一只上。输入必须非负。

    用排序后的闭式解，而不是两两求差的双重循环：
        G = 2 * Σ(i * x_i) / (n * Σx) - (n + 1) / n     （x 已升序，i 从 1 起）
    """
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or (x < 0).any() or x.sum() <= 0:
        return np.nan
    n = x.size
    index = np.arange(1, n + 1)
    return 2.0 * (index * x).sum() / (n * x.sum()) - (n + 1) / n


def holding_returns(selection, close, last_session=None):
    """每个持有期里，各只持仓从期初到期末的收益率（宽表，行 = 期初调仓日）。

    期末价按「截至期末最后一个有效价」取：期间退市或停牌的股票不会因为末日
    没有报价就整段作废，与净值引擎「按最后有效价冻结」的处理保持一致。
    """
    bounds = list(selection.index)
    if last_session is not None and pd.Timestamp(last_session) > bounds[-1]:
        bounds.append(pd.Timestamp(last_session))
    rows = {}
    for start, end in zip(bounds, bounds[1:]):
        held = selection.loc[start]
        held = held[held.astype(bool)].index
        window = close.loc[start:end, close.columns.intersection(held)]
        if window.empty:
            continue
        rows[start] = window.ffill().iloc[-1] / window.iloc[0] - 1.0
    return pd.DataFrame(rows).T.reindex(columns=close.columns)


def contribution_gini(returns, min_names=10):
    """图8/9 —— 每期盈利股票之间、亏损股票之间收益贡献的不均匀程度。

    组合等权，所以每只股票的贡献就是收益率乘同一个权重；基尼系数与尺度无关，
    直接对收益率算即可。

    研报 4.3 节：「若每期盈利或亏损股票数量小于 10 只，则不计算当期的基尼
    系数」—— 只数太少时基尼系数抖得没法看。
    """
    def one(row, sign):
        picked = row.dropna()
        picked = picked[picked * sign > 0].abs()
        return gini(picked) if len(picked) >= min_names else np.nan

    return pd.DataFrame({
        "盈利": returns.apply(lambda r: one(r, +1), axis=1),
        "亏损": returns.apply(lambda r: one(r, -1), axis=1),
    })


# --------------------------------------------------------------------------- 图10

def cap_percentile(market_cap):
    """每日横截面市值分位数，(0, 1]，越大表示市值越大。"""
    return market_cap.rank(axis=1, pct=True)


def percentile_shift(members, percentile):
    """图10 —— 期初持仓的平均市值分位数，从期初到期末涨了多少。

    研报 4.4 节：期初、期末各算一次**全市场**分位数，取组合成员的均值再作差。

    期间退市的股票期末没有分位数。这里的处理是**两端取同一批股票**，而不是
    各自按现有的算：记成 0 等于说「它跌到了全市场最小」，把退市当成一次超额
    下跌；只在期末把它剔掉又会只减掉一个低值而期初还留着，把差值抬高。研报
    没有交代这个细节，成对取用是其中唯一不引入方向性偏差的做法。
    """
    dates = list(members.index)
    shifts = {}
    for start, end in zip(dates, dates[1:]):
        held = members.loc[start]
        held = held[held.astype(bool)].index
        if len(held) == 0 or start not in percentile.index or end not in percentile.index:
            continue
        pair = pd.DataFrame(
            {"began": percentile.loc[start].reindex(held),
             "ended": percentile.loc[end].reindex(held)}
        ).dropna()
        if pair.empty:
            continue
        shifts[end] = pair["ended"].mean() - pair["began"].mean()
    return pd.Series(shifts, dtype=float)


# --------------------------------------------------------------------------- 图11–14

def price_percentile_series(prices):
    """图11 —— 后复权价在**自身历史**中的时序分位数，逐日扩张窗口。

    研报 3.3 节：上市早于 2000-01-03 的从 2000-01-03 起算，否则从上市日起算。
    本地缓存的后复权价正好从 2000-01-04（2000 年第一个交易日）开始，每只股票
    的序列又天然始于它的上市日，所以「用它自己的全部历史」就已经满足这条口径。
    """
    p = prices.dropna()
    ranks = np.empty(len(p), dtype=float)
    # 逐日插入一个已排序数组，二分出「历史上不高于今天的天数」，O(n log n)。
    seen = []
    for i, value in enumerate(p.to_numpy()):
        bisect.insort(seen, value)
        ranks[i] = bisect.bisect_right(seen, value) / (i + 1)
    return pd.Series(ranks, index=p.index)


def event_percentiles(events, close):
    """图12–14 —— 每次建仓 / 卖出发生时，该股的时序股价分位数。

    只在需要的那几千个 (股票, 日期) 上算，不铺满整张 5,500 × 6,445 的价格表。
    """
    out = pd.Series(np.nan, index=events.index, dtype=float)
    for sid, group in events.groupby("order_book_id"):
        if sid not in close.columns:
            continue
        history = close[sid].dropna()
        if history.empty:
            continue
        for i, date in group["date"].items():
            past = history.loc[:date].to_numpy()
            if past.size == 0:
                continue
            out.at[i] = float((past <= past[-1]).mean())
    return events.assign(percentile=out)


# --------------------------------------------------------------------------- 图15–18

def industry_weights(ids, caps, industry):
    """图15–18 —— 按 `caps` 加权的行业占比（只数占比会被小盘拉平，没意义）。

    口径由调用方给定，函数本身与口径无关。研报注写「各行业总市值占指数总市值
    之比」，但那句注写错了口径：图16 实测研报用的是**自由流通发布权重**、分类用
    citics_2019（见 grill.md「两处对不齐」#2），所以 `03_analytics` 传的是
    `free_float_cap` 而非 `market_cap`。
    """
    frame = pd.DataFrame({"cap": pd.Series(caps).reindex(ids)}).dropna()
    frame["industry"] = pd.Series(industry).reindex(frame.index)
    frame = frame.dropna(subset=["industry"])
    if frame.empty:
        return pd.Series(dtype=float)
    weights = frame.groupby("industry")["cap"].sum()
    return (weights / weights.sum()).sort_values(ascending=False)


# --------------------------------------------------------------------------- 图19

def aggregate_pb(members, market_cap, pb):
    """图19 —— 整体法 PB = Σ总市值 / Σ净资产，净资产由 总市值 / PB 反推。

    整体法而不是个股 PB 取平均：后者会被几只极小净资产的股票拉爆。只用两个
    字段都齐的股票，保证分子分母是同一批成分算出来的。
    """
    cap = _aligned(market_cap, members)
    ratio = _aligned(pb, members)
    equity = cap / ratio.where(ratio > 0)
    return cap.where(equity.notna()).sum(axis=1) / equity.sum(axis=1)


# --------------------------------------------------------------------------- 图20–23

def median_cap(members, market_cap):
    """图20–23 —— 成分股总市值的中位数。"""
    return _aligned(market_cap, members).median(axis=1)


def trailing_mean(series, months=12):
    """研报几张图上那条「TTM 移动平均」。按调仓期数而不是自然月计。"""
    return series.rolling(months, min_periods=months).mean()


# --------------------------------------------------------------------------- 图24

def aggregate_turnover_by_cap(members, total_turnover, market_cap):
    """图24 —— 整体法日均换手率 = Σ成交额 / Σ市值。分母用哪种市值口径由调用方给定。

    对个股比值取平均会被几只极小市值的股票拉爆，所以先各自加总再相除（研报的
    「整体法」）；只在两个字段都齐的成员格子上求和，保证分子分母是同一批股票。
    """
    amount = _aligned(total_turnover, members)
    cap = _aligned(market_cap, members)
    return amount.where(cap.notna()).sum(axis=1) / cap.sum(axis=1)


def aggregate_turnover(members, total_turnover, turnover_rate):
    """流通A股口径的整体法换手率 —— 分母由换手率恒等式反推。

    数据里没有市值列，但换手率的定义直接给出分母：流通市值 = 成交额 / 换手率。
    **实测这个反推出来的分母就是流通A股市值**（重建流通市值 /(收盘×流通A股) = 0.997），
    而研报图24 的水平更像自由流通口径。所以正式图改用 `free_float_cap` 作分母
    （见 `03_analytics.figures_24_25` 与 grill.md「两处对不齐」），这个函数保留作对照。
    （`turnover_rate` 是百分数，实测：平安银行 2015-06-12 为 1.6125。）
    """
    circulating = total_turnover / (turnover_rate / 100.0).where(turnover_rate > 0)
    return aggregate_turnover_by_cap(members, total_turnover, circulating)


def free_float_cap(raw_close, free_circulation):
    """自由流通市值 = 不复权收盘价 × 自由流通股本。

    rqdatac 没有现成的自由流通市值因子（标定过 6 个候选：3 个是总市值、3 个是
    流通市值，无一是自由流通），只能由 `get_shares` 的 free_circulation 自己算。
    用不复权价，与 market_cap 的口径一致（市值 = 不复权价 × 股本）。
    """
    return raw_close * free_circulation


# --------------------------------------------------------------------------- 图25

def nominal_turnover(selection, size=None):
    """图25 —— 名义双边换手率 = (调入 + 调出只数) / 组合只数。

    与 `backtest.Result.turnover` 的 `sum|Δw|` 价值口径是两回事，研报自己前后
    用了这两种口径（见 grill.md）。被调出的正是涨上去的票，权重已漂移到平均
    之上，所以价值口径必然高于名义口径。
    """
    counts = selection.sum(axis=1) if size is None else pd.Series(size, index=selection.index)
    changes = membership_changes(selection)
    moved = changes.groupby("date").size().reindex(selection.index).fillna(0)
    return moved / counts.replace(0, np.nan)


# --------------------------------------------------------------------------- 图27

def rolling_correlation(nav, index_close, window=252):
    """图27 —— 组合与各指数日收益的滚动 12 个月相关系数。"""
    returns = nav.pct_change()
    index_returns = index_close.reindex(nav.index).pct_change()
    return pd.DataFrame(
        {col: returns.rolling(window).corr(index_returns[col]) for col in index_close.columns}
    )


# --------------------------------------------------------------------------- 图28

def calendar_effect(nav, phases):
    """图28 —— 各工作日的平均日收益，减去该阶段全部交易日的平均日收益。

    研报 4.9 节：「周一的日度平均收益的含义是从上一个交易日到周一收盘的日度
    收益率」—— 就是普通的日收益按**当天**是星期几归类，跨周末那一段算在周一。
    """
    returns = nav.pct_change().dropna()
    columns = {}
    for label, (start, end) in phases.items():
        segment = returns.loc[str(start):str(end)]
        if segment.empty:
            continue
        by_day = segment.groupby(segment.index.weekday).mean().reindex(range(5))
        columns[label] = (by_day - segment.mean()).to_numpy()
    return pd.DataFrame(columns, index=WEEKDAYS)
