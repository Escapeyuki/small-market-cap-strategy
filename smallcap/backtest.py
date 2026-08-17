"""持仓构造与日度净值引擎。

等权、月度调仓、期间自然漂移、按成交额计费。自写向量化 pandas 而不是用现成框架
（grill.md Q5）：研报的诊断图需要逐期持仓与逐票盈亏，这些东西从黑盒里拿不出来。

**唯一的成交口径：T 日收盘出信号，T+1 日开盘成交**（grill.md Q19，取代 Q4）。

**研报从头到尾没有交代它在什么价上成交**（已逐页查证）。它只规定了选股范围、
调仓频率与费率。本项目原先把「T 日收盘排名 + 同一个收盘价成交」作为对它的
重建并行实现——那样拿收盘价排完序再按那个已经知道的价格成交，是未来函数。
删掉它的理由与研报用没用过它无关：一条有偏的路径不该当默认值。

代价要说清楚：**本引擎跑出来的不是研报那个数**，是「同样的策略，正确执行」的
结果。两者月频只差 0.09pp（那次测量记在 grill.md，不因代码删掉而失效），日频
差 13.87pp 且方向相反。

净值起点约定：`nav[0] = 1.0` 是建仓前的期初资金。第一个信号最早也要到第二个
交易日的开盘才成交，所以首日必然是空仓持币，这不是特例而是口径的必然结果。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG

COST_PER_SIDE = CFG["backtest"]["cost_per_side"]


@dataclass
class Result:
    """一次回测的全部产物。诊断图要的东西都在这里，不必重跑。

    除 `nav` 外各表都按**成交日**索引，不是信号日——「2013-01-04 的换手率」
    指的就是那天真的换了多少手。要按信号日对齐时用 `trade_dates()` 反查。
    """

    nav: pd.Series                  # 日度净值，起点 1.0
    weights: pd.DataFrame           # 调仓后的目标权重，行 = 成交日
    drifted: pd.DataFrame           # 调仓前漂移到的权重，行 = 成交日
    turnover: pd.Series             # 双边换手率 = sum|Δw|，对应图25
    cost: pd.Series                 # 每次调仓扣掉的费用（占净值比例）


def rebalance_dates(calendar, freq, start, end):
    """信号日。

    月频取**每月第一个交易日**。这是先验取值，理由：研报主区间起点 2012-12-03
    恰好是 2012 年 12 月的第一个交易日，而 2022-03-31 只是数据截止。
    存疑之处记录在案——表2 容量测试的起点 2019-01-07 并不是 2019 年 1 月的第一
    个交易日（那天是 01-02），却同样是当月第一个**周一**，2012-12-03 也是。
    两种规则都解释得了主区间起点，只有周一规则同时解释得了容量测试起点。
    先按月初第一个交易日跑，再作单变量敏感性测试（grill.md Q14）。

    注意这些是**出信号**的日子，不是成交的日子；成交在下一个交易日的开盘。
    """
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if freq == "daily":
        return sessions
    period = {"monthly": "M", "weekly": "W"}[freq]
    grouped = pd.Series(sessions, index=sessions.to_period(period))
    return pd.DatetimeIndex(grouped.groupby(level=0).first()).sort_values()


def trade_dates(signal_dates, sessions):
    """信号日 → 成交日，返回一个 Series（索引 = 信号日，值 = 成交日）。

    T 日收盘出信号，T+1 日开盘成交。最后一个信号日若后面已经没有交易日，该信号
    **作废**——比假装它能在当天成交诚实。

    分析层也要用它：引擎按成交日持仓，统计若还按信号日取持仓，算的就是一个引擎
    从未持有过的组合。
    """
    sessions = pd.DatetimeIndex(sessions)
    position = {d: i for i, d in enumerate(sessions)}
    pairs = [(d, sessions[position[d] + 1])
             for d in pd.DatetimeIndex(signal_dates)
             if d in position and position[d] + 1 < len(sessions)]
    return pd.Series([trade for _, trade in pairs],
                     index=pd.DatetimeIndex([signal for signal, _ in pairs]),
                     dtype="datetime64[ns]")


def _legs(close, open_):
    """把逐日收益拆成隔夜腿与盘中腿。

    成交日当天，开盘前还是旧持仓、开盘后才是新持仓，所以那一天的收益必须拆成
    两段分别计。非成交日用整段的收盘到收盘。

    缺开盘价时（2007 年前的回补数据、停牌日）把整段收益记在盘中腿上，
    保持 (1+隔夜)(1+盘中) == (1+收盘到收盘) 恒等，不凭空造出跳空。

    停牌日后复权收盘价沿用前值 → 当日收益 0，正是「持有但动不了」的效果。
    退市后价格变 NaN → 收益按 0 处理，等价于按最后一个有效价清仓后持币，
    直到下一个调仓日再分配给其余持仓。
    """
    prev = close.shift(1)
    usable = open_.notna() & (open_ > 0) & prev.notna()
    at_open = open_.where(usable, prev)
    overnight = (at_open / prev - 1).fillna(0.0).to_numpy()
    intraday = (close / at_open - 1).fillna(0.0).to_numpy()
    close_to_close = (close / prev - 1).fillna(0.0).to_numpy()
    return overnight, intraday, close_to_close


def _drift(weights, returns):
    grown = weights * (1.0 + returns)
    total = grown.sum()
    return grown / total if total > 0 else grown


def run(selection, post_close, post_open, sessions, cost_per_side=COST_PER_SIDE,
        suspended=None):
    """跑一遍净值。

    `selection` 是布尔宽表（行 = **信号日**），来自 universe.smallest / deciles。
    只有真正被选中过的股票参与计算，其余列直接丢掉。

    `suspended`（可选，日期 × order_book_id 布尔宽表，取自 volume == 0）一旦给出，
    引擎就**不在停牌股上成交**：成交日把「当天停牌」的仓位权重原样冻结——持仓的
    停牌股卖不掉、继续持有，刚选中却停牌的票也买不进——只把剩下的资金
    (1 − 冻结权重) 按目标铺到能交易的票上。停牌持仓因此被一路持有到复牌，复牌当天
    的跳空由 `_legs` 如实计入，而不是按停牌前的冻结价"卖掉"。**不给 `suspended`
    就是旧口径**（停牌持仓按冻结价被调出，方向上高估收益）——这是一个单变量开关，
    两种口径的差见 grill.md「停牌持仓的处理」。
    """
    sessions = pd.DatetimeIndex(sessions)
    ids = selection.columns[selection.any(axis=0)]
    close = post_close.reindex(index=sessions, columns=ids)
    open_ = post_open.reindex(index=sessions, columns=ids)
    overnight, intraday, close_to_close = _legs(close, open_)
    halt = (suspended.reindex(index=sessions, columns=ids).fillna(False).astype(bool).to_numpy()
            if suspended is not None else None)

    counts = selection[ids].sum(axis=1)
    targets = selection[ids].div(counts.where(counts > 0), axis=0).fillna(0.0)

    position = {d: i for i, d in enumerate(sessions)}
    schedule = trade_dates(selection.index, sessions)
    # 成交日 -> 信号日。空仓的信号（当期一只都选不出来）直接跳过。
    trades = {position[trade]: signal for signal, trade in schedule.items()
              if counts.loc[signal] > 0}

    weights = np.zeros(len(ids))
    value = 1.0
    nav = np.empty(len(sessions))
    recorded = {"weights": {}, "drifted": {}, "turnover": {}, "cost": {}}

    def rebalance(signal_date, trade_date, weights, value, t):
        target = targets.loc[signal_date].to_numpy()
        if halt is not None and halt[t].any():
            # 当天停牌的票买不进也卖不出：停牌位（持仓的 + 刚选中却停牌的空仓位）的
            # 权重原样冻结，剩下的资金 (1 − 冻结权重) 才按目标铺到能交易的票上。停牌
            # 持仓由此被持有到复牌，复牌的跳空落在 _legs 里，不在这里按冻结价卖出。
            frozen = weights * halt[t]
            buyable = target * ~halt[t]
            scale = buyable.sum()
            final = frozen + (buyable / scale * (1.0 - frozen.sum()) if scale > 0 else 0.0)
        else:
            final = target
        traded = np.abs(final - weights).sum()         # 双边换手率，只算真的动了的部分
        charge = cost_per_side * traded
        recorded["weights"][trade_date] = final
        recorded["drifted"][trade_date] = weights.copy()
        recorded["turnover"][trade_date] = traded
        recorded["cost"][trade_date] = charge
        return final, value * (1.0 - charge)

    for t in range(len(sessions)):
        # 成交日最早也是第 1 个交易日（信号日在它前一天），所以 t == 0 必然空仓，
        # nav[0] 自然就是期初资金 1.0，不需要特判。
        if t in trades:
            value *= 1.0 + float(weights @ overnight[t])
            weights = _drift(weights, overnight[t])
            weights, value = rebalance(trades[t], sessions[t], weights, value, t)
            value *= 1.0 + float(weights @ intraday[t])
            weights = _drift(weights, intraday[t])
        elif t > 0:
            value *= 1.0 + float(weights @ close_to_close[t])
            weights = _drift(weights, close_to_close[t])
        nav[t] = value

    order = list(recorded["weights"])
    return Result(
        nav=pd.Series(nav, index=sessions),
        weights=pd.DataFrame(recorded["weights"], index=ids).T.reindex(order),
        drifted=pd.DataFrame(recorded["drifted"], index=ids).T.reindex(order),
        turnover=pd.Series(recorded["turnover"]).reindex(order),
        cost=pd.Series(recorded["cost"]).reindex(order),
    )


def run_with_stops(selection, post_close, post_open, sessions,
                   take_profit, stop_loss, cost_per_side=COST_PER_SIDE):
    """在月度小市值组合上叠加止盈 / 止损（研报 4.12 节、图31）。

    研报 4.12（PDF 第 20-21 页 / 研报页码 21-22，4.12 节）原文：「止盈和止损的
    比率分布设置为 13%和 26%……在每个交易日卖出所有达到止盈或止损条件的股票，
    并将获得的资金平均应用于剩下的股票中，以保持始终满仓位。」`take_profit` /
    `stop_loss` 是正的比率阈值；传 `float("inf")` 关掉对应的那一条，两条都传 inf
    时本函数的净值逐点等于 `run()`（tests/test_backtest 有断言）。

    **研报没交代的三处口径，按 grill.md「止盈止损」的推断实现，均标注在此：**

    1. **参照价 = 本持仓段的建仓成交价（后复权），月内因再分配加仓不重置。** 研报
       只说「达到止盈/止损条件」，没说相对什么价。取相对建仓价的累计涨跌，而非
       相对持仓期最高点的回撤——依据是研报自己那句「26%的止损比率较高，很难被
       触发」：相对建仓价跌 26% 才谈得上「很难触发」，相对最高点回撤 26% 在小市值
       里极易触发，与原文矛盾。研报按**股票**判定（「卖出所有达到条件的股票」），
       所以参照价按股票记、不按每笔资金记，加仓不重置。

    2. **成交口径 = T 日收盘判定、T+1 开盘成交**，与主引擎同口径（grill.md Q19）。
       用 T 日收盘价既判定触发又按它成交是未来函数，与 Q1「无隐式未来函数」抵触。
       代价是本函数跑出来的图31 与研报之间同样含一项口径差，不是纯复现误差。

    3. **「平均应用于剩下的股票」= 卖出所得对每只剩余持仓等额加仓**（average 取
       等额，不是按现有权重比例）。

    返回 `(nav, events)`：`nav` 是日度净值；`events` 是按止盈止损**动作日**索引的
    DataFrame，列 `tp` / `sl`（当次触发的股票只数）、`turnover`（当次双边换手），
    用来量化研报「止盈多数时段有效、止损很难触发」这句定性结论。
    """
    sessions = pd.DatetimeIndex(sessions)
    ids = selection.columns[selection.any(axis=0)]
    close = post_close.reindex(index=sessions, columns=ids)
    open_ = post_open.reindex(index=sessions, columns=ids)
    overnight, intraday, close_to_close = _legs(close, open_)

    # 建仓参照价取成交日的实际成交价（有开盘价用开盘价，缺则沿用前收——与 _legs
    # 内部的 at_open 同一口径），触发判定用后复权收盘价。两者必须和净值引擎一致，
    # 否则「净值涨了多少」与「判定涨了多少」会对不上。
    prev = close.shift(1)
    at_open = open_.where(open_.notna() & (open_ > 0) & prev.notna(), prev).to_numpy()
    close_px = close.to_numpy()

    counts = selection[ids].sum(axis=1)
    targets = selection[ids].div(counts.where(counts > 0), axis=0).fillna(0.0)

    position = {d: i for i, d in enumerate(sessions)}
    schedule = trade_dates(selection.index, sessions)
    trades = {position[trade]: signal for signal, trade in schedule.items()
              if counts.loc[signal] > 0}

    weights = np.zeros(len(ids))
    basis = np.full(len(ids), np.nan)          # 每只持仓的建仓参照价（后复权），空仓位 NaN
    value = 1.0
    nav = np.empty(len(sessions))
    pending = None                             # 下一开盘要执行的止盈止损：(sell, tp_n, sl_n)
    events = {}

    for t in range(len(sessions)):
        rebalancing = t in trades
        stopping = pending is not None and not rebalancing   # 月度调仓当天让位给整体换仓
        if rebalancing or stopping:
            value *= 1.0 + float(weights @ overnight[t])      # 隔夜腿走旧权重
            weights = _drift(weights, overnight[t])
            if rebalancing:
                target = targets.loc[trades[t]].to_numpy()
                traded = np.abs(target - weights).sum()
                weights = target.copy()
                basis = np.where(target > 0, at_open[t], np.nan)   # 新建仓参照价 = 成交价
            else:
                sell, tp_n, sl_n = pending
                remaining = (weights > 0) & ~sell
                freed = float(weights[sell].sum())
                new_weights = weights.copy()
                new_weights[sell] = 0.0
                new_weights[remaining] += freed / int(remaining.sum())   # 均分给剩余持仓
                traded = np.abs(new_weights - weights).sum()
                weights = new_weights
                basis[sell] = np.nan                            # 剩余持仓参照价不重置
                events[sessions[t]] = (tp_n, sl_n, float(traded))
            value *= 1.0 - cost_per_side * traded
            value *= 1.0 + float(weights @ intraday[t])         # 盘中腿走新权重
            weights = _drift(weights, intraday[t])
            pending = None
        elif t > 0:
            value *= 1.0 + float(weights @ close_to_close[t])
            weights = _drift(weights, close_to_close[t])
        nav[t] = value

        # 收盘：这天的收盘价一出就判定下一开盘要不要止盈止损。下一开盘若本就是月度
        # 成交日（t+1 in trades），整体换仓优先，不必再挂。
        if (t + 1) not in trades and weights.any():
            with np.errstate(invalid="ignore"):
                gain = close_px[t] / basis - 1.0
            held = weights > 0
            tp_hit = held & (gain >= take_profit)
            sl_hit = held & (gain <= -stop_loss)
            sell = tp_hit | sl_hit
            # 「平均应用于剩下的股票」要求至少留一只可分配。全部触发是无定义的极端
            # 边界（实测从不发生），此时不动，等下一个月度调仓。
            if sell.any() and (held & ~sell).any():
                pending = (sell, int(tp_hit.sum()), int(sl_hit.sum()))

    return (pd.Series(nav, index=sessions),
            pd.DataFrame(events, index=["tp", "sl", "turnover"]).T)
