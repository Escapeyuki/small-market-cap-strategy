"""持仓构造与日度净值引擎。

等权、月度调仓、期间自然漂移、按成交额计费。自写向量化 pandas 而不是用现成框架
（grill.md Q5）：研报的诊断图需要逐期持仓与逐票盈亏，这些东西从黑盒里拿不出来。

**唯一的成交口径：T 日收盘出信号，T+1 日开盘成交**（grill.md Q19，取代 Q4）。

研报用的是 T 日收盘排名 + 同一个 T 日收盘价成交——拿收盘价排完序，再假装能按
那个已经知道的价格成交，这是未来函数。本项目一度把它作为「研报口径」并行实现，
为的是验证 43.1%；实测两者月频只差 0.09pp（那次测量的结果记在 grill.md，不会
因为代码删掉而失效），而保留一条有偏的路径当默认值与项目自己的实盘意图标准
相抵触，所以现在只剩可执行的这一条。

代价要说清楚：**本引擎不再能复算研报自己的数**，跑出来的是「同样的策略，正确
执行」的结果，不是研报那个数。

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


def run(selection, post_close, post_open, sessions, cost_per_side=COST_PER_SIDE):
    """跑一遍净值。

    `selection` 是布尔宽表（行 = **信号日**），来自 universe.smallest / deciles。
    只有真正被选中过的股票参与计算，其余列直接丢掉。
    """
    sessions = pd.DatetimeIndex(sessions)
    ids = selection.columns[selection.any(axis=0)]
    close = post_close.reindex(index=sessions, columns=ids)
    open_ = post_open.reindex(index=sessions, columns=ids)
    overnight, intraday, close_to_close = _legs(close, open_)

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

    def rebalance(signal_date, trade_date, weights, value):
        target = targets.loc[signal_date].to_numpy()
        traded = np.abs(target - weights).sum()        # 双边换手率
        charge = cost_per_side * traded
        recorded["weights"][trade_date] = target
        recorded["drifted"][trade_date] = weights.copy()
        recorded["turnover"][trade_date] = traded
        recorded["cost"][trade_date] = charge
        return target, value * (1.0 - charge)

    for t in range(len(sessions)):
        # 成交日最早也是第 1 个交易日（信号日在它前一天），所以 t == 0 必然空仓，
        # nav[0] 自然就是期初资金 1.0，不需要特判。
        if t in trades:
            value *= 1.0 + float(weights @ overnight[t])
            weights = _drift(weights, overnight[t])
            weights, value = rebalance(trades[t], sessions[t], weights, value)
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
