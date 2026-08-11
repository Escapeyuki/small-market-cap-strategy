"""持仓构造与日度净值引擎。

等权、月度调仓、期间自然漂移、按成交额计费。自写向量化 pandas 而不是用现成框架
（grill.md Q5）：研报的诊断图需要逐期持仓与逐票盈亏，这些东西从黑盒里拿不出来。

**两种成交口径并行**（grill.md Q4）：

  * `report_close` —— T 日收盘排名 + T 日**当日收盘**成交。这是研报口径，属轻度
    未来函数：用收盘价排名却又按同一个收盘价成交。对小市值策略尤其放大收益。
  * `next_open`   —— T 日收盘排名 + T+1 日**开盘**成交。这是能实盘执行的口径。

两者都跑，差多少就报多少。只做正确口径就无法验证 43.1% 是怎么来的。

净值起点约定：`nav[0] = 1.0` 是**建仓前**的期初资金，首次建仓的手续费体现在此后
的净值里。否则 `metrics.performance` 按 `nav.iloc[0]` 归一时会把首笔费用抹掉。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG

COST_PER_SIDE = CFG["backtest"]["cost_per_side"]


@dataclass
class Result:
    """一次回测的全部产物。诊断图要的东西都在这里，不必重跑。"""

    nav: pd.Series                  # 日度净值，起点 1.0
    weights: pd.DataFrame           # 调仓后的目标权重，行 = 调仓日
    drifted: pd.DataFrame           # 调仓前漂移到的权重，行 = 调仓日
    turnover: pd.Series             # 双边换手率 = sum|Δw|，对应图25
    cost: pd.Series                 # 每次调仓扣掉的费用（占净值比例）


def rebalance_dates(calendar, freq, start, end):
    """调仓日。

    月频取**每月第一个交易日**。这是先验取值，理由：研报主区间起点 2012-12-03
    恰好是 2012 年 12 月的第一个交易日，而 2022-03-31 只是数据截止。
    存疑之处记录在案——表2 容量测试的起点 2019-01-07 并不是 2019 年 1 月的第一
    个交易日（那天是 01-02），却同样是当月第一个**周一**，2012-12-03 也是。
    两种规则都解释得了主区间起点，只有周一规则同时解释得了容量测试起点。
    先按月初第一个交易日跑，再作单变量敏感性测试（grill.md Q14）。
    """
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if freq == "daily":
        return sessions
    period = {"monthly": "M", "weekly": "W"}[freq]
    grouped = pd.Series(sessions, index=sessions.to_period(period))
    return pd.DatetimeIndex(grouped.groupby(level=0).first()).sort_values()


def _legs(close, open_):
    """把逐日收益拆成隔夜腿与盘中腿，供 next_open 口径使用。

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


def run(selection, post_close, post_open, sessions,
        mode="report_close", cost_per_side=COST_PER_SIDE):
    """跑一遍净值。

    `selection` 是布尔宽表（行 = 调仓日），来自 universe.smallest / deciles。
    只有真正被选中过的股票参与计算，其余列直接丢掉。
    """
    sessions = pd.DatetimeIndex(sessions)
    ids = selection.columns[selection.any(axis=0)]
    close = post_close.reindex(index=sessions, columns=ids)
    open_ = post_open.reindex(index=sessions, columns=ids)
    overnight, intraday, close_to_close = _legs(close, open_)

    counts = selection[ids].sum(axis=1)
    targets = selection[ids].div(counts.where(counts > 0), axis=0).fillna(0.0)

    # 排名日 -> 成交日。next_open 下最后一个排名日若没有下一个交易日，只能作废。
    position = {d: i for i, d in enumerate(sessions)}
    trades = {}
    for signal_date in selection.index:
        if counts.loc[signal_date] == 0:
            continue
        i = position[signal_date] + (1 if mode == "next_open" else 0)
        if i < len(sessions):
            trades[i] = signal_date

    n_ids = len(ids)
    weights = np.zeros(n_ids)
    value = 1.0
    nav = np.empty(len(sessions))
    recorded = {"weights": {}, "drifted": {}, "turnover": {}, "cost": {}}

    def rebalance(signal_date, weights, value):
        target = targets.loc[signal_date].to_numpy()
        traded = np.abs(target - weights).sum()        # 双边换手率
        charge = cost_per_side * traded
        recorded["weights"][signal_date] = target
        recorded["drifted"][signal_date] = weights.copy()
        recorded["turnover"][signal_date] = traded
        recorded["cost"][signal_date] = charge
        return target, value * (1.0 - charge)

    for t in range(len(sessions)):
        if t > 0:
            if mode == "next_open" and t in trades:
                value *= 1.0 + float(weights @ overnight[t])
                weights = _drift(weights, overnight[t])
                weights, value = rebalance(trades[t], weights, value)
                value *= 1.0 + float(weights @ intraday[t])
                weights = _drift(weights, intraday[t])
            else:
                value *= 1.0 + float(weights @ close_to_close[t])
                weights = _drift(weights, close_to_close[t])

        if mode == "report_close" and t in trades:
            weights, value = rebalance(trades[t], weights, value)

        # nav[0] 记期初资金 1.0；首次建仓的费用留给下一日体现（见模块说明）。
        nav[t] = 1.0 if t == 0 else value

    order = list(recorded["weights"])
    return Result(
        nav=pd.Series(nav, index=sessions),
        weights=pd.DataFrame(recorded["weights"], index=ids).T.reindex(order),
        drifted=pd.DataFrame(recorded["drifted"], index=ids).T.reindex(order),
        turnover=pd.Series(recorded["turnover"]).reindex(order),
        cost=pd.Series(recorded["cost"]).reindex(order),
    )
