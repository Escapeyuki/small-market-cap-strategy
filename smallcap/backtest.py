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

    def firsts(period):
        g = pd.Series(sessions, index=sessions.to_period(period))
        return pd.DatetimeIndex(g.groupby(level=0).first()).sort_values()

    if freq in ("monthly", "weekly", "quarterly"):
        return firsts({"monthly": "M", "weekly": "W", "quarterly": "Q"}[freq])
    if freq == "biweekly":
        # 隔周取该周首个交易日，自区间首个调仓周起（专题之二 E12）。
        return firsts("W")[::2]
    if freq == "bimonthly":
        return firsts("M")[::2]
    if freq == "monthend":
        # 每月最后一个交易日，研报明示除 12 月与 3 月（p.16 §3.2.1，E12）。
        g = pd.Series(sessions, index=sessions.to_period("M"))
        lasts = pd.DatetimeIndex(g.groupby(level=0).last()).sort_values()
        return lasts[~lasts.month.isin([12, 3])]
    raise ValueError(f"unknown freq {freq!r}")


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
        suspended=None, cash_dates=None, limit_down=None):
    """跑一遍净值。

    `limit_down`（可选，日期 × id 布尔宽表，True = 当日**收盘跌停**）—— 专题之二的
    **跌停惩罚**（grill_enhance.md E7）。调仓时要清仓（target=0）的持仓若当天收盘
    跌停就卖不掉：冻结持有、标记为 pending，到**首个非收盘跌停日按当日收盘价卖出**，
    卖出资金**均分给其余持仓**（沿用 run_with_stops 均分约定）。只冻结 sell-side——
    刚选中却跌停的票照常买得进（跌停有卖盘）。不给 `limit_down` 就是旧口径（跌停股
    按调仓日冻结价被调出）。它与 `cash_dates` 可叠加（旗舰 = 择时 + 跌停惩罚）。

    `cash_dates`（可选，信号日集合）—— 专题之二的**择时空仓**（grill_enhance.md E9）：
    这些信号日在其成交日**强制清仓持币**（目标权重全 0，卖光付一次费），净值随后走平
    到下一个非空仓调仓日。它与「当期选不出票」在语义上不同——后者保持旧仓漂移、
    跳过调仓，前者主动清仓。若同时给了 `suspended`，停牌持仓卖不掉的部分仍按冻结
    逻辑保留，其余清成现金（target=0 时 halt 逻辑自然退化为「只留冻结位」）。

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
    ld = (limit_down.reindex(index=sessions, columns=ids).fillna(False).astype(bool).to_numpy()
          if limit_down is not None else None)

    counts = selection[ids].sum(axis=1)
    targets = selection[ids].div(counts.where(counts > 0), axis=0).fillna(0.0)

    position = {d: i for i, d in enumerate(sessions)}
    schedule = trade_dates(selection.index, sessions)
    # 成交日 -> 信号日。空仓的信号（当期一只都选不出来）直接跳过。
    trades = {position[trade]: signal for signal, trade in schedule.items()
              if counts.loc[signal] > 0}

    # 择时空仓（E9）：这些信号日的成交日强制清仓持币，覆盖并标记成 cash。
    cash_positions = set()
    if cash_dates is not None:
        for signal, trade in trade_dates(pd.DatetimeIndex(cash_dates), sessions).items():
            trades[position[trade]] = signal
            cash_positions.add(position[trade])

    weights = np.zeros(len(ids))
    pending = np.zeros(len(ids), dtype=bool)       # 跌停惩罚：待释放的冻结调出位（E7）
    value = 1.0
    nav = np.empty(len(sessions))
    recorded = {"weights": {}, "drifted": {}, "turnover": {}, "cost": {}}

    def rebalance(signal_date, trade_date, weights, value, t, to_cash=False):
        target = np.zeros(len(ids)) if to_cash else targets.loc[signal_date].to_numpy()
        # 冻结集合：停牌位（买卖两侧，halt）∪ 跌停调出位（仅 sell-side）。停牌的票买不进
        # 也卖不出；跌停只挡卖出——要清仓（target≈0）的持仓当天收盘跌停就卖不掉，冻结
        # 持有并标记 pending，等日内释放逻辑到首个非跌停日卖出（E7）。
        frozen_mask = np.zeros(len(ids), dtype=bool)
        if halt is not None:
            frozen_mask |= halt[t]
        if ld is not None:
            stuck = (weights > 1e-12) & (target <= 1e-12) & ld[t]
            frozen_mask |= stuck
            pending[stuck] = True
        if frozen_mask.any():
            frozen = weights * frozen_mask
            buyable = target * ~frozen_mask
            scale = buyable.sum()
            final = frozen + (buyable / scale * (1.0 - frozen.sum()) if scale > 0 else 0.0)
        else:
            final = target
        if ld is not None:
            pending[final <= 1e-12] = False            # 已成功卖出的清掉 pending 标记
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
            weights, value = rebalance(trades[t], sessions[t], weights, value, t,
                                       to_cash=t in cash_positions)
            value *= 1.0 + float(weights @ intraday[t])
            weights = _drift(weights, intraday[t])
        elif t > 0:
            value *= 1.0 + float(weights @ close_to_close[t])
            weights = _drift(weights, close_to_close[t])

        # 收盘释放跌停惩罚的冻结位（E7）：pending 中当天不再收盘跌停的，按今日收盘价
        # 卖出、均分给其余非 pending 持仓。卖出与买入各计一次单边费。
        if ld is not None and pending.any():
            release = pending & ~ld[t] & (weights > 1e-12)
            if release.any():
                recipients = (weights > 1e-12) & ~pending
                freed = float(weights[release].sum())
                traded = freed + (freed if recipients.any() else 0.0)
                value *= 1.0 - cost_per_side * traded
                weights = weights.copy()
                weights[release] = 0.0
                if recipients.any():
                    weights[recipients] += freed / int(recipients.sum())
                pending[release] = False
        nav[t] = value

    order = list(recorded["weights"])
    return Result(
        nav=pd.Series(nav, index=sessions),
        weights=pd.DataFrame(recorded["weights"], index=ids).T.reindex(order),
        drifted=pd.DataFrame(recorded["drifted"], index=ids).T.reindex(order),
        turnover=pd.Series(recorded["turnover"]).reindex(order),
        cost=pd.Series(recorded["cost"]).reindex(order),
    )


def run_with_capacity(selection, post_close, turnover_yuan, sessions, initial_capital,
                      max_participation=0.05, cost_per_side=COST_PER_SIDE, cash_dates=None):
    """容量测试（研报表22 / p.22 §3.4.2，grill_enhance.md E8）。

    研报原文：「每次最多成交该股票日成交额的 5%」。这里按**金额**记账、逐日**顺延**
    部分成交：每次调仓设定各票的目标金额（目标权重 × 当时组合总值），此后每个交易日
    每票最多成交 `max_participation × 当日成交额`（`turnover_yuan` = total_turnover，元），
    未成交的顺延到下一交易日，直至补齐或下次调仓。资金量越大、目标金额越超过小微盘
    的 5% 日成交额，越填不满 → 长期欠配持币 → 收益与波动同时下降（表22 的形态）。

    简化（均标注，E8）：①成交按**收盘价**逐日撮合、按收盘价 mark（非主引擎的 T+1 开盘，
    容量效应是多日的、对开/收微观结构不敏感）；②不含跌停惩罚（E7 实测仅 −0.4pp，
    相对容量效应可忽略）。`cash_dates` 支持择时空仓（清仓=目标金额全 0）。返回日度净值
    （起点 1.0，= 组合总值 / 初始资金）。
    """
    sessions = pd.DatetimeIndex(sessions)
    ids = selection.columns[selection.any(axis=0)]
    close = post_close.reindex(index=sessions, columns=ids)
    ret = close.pct_change().fillna(0.0).to_numpy()
    turn = turnover_yuan.reindex(index=sessions, columns=ids).fillna(0.0).to_numpy()

    counts = selection[ids].sum(axis=1)
    weights = selection[ids].div(counts.where(counts > 0), axis=0).fillna(0.0)

    position = {d: i for i, d in enumerate(sessions)}
    schedule = trade_dates(selection.index, sessions)
    trade_at = {position[trade]: signal for signal, trade in schedule.items()
                if counts.loc[signal] > 0}
    cash_positions = set()
    if cash_dates is not None:
        for signal, trade in trade_dates(pd.DatetimeIndex(cash_dates), sessions).items():
            trade_at[position[trade]] = signal
            cash_positions.add(position[trade])

    holdings = np.zeros(len(ids))              # 各票持仓市值（元）
    cash = float(initial_capital)
    w = np.zeros(len(ids))                     # 目标权重，调仓日更新、期间不变
    nav = np.empty(len(sessions))

    for t in range(len(sessions)):
        holdings *= 1.0 + ret[t]               # 逐日 mark to market
        total = holdings.sum() + cash
        if t in trade_at:
            w = np.zeros(len(ids)) if t in cash_positions else weights.loc[trade_at[t]].to_numpy()
        # 目标金额按**目标权重 × 当日总值**逐日重算——铺满后 desired≈0，价格漂移不再回撤；
        # 只在未铺满时继续买。大资金下 w×total 远超 5% 封顶 → 长期欠配（表22 的形态）。
        target_val = w * total
        cap = max_participation * turn[t]                       # 各票当日成交额上限（元）
        fill = np.clip(target_val - holdings, -cap, cap)        # +买 −卖，受 5% 封顶
        sells = -np.minimum(fill, 0.0)
        buys = np.maximum(fill, 0.0)
        budget = cash + sells.sum()                             # 卖出所得可用于买入
        if buys.sum() > budget and buys.sum() > 0:              # 现金约束：买不超过可用
            buys *= budget / buys.sum()
        traded = buys.sum() + sells.sum()
        holdings += buys - sells
        cash += sells.sum() - buys.sum() - cost_per_side * traded
        nav[t] = (holdings.sum() + cash) / initial_capital

    return pd.Series(nav, index=sessions)


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
