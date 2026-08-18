"""专题之二增强族的策略构造 —— 样本内 `06_enhance.py` 与样本外 `07_enhance_oos.py` 共用。

把「策略清单 ROSTER + 日历月择时 apply_timing + 逐策略选股/回测 run_strategy」放这里，
让两个驱动脚本走**同一条构造路径**：样本外与样本内除区间外逐字相同（grill.md Q14），
而不是两处各写一份、靠对数字来发现走岔。口径全部见 grill_enhance.md 的 E 系列。

- ROSTER：研报第 3 部分 15 个核心策略的构造参数 + 研报发布值（对照列）；
  容量测试（表22）与分析师覆盖（表24）另在 06 的 capacity_test / analyst_test。
- apply_timing：把择时改成日历月口径（E9），与调仓频率解耦。
- build_freq / run_strategy：按频率缓存面板、构造单个策略并跑净值，返回原始积木
  （不算指标——各驱动自行切片：样本内算全区间，样本外分样本内/样本外两段算）。
"""
import pandas as pd

from . import backtest as bt, factors as fa, universe as u


# 15 个核心策略。研报表26 汇总的发布值：[年化, 换手率, 胜率, 策略平均分位数, 基准平均分位数]。
# 出处逐条见 project_enhance.md 的策略清单；惩罚列见 grill_enhance.md E6/E7。
# 样本外驱动只用前 5 个字段（label/freq/steps/cash/penalty）——ref 列是样本内对研报的
# 对照，已越过研报样本的样本外块用不到它（E1：样本外面板无研报列）。
ROSTER = [
    # 标签,                频率,        逐级 steps,                  空仓月,   惩罚,  研报[ann,turn,win,sp,bp]
    ("小市值100基准",      "monthly",   [("cap", True, 100)],        (),       False, [26.7, 44.37, 73.83, 74.14, 56.32]),
    ("小市值50",           "monthly",   [("cap", True, 50)],         (),       False, [33.3, 51.00, 72.48, 73.00, 57.33]),
    ("小市值低波50",       "monthly",   [("cap", True, 100), ("vol", True, 50)], (), False, [32.8, 53.34, 71.14, 73.51, 57.33]),
    ("择时小市值100",      "monthly",   [("cap", True, 100)],        (1, 4),   False, [39.2, 62.92, 75.17, 75.11, 56.99]),
    ("择时小市值50",       "monthly",   [("cap", True, 50)],         (1, 4),   False, [42.1, 66.49, 73.83, 74.61, 57.33]),
    ("择时低波50",         "monthly",   [("cap", True, 100), ("vol", True, 50)], (1, 4), False, [39.5, 68.79, 72.48, 74.32, 57.67]),
    ("低波50月末调仓",     "monthend",  [("cap", True, 100), ("vol", True, 50)], (1, 4), False, [37.6, 68.47, 70.95, 73.48, 58.04]),
    ("跌停惩罚择时100",    "monthly",   [("cap", True, 100)],        (1, 4),   True,  [36.4, 62.95, 74.00, 74.76, 56.98]),
    ("单周+惩罚择时100",   "weekly",    [("cap", True, 100)],        (1, 4),   True,  [39.0, 27.23, 64.25, 69.98, 61.02]),
    ("双周+惩罚择时100",   "biweekly",  [("cap", True, 100)],        (1, 4),   True,  [44.3, 40.03, 68.83, 72.46, 59.90]),
    ("双月+惩罚择时100",   "bimonthly", [("cap", True, 100)],        (1, 4),   True,  [33.6, 54.41, 70.67, 74.21, 59.34]),
    ("季度+惩罚择时100",   "quarterly", [("cap", True, 100)],        (1, 4),   True,  [32.2, 49.04, 74.00, 74.91, 58.66]),
    ("★旗舰双周低波50",    "biweekly",  [("cap", True, 100), ("vol", True, 50)], (1, 4, 6), True, [50.9, 47.49, 63.69, 70.91, 60.50]),
    ("费用测试(千十)",     "biweekly",  [("cap", True, 100), ("vol", True, 50)], (1, 4, 6), True, [44.4, 47.49, 63.69, 70.91, 60.50]),
    ("保留六月",           "biweekly",  [("cap", True, 100), ("vol", True, 50)], (1, 4),   True,  [48.5, 44.26, 65.23, 71.54, 59.73]),
]


def apply_timing(sel, sessions, cash_months):
    """把择时改成**日历月**口径（E9）：Jan/Apr(/Jun) 整月持币，与调仓频率无关。

    简单地「在落入空仓月的调仓日清仓、持到下一次调仓」对粗频率是错的——季度频在
    1 月初清仓会一直空到 4 月。正解是：每个空仓月的首个交易日清仓，下一个非空仓月
    的首个交易日按**最近一次原生选股**（ffill）重新建仓；空仓月里的原生调仓日只更新
    选股、不交易。返回 (增广选股宽表, 清仓信号日)。
    """
    sessions = pd.DatetimeIndex(sessions)
    first = pd.Series(sessions).groupby(sessions.to_period("M")).first()
    first_days = pd.DatetimeIndex(first.values)
    is_cash = pd.Series([p.month in cash_months for p in first.index]).reset_index(drop=True)
    cash_out = first_days[is_cash.values]
    reentry = first_days[(~is_cash.values) & is_cash.shift(1, fill_value=False).values]
    native = pd.DatetimeIndex(sel.index)
    keep = native[~native.month.isin(cash_months)]
    aug_index = pd.DatetimeIndex(sorted(set(keep) | set(cash_out) | set(reentry)))
    grid = pd.DatetimeIndex(sorted(set(native) | set(aug_index)))
    full = sel.astype(float).reindex(grid).ffill()               # ffill 需历史，故先并到 grid
    aug = full.reindex(aug_index) > 0.5
    aug.loc[aug.index.isin(cash_out)] = False                    # 空仓月首日清仓
    return aug, cash_out


def build_freq(cache, cal, freq, start, end, full_close, vol_w):
    """(rebalances, panel, volatility) 按频率缓存，避免多个策略在同一频率上重复建面板。

    `cache` 由调用方持有（一个 dict）；键只用 freq，因为单个驱动内 (start, end) 恒定。
    """
    if freq not in cache:
        reb = bt.rebalance_dates(cal, freq, start, end)
        panel = u.panel(reb)
        vol = fa.volatility(full_close, reb, vol_w, ids=panel.ids)
        cache[freq] = (reb, panel, vol)
    return cache[freq]


def run_strategy(cache, cal, sessions, close, open_, full_close, is_ld,
                 freq, steps, cash_months, penalty, cost, vol_w, start, end):
    """构造一个增强策略并跑净值，返回原始积木（nav/res/sel/cash/成交日/仓位暴露）。

    不在此算年化/胜率/分位——那些留给各驱动：样本内 `06` 算全区间，样本外 `07` 分
    样本内切片与样本外块两段各算一遍。构造顺序与 grill_enhance.md 一致：
    cascade 逐级选股 → apply_timing 日历月择时(E9) → bt.run（跌停惩罚 E7 仅当 penalty）。
    **样本内/样本外唯一差别是传入的 start/end/sessions**（Q14）。
    """
    reb, panel, vol = build_freq(cache, cal, freq, start, end, full_close, vol_w)
    sel = u.cascade(panel, steps, {"vol": vol}, predicates=u.BUYABLE2)
    cash = None
    if cash_months:
        sel, cash = apply_timing(sel, sessions, cash_months)
    ld = is_ld if penalty else None
    res = bt.run(sel, close, open_, sessions, cost_per_side=cost, cash_dates=cash, limit_down=ld)
    sched = bt.trade_dates(sel.index, sessions)
    tds = pd.DatetimeIndex(sched.values)
    cash_td = pd.DatetimeIndex(sched.reindex(cash).dropna().values) if cash is not None else None
    exposure = res.weights.sum(axis=1).reindex(sessions).ffill().fillna(0.0)
    return dict(nav=res.nav, res=res, sel=sel, cash=cash, sched=sched,
                tds=tds, cash_td=cash_td, exposure=exposure)
