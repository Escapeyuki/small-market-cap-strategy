"""可投资股票池 —— point-in-time，不含未来信息。

研报 3 节原文：「我们进行选股的范围是排除 ST 股、*ST 股、北交所股票、上市不满 20
日次新股票后的所有股票，此外，在每个调仓日，排除当天涨停、不交易的股票。」

这句话里其实是两级过滤，本模块也就分两级：

  * `STANDING` 常设股票池（ST、北交所、次新）—— 所有测试共用；
  * `REBALANCE_DAY` 调仓日附加（涨停、停牌）—— 只约束「买不买得进」。
    研报没说持仓股在调仓日涨停或停牌时能不能卖，本项目先验假设「能卖」，
    这是一条记录在案的假设，2015 年停牌潮下需要单独量化（见 grill.md）。

谓词是普通函数，签名 `(Panel) -> 布尔宽表`；加一个变体就是往列表里追加一个函数
（grill.md Q7）。没有 DSL、没有注册表、没有基类。

**北交所不需要实现**：rqdatac 的 CS 股票池里根本不存在北交所标的，该过滤自动
满足。这一点由 tests/test_universe.py 断言，而不是写一个永远不会触发的过滤器。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data
from .config import CFG

MIN_LISTED_DAYS = CFG["universe"]["min_listed_days"]


@dataclass(frozen=True)
class Panel:
    """在若干个选股日上对齐好的宽表：行 = 日期，列 = order_book_id。

    只在选股日上取值，不是逐日全量。研报的选股逻辑本来就只在调仓日发生，
    月频下这是 112 行而不是 2267 行，省下的内存正好留给净值引擎。
    """

    market_cap: pd.DataFrame
    raw_close: pd.DataFrame
    limit_up: pd.DataFrame
    volume: pd.DataFrame
    st: pd.DataFrame                # 布尔
    listed_days: pd.DataFrame       # 上市至今的交易日数
    registration: pd.DataFrame = None   # 注册制布尔（专题之二 E10）；专题之一不设，留 None

    @property
    def dates(self):
        return self.market_cap.index

    @property
    def ids(self):
        return self.market_cap.columns


# --------------------------------------------------------------------------- 谓词

def has_price(panel):
    """当天在市。未上市与已退市的行市值都是 NaN，一并挡在外面。"""
    return panel.market_cap.notna()


def not_st(panel):
    """ST / *ST。事件表里只有 True 行，缺席即非 ST。"""
    return ~panel.st


def seasoned(panel, min_days=MIN_LISTED_DAYS):
    """次新股。

    研报只写「上市不满 20 日」，没说自然日还是交易日；本项目取**交易日**
    （config.universe.min_listed_days），属先验取值，日后作单变量敏感性测试。
    """
    return panel.listed_days >= min_days


def not_limit_up(panel):
    """涨停：不复权收盘价触及当日涨停价。

    收益用后复权、涨停判定用不复权，两者绝不能混在一次 get_price 里取（见
    grill.md 实施中的发现）。

    `limit_up == 0` 表示「当日无涨跌幅参照」，含停牌与注册制新股上市初期两种
    情形，此时无从判定涨停——但这两种情形已分别被 `trading` 与 `seasoned`
    排除，所以这里按「未涨停」放行，而不是当成脏数据。
    """
    known = panel.limit_up > 0
    return ~(known & (panel.raw_close >= panel.limit_up - 1e-6))


def trading(panel):
    """停牌：研报的说法是「不交易的股票」，直接按当日无成交量判定。

    `volume == 0` 是一个不额外花额度的停牌标记，与 is_suspended 互为交叉校验
    （见 tests/test_universe.py）。
    """
    return panel.volume > 0


# --------------------------------------------------------------------------- 专题之二

def seasoned_1y(panel):
    """专题之二的次新口径：上市满 1 年（研报 p.6 §1.1「上市满 1 年」）。

    「1 年」研报未说自然日还是交易日；本项目先验取 243 交易日（A 股年均约 243 个
    交易日，见 grill.md Q16 与 grill_enhance.md E11）。
    """
    return seasoned(panel, CFG["report2"]["min_listed_days_1y"])


def not_registration(panel):
    """非注册制（研报 p.6 §1.1「非注册制、非北交所」）。

    研报未给判据，本项目按 board_type + 上市日期推断（grill_enhance.md E10）：
    科创板（KSH，全部 688）全部为注册制；创业板（GEM）中上市日 ≥ 2020-08-24 者
    为注册制 IPO。北交所本就不在 rqdatac CS 池，无需另判。`registration` 未设
    （专题之一口径）时按「全非注册制」放行。
    """
    if panel.registration is None:
        return pd.DataFrame(True, index=panel.dates, columns=panel.ids)
    return ~panel.registration


STANDING = [has_price, not_st, seasoned]
REBALANCE_DAY = [not_limit_up, trading]
BUYABLE = STANDING + REBALANCE_DAY

# 专题之二选股范围：上市满 1 年、非 ST、非注册制；调仓日附加非涨停、非停牌。
STANDING2 = [has_price, not_st, seasoned_1y, not_registration]
BUYABLE2 = STANDING2 + REBALANCE_DAY


def eligible(panel, predicates=BUYABLE):
    """把谓词列表求交集。"""
    mask = predicates[0](panel)
    for predicate in predicates[1:]:
        mask &= predicate(panel)
    return mask


# --------------------------------------------------------------------------- 选股

def size_rank(panel, mask=None):
    """合格股票按总市值升序排名，1 = 最小。不合格的位置是 NaN。"""
    mask = eligible(panel) if mask is None else mask
    return panel.market_cap.where(mask).rank(axis=1, method="first")


def smallest(panel, n, skip=0, predicates=BUYABLE):
    """每个选股日市值最小的第 `skip+1` 至 `skip+n` 只股票，返回布尔宽表。

    `skip` 就是研报 3.2 节的市值次低组合：小市值200 = 第 101-200 只，
    即 `smallest(panel, 100, skip=100)`。
    """
    rank = size_rank(panel, eligible(panel, predicates))
    return (rank > skip) & (rank <= skip + n)


def cascade(panel, steps, factors=None, predicates=BUYABLE2, mask=None):
    """逐级筛选选股（专题之二第 3 部分「逐级筛选」，研报 p.7-8 §2）。

    `steps` = [(因子名, ascending, keep_n), ...]，从合格池出发，每级按该因子在**上一级
    存活者**里排名、保留 keep_n 只，交给下一级。`ascending=True` 取因子值小的
    （研报表2 注「正向排序取因子值小的」），如市值/波动率/股价的「小/低」。

    因子名 `'cap'` 取 `panel.market_cap`；其余在 `factors` 字典里查（如 `'vol'` →
    波动率宽表，来自 smallcap.factors.volatility，须已对齐到 panel.dates × panel.ids）。

    低波 50 = `cascade(panel, [('cap', True, 100), ('vol', True, 50)], {'vol': vol})`。
    单级特例 `cascade(panel, [('cap', True, n)])` 与 `smallest(panel, n)` 等价。
    """
    factors = factors or {}
    base = eligible(panel, predicates)
    if mask is not None:                                     # 附加约束（如表24 分析师覆盖）
        base = base & mask.reindex(index=base.index, columns=base.columns).fillna(False)
    mask = base
    for name, ascending, keep in steps:
        frame = panel.market_cap if name == "cap" else factors[name]
        rank = frame.where(mask).rank(axis=1, ascending=ascending, method="first")
        mask = mask & rank.le(keep)          # NaN.le(keep) == False，历史不足者自然出局
    return mask


def deciles(panel, n_groups=10, predicates=BUYABLE):
    """按市值把合格股票等分成 n 组，返回 {组号: 布尔宽表}。

    研报 4.1 节：「组编号越大，市值越小」，所以第 1 组是市值最大的那一档。
    每期合格股票数不同，所以按分位数切而不是按固定只数切。
    """
    rank = size_rank(panel, eligible(panel, predicates))
    total = rank.notna().sum(axis=1)
    quantile = rank.div(total, axis=0)                       # (0, 1]，越小市值越小
    groups = {}
    for i in range(n_groups):
        lower, upper = i / n_groups, (i + 1) / n_groups
        smaller = (quantile > lower) & (quantile <= upper)
        groups[n_groups - i] = smaller                       # 组号越大市值越小
    return groups


# --------------------------------------------------------------------------- 组装

def trading_calendar():
    return pd.DatetimeIndex(data.load("calendar")["date"]).sort_values()


def listed_sessions(dates, instruments, calendar, ids=None):
    """每个 (选股日, 股票) 上市至今的交易日数，上市当日记 1。

    用交易日序号相减而不是自然日相减，见 `seasoned` 的口径说明。
    """
    if ids is not None:
        instruments = instruments[instruments["order_book_id"].isin(ids)]
    listed = pd.to_datetime(instruments["listed_date"], errors="coerce")
    first = np.searchsorted(calendar.values, listed.values, side="left")
    here = np.searchsorted(calendar.values, pd.DatetimeIndex(dates).values, side="left")
    counted = here[:, None] - first[None, :] + 1
    return pd.DataFrame(counted, index=dates, columns=instruments["order_book_id"].values)


def panel(dates):
    """从本地 Parquet 组装选股日上的 Panel。`dates` 必须是交易日。"""
    dates = pd.DatetimeIndex(dates)
    lo, hi = dates[0], dates[-1]

    market_cap = data.series("market_cap", "market_cap", lo, hi, dates).reindex(dates)
    raw = data.series_many("price_raw", ["close", "limit_up", "volume"], lo, hi, dates)
    ids = market_cap.columns

    def align(frame, fill=np.nan):
        return frame.reindex(index=dates, columns=ids).fillna(fill)

    st_long = data.load("st", lo, hi)
    st = st_long.pivot(index="date", columns="order_book_id", values="is_st")

    instruments = data.load("instruments")
    sessions = listed_sessions(dates, instruments, trading_calendar(), ids)

    # 注册制标记（专题之二 E10）：静态、逐股恒定，按 board_type + 上市日期判定后
    # 广播到每个选股日。科创板（KSH）全注册制；创业板（GEM）上市 ≥ cutoff 者为
    # 注册制 IPO。专题之一的 BUYABLE 不含 not_registration，这个字段对它无副作用。
    meta = instruments.drop_duplicates("order_book_id").set_index("order_book_id")
    board = meta["board_type"].reindex(ids)
    listed = pd.to_datetime(meta["listed_date"], errors="coerce").reindex(ids)
    cutoff = pd.Timestamp(CFG["report2"]["registration_cutoff"])
    is_reg = (board == "KSH") | ((board == "GEM") & (listed >= cutoff))
    registration = pd.DataFrame(
        np.repeat(is_reg.to_numpy()[None, :], len(dates), axis=0), index=dates, columns=ids
    )

    return Panel(
        market_cap=market_cap,
        raw_close=align(raw["close"]),
        limit_up=align(raw["limit_up"], 0.0),
        volume=align(raw["volume"], 0.0),
        # 事件表只有 True 行，缺席即非 ST；用 eq(True) 一步补齐并落到 bool。
        st=st.reindex(index=dates, columns=ids).eq(True),
        listed_days=align(sessions, 0).astype(int),
        registration=registration,
    )
