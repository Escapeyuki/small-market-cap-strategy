"""选股池的谓词逻辑，用合成数据逐条钉死。

这些用例不碰网络也不碰 data/（除了末尾两条显式标注的真实数据不变量），因为
谓词的正确与否是纯逻辑问题：一只当天涨停的股票该不该进组合，跟 2015 年发生过
什么无关。真实数据只用来验两件**关于数据本身**的断言。
"""
import numpy as np
import pandas as pd
import pytest

from smallcap import data, universe as u

DATES = pd.to_datetime(["2020-01-02", "2020-02-03"])
IDS = ["A.XSHE", "B.XSHE", "C.XSHE", "D.XSHE"]


def frame(values, dtype=float):
    return pd.DataFrame(values, index=DATES, columns=IDS, dtype=dtype)


def make_panel(**overrides):
    """一个默认全部合格的 Panel，测哪条谓词就只改哪一列。"""
    fields = dict(
        market_cap=frame([[1e8, 2e8, 3e8, 4e8], [1e8, 2e8, 3e8, 4e8]]),
        raw_close=frame([[10.0] * 4] * 2),
        limit_up=frame([[11.0] * 4] * 2),
        volume=frame([[1e6] * 4] * 2),
        st=frame([[False] * 4] * 2, dtype=bool),
        listed_days=frame([[999] * 4] * 2, dtype=int),
    )
    fields.update(overrides)
    return u.Panel(**fields)


# --------------------------------------------------------------------------- 谓词

def test_default_panel_is_entirely_eligible():
    assert u.eligible(make_panel()).all().all()


def test_delisted_and_unlisted_rows_have_no_market_cap():
    panel = make_panel(market_cap=frame([[1e8, np.nan, 3e8, 4e8], [1e8, 2e8, 3e8, 4e8]]))
    assert not u.eligible(panel).loc[DATES[0], "B.XSHE"]
    assert u.eligible(panel).loc[DATES[1], "B.XSHE"]


def test_st_stocks_are_excluded_on_the_days_they_are_flagged():
    panel = make_panel(st=frame([[False, True, False, False], [False] * 4], dtype=bool))
    assert not u.eligible(panel).loc[DATES[0], "B.XSHE"]
    assert u.eligible(panel).loc[DATES[1], "B.XSHE"]


def test_new_listings_are_excluded_until_their_twentieth_session():
    panel = make_panel(listed_days=frame([[19, 20, 21, 999], [999] * 4], dtype=int))
    eligible = u.eligible(panel).loc[DATES[0]]
    assert not eligible["A.XSHE"]                    # 上市不满 20 日
    assert eligible["B.XSHE"] and eligible["C.XSHE"]


def test_limit_up_stocks_are_excluded_on_the_rebalance_day():
    panel = make_panel(raw_close=frame([[10.0, 11.0, 10.9, 10.0], [10.0] * 4]))
    eligible = u.eligible(panel).loc[DATES[0]]
    assert not eligible["B.XSHE"]                    # 收盘正好在涨停价
    assert eligible["C.XSHE"]                        # 差一点，没封上


def test_limit_up_is_judged_to_the_cent_not_approximately():
    """涨停价本身按分四舍五入，判定必须允许浮点误差但不能宽到放过真涨停。"""
    panel = make_panel(
        raw_close=frame([[10.999999, 11.000001, 10.99, 10.0], [10.0] * 4]),
        limit_up=frame([[11.0] * 4] * 2),
    )
    eligible = u.eligible(panel).loc[DATES[0]]
    assert not eligible["A.XSHE"] and not eligible["B.XSHE"]
    assert eligible["C.XSHE"]


def test_halted_stocks_are_excluded_by_zero_volume():
    panel = make_panel(volume=frame([[1e6, 0.0, 1e6, 1e6], [1e6] * 4]))
    assert not u.eligible(panel).loc[DATES[0], "B.XSHE"]


def test_zero_limit_up_is_not_treated_as_a_limit_up():
    """limit_up == 0 是「无涨跌幅参照」，不是「涨停价为零」。

    这两种情形（停牌 / 注册制新股初期）已分别被 trading 与 seasoned 排除，
    所以 not_limit_up 必须放行，否则会重复排除并掩盖真正的原因。
    """
    panel = make_panel(limit_up=frame([[0.0, 11.0, 11.0, 11.0], [11.0] * 4]))
    assert u.not_limit_up(panel).loc[DATES[0], "A.XSHE"]


def test_standing_universe_ignores_rebalance_day_conditions():
    """涨停与停牌只影响能不能买，不影响一只票算不算在选股范围内。"""
    panel = make_panel(
        volume=frame([[0.0] * 4] * 2),
        raw_close=frame([[11.0] * 4] * 2),
    )
    assert u.eligible(panel, u.STANDING).all().all()
    assert not u.eligible(panel, u.BUYABLE).any().any()


# --------------------------------------------------------------------------- 选股

def test_smallest_picks_by_market_cap_ascending():
    picked = u.smallest(make_panel(), 2)
    assert list(picked.loc[DATES[0]][picked.loc[DATES[0]]].index) == ["A.XSHE", "B.XSHE"]


def test_skip_reproduces_the_next_size_band():
    """研报 3.2 节：小市值200 = 市值最小的第 101-200 只，这里缩比例验证。"""
    picked = u.smallest(make_panel(), 2, skip=2)
    assert list(picked.loc[DATES[0]][picked.loc[DATES[0]]].index) == ["C.XSHE", "D.XSHE"]


def test_ineligible_stocks_do_not_consume_a_slot():
    """最小的那只被剔除后，应当由第 3 小的补位，而不是只剩一只。"""
    panel = make_panel(st=frame([[True, False, False, False], [False] * 4], dtype=bool))
    picked = u.smallest(panel, 2).loc[DATES[0]]
    assert list(picked[picked].index) == ["B.XSHE", "C.XSHE"]


def test_decile_group_numbers_run_from_large_cap_to_small():
    """研报 4.1 节：「组编号越大，市值越小」。"""
    groups = u.deciles(make_panel(), n_groups=4)
    assert list(groups[4].loc[DATES[0]].pipe(lambda r: r[r].index)) == ["A.XSHE"]
    assert list(groups[1].loc[DATES[0]].pipe(lambda r: r[r].index)) == ["D.XSHE"]


def test_deciles_partition_the_eligible_set_exactly_once():
    stacked = sum(g.astype(int) for g in u.deciles(make_panel(), n_groups=4).values())
    assert (stacked == 1).all().all()


# --------------------------------------------------------------------------- 上市日

def test_listed_sessions_counts_trading_days_not_calendar_days():
    calendar = pd.bdate_range("2020-01-06", periods=10)          # 连续两周工作日
    instruments = pd.DataFrame(
        {"order_book_id": ["A.XSHE", "B.XSHE"],
         "listed_date": ["2020-01-06", "2020-01-13"]}
    )
    sessions = u.listed_sessions(calendar[[0, 6, 9]], instruments, calendar)

    assert sessions.loc[calendar[0], "A.XSHE"] == 1              # 上市当日记 1
    assert sessions.loc[calendar[6], "A.XSHE"] == 7
    assert sessions.loc[calendar[0], "B.XSHE"] <= 0              # 还没上市
    assert sessions.loc[calendar[9], "B.XSHE"] == 5              # 跨了个周末仍只数 5 个交易日


# --------------------------------------------------------------------------- 真实数据不变量

@pytest.fixture(scope="module")
def instruments():
    try:
        return data.load("instruments")
    except FileNotFoundError:
        pytest.skip("instruments cache absent; run scripts/01_fetch.py")


def test_beijing_exchange_listings_do_not_exist_so_the_filter_is_free(instruments):
    """研报要求剔除北交所，但 rqdatac 的 CS 股票池里根本没有北交所标的。

    与其写一个永远不触发的过滤器，不如在这里断言这个前提。哪天数据源开始收录
    北交所，这条会先红，而不是让组合里悄悄混进 8 字头股票。
    """
    assert set(instruments["exchange"]) == {"XSHE", "XSHG"}
    assert not instruments["order_book_id"].str.endswith(".BJSE").any()


def test_zero_volume_agrees_with_the_is_suspended_flag():
    """volume == 0 是免费的停牌标记；与 is_suspended 对得上才敢拿它当过滤条件。"""
    try:
        raw = data.series("price_raw", "volume", "2015-01-01", "2015-06-30")
        flags = data.load("suspended", "2015-01-01", "2015-06-30")
    except FileNotFoundError:
        pytest.skip("price cache absent; run scripts/01_fetch.py")

    silent = raw == 0
    flagged = (
        flags.pivot(index="date", columns="order_book_id", values="is_suspended")
        .reindex(index=silent.index, columns=silent.columns)
        .eq(True)
    )
    agreement = (silent == flagged).to_numpy().mean()
    assert agreement > 0.99, f"两个停牌口径只有 {agreement:.3%} 一致"
