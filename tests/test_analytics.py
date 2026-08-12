"""诊断统计，用合成数据逐条钉死。

与 test_backtest.py 同一条纪律：每个期望值都能在纸上算出来，不是把当前实现的
输出抄回来当基准。这些统计是用来解释「为什么有效」的，如果它们本身算错了，
错的结论会比错的年化数字更难被发现——年化对不上研报会立刻暴露，而一张形状
大致合理的概率密度图可以错很久都没人察觉。
"""
import numpy as np
import pandas as pd
import pytest

from smallcap import analytics as a, universe as u

DATES = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
IDS = ["A.XSHE", "B.XSHE", "C.XSHE", "D.XSHE"]


def frame(values, index=DATES, columns=IDS, dtype=float):
    return pd.DataFrame(values, index=index, columns=columns, dtype=dtype)


def picks(rows):
    """rows: [[持有的股票], ...]，每个调仓日一行。"""
    out = pd.DataFrame(False, index=DATES[:len(rows)], columns=IDS)
    for date, held in zip(out.index, rows):
        out.loc[date, held] = True
    return out


def make_panel(**overrides):
    """默认全部合格的 Panel，测哪条原因就只改哪一列（同 test_universe.py）。"""
    fields = dict(
        market_cap=frame([[1e8, 2e8, 3e8, 4e8]] * 3),
        raw_close=frame([[10.0] * 4] * 3),
        limit_up=frame([[11.0] * 4] * 3),
        volume=frame([[1e6] * 4] * 3),
        st=frame([[False] * 4] * 3, dtype=bool),
        listed_days=frame([[999] * 4] * 3, dtype=int),
    )
    fields.update(overrides)
    return u.Panel(**fields)


# --------------------------------------------------------------------------- 基尼

def test_gini_is_zero_when_every_stock_contributes_equally():
    assert a.gini([3.0, 3.0, 3.0, 3.0]) == pytest.approx(0.0)


def test_gini_hits_its_n_dependent_maximum_when_one_stock_contributes_everything():
    """n 只股票时基尼系数的上界是 (n-1)/n，不是 1。"""
    assert a.gini([0.0, 0.0, 0.0, 4.0]) == pytest.approx(0.75)


def test_gini_of_an_arithmetic_ladder_is_hand_computable():
    """x=[1,2,3,4]：2*Σ(i*x_i)/(n*Σx) - (n+1)/n = 60/40 - 1.25 = 0.25。"""
    assert a.gini([4.0, 1.0, 3.0, 2.0]) == pytest.approx(0.25)


def test_gini_refuses_negative_input_instead_of_returning_a_number():
    """亏损那一侧必须先取绝对值再送进来，否则结果没有意义。"""
    assert np.isnan(a.gini([-1.0, 2.0]))
    assert np.isnan(a.gini([]))
    assert np.isnan(a.gini([0.0, 0.0]))


# --------------------------------------------------------------------------- 建仓 / 卖出

def test_the_first_rebalance_counts_every_holding_as_an_entry():
    changes = a.membership_changes(picks([["A.XSHE", "B.XSHE"]]))
    assert list(changes["event"]) == [a.ENTRY, a.ENTRY]
    assert set(changes["order_book_id"]) == {"A.XSHE", "B.XSHE"}


def test_swapping_one_name_records_one_entry_and_one_exit():
    changes = a.membership_changes(picks([["A.XSHE", "B.XSHE"], ["A.XSHE", "C.XSHE"]]))
    second = changes[changes["date"] == DATES[1]]
    assert list(second[second["event"] == a.ENTRY]["order_book_id"]) == ["C.XSHE"]
    assert list(second[second["event"] == a.EXIT]["order_book_id"]) == ["B.XSHE"]


def test_holding_the_same_names_records_nothing_after_the_first_day():
    changes = a.membership_changes(picks([["A.XSHE"], ["A.XSHE"], ["A.XSHE"]]))
    assert set(changes["date"]) == {DATES[0]}


# --------------------------------------------------------------------------- 图26 调出归因

def selection_dropping_b():
    return picks([["A.XSHE", "B.XSHE"], ["A.XSHE", "C.XSHE"]])


def reason_for_b(panel):
    _, detail = a.exit_reasons(selection_dropping_b(), panel)
    row = detail[detail["order_book_id"] == "B.XSHE"]
    assert len(row) == 1
    return row["reason"].iloc[0]


def test_a_stock_that_merely_grew_out_of_the_bottom_100_is_attributed_to_size():
    assert reason_for_b(make_panel()) == "市值上涨"


def test_a_stock_flagged_st_on_the_rebalance_day_is_attributed_to_st():
    st = frame([[False] * 4, [False, True, False, False], [False] * 4], dtype=bool)
    assert reason_for_b(make_panel(st=st)) == "戴帽"


def test_a_stock_with_no_market_cap_left_is_attributed_to_delisting():
    caps = frame([[1e8, 2e8, 3e8, 4e8], [1e8, np.nan, 3e8, 4e8], [1e8, 2e8, 3e8, 4e8]])
    assert reason_for_b(make_panel(market_cap=caps)) == "退市"


def test_a_halted_stock_is_attributed_to_the_halt_not_to_size():
    volume = frame([[1e6] * 4, [1e6, 0.0, 1e6, 1e6], [1e6] * 4])
    assert reason_for_b(make_panel(volume=volume)) == "停牌"


def test_a_limit_up_stock_is_attributed_to_the_limit_not_to_size():
    close = frame([[10.0] * 4, [10.0, 11.0, 10.0, 10.0], [10.0] * 4])
    assert reason_for_b(make_panel(raw_close=close)) == "涨停"


def test_delisting_outranks_st_when_both_apply():
    """退市的股票几乎总是先戴帽。优先级搞反了会把 2 次退市全记成戴帽。"""
    caps = frame([[1e8, 2e8, 3e8, 4e8], [1e8, np.nan, 3e8, 4e8], [1e8, 2e8, 3e8, 4e8]])
    st = frame([[False] * 4, [False, True, False, False], [False] * 4], dtype=bool)
    assert reason_for_b(make_panel(market_cap=caps, st=st)) == "退市"


def test_st_outranks_a_halt_when_both_apply():
    """戴帽的股票当天常常停牌，研报把它算作戴帽。"""
    st = frame([[False] * 4, [False, True, False, False], [False] * 4], dtype=bool)
    volume = frame([[1e6] * 4, [1e6, 0.0, 1e6, 1e6], [1e6] * 4])
    assert reason_for_b(make_panel(st=st, volume=volume)) == "戴帽"


def test_every_exit_gets_exactly_one_reason():
    """分类直接复用把股票挤出去的谓词，所以必须是穷尽且互斥的。"""
    counts, detail = a.exit_reasons(selection_dropping_b(), make_panel())
    assert counts.sum().sum() == len(detail)
    assert set(counts.columns) == set(a.REASONS)


def test_collapsing_to_the_reports_three_reasons_conserves_the_total():
    counts, detail = a.exit_reasons(selection_dropping_b(), make_panel())
    assert a.collapse_reasons(counts).sum().sum() == len(detail)


# --------------------------------------------------------------------------- 持有期收益

def test_holding_returns_measure_each_name_from_entry_to_the_next_rebalance():
    close = frame([[10.0, 10.0, 10.0, 10.0],
                   [12.0, 20.0, 10.0, 10.0],
                   [12.0, 20.0, 10.0, 10.0]])
    returns = a.holding_returns(picks([["A.XSHE", "B.XSHE"], ["A.XSHE"]]), close)
    assert returns.loc[DATES[0], "A.XSHE"] == pytest.approx(0.2)
    assert returns.loc[DATES[0], "B.XSHE"] == pytest.approx(1.0)
    assert np.isnan(returns.loc[DATES[0], "C.XSHE"])          # 没持有


def test_a_name_that_stops_quoting_is_valued_at_its_last_print():
    """与净值引擎一致：退市后按最后一个有效价冻结，不是整段作废。"""
    close = frame([[10.0] * 4, [15.0, 10.0, 10.0, 10.0], [np.nan, 10.0, 10.0, 10.0]])
    returns = a.holding_returns(picks([["A.XSHE"], ["A.XSHE"]]), close,
                               last_session=DATES[-1])
    assert returns.loc[DATES[1], "A.XSHE"] == pytest.approx(0.0)   # 15 → 冻结在 15


def test_the_final_period_runs_to_the_last_session_not_to_the_last_rebalance():
    close = frame([[10.0] * 4, [10.0] * 4, [13.0, 10.0, 10.0, 10.0]])
    returns = a.holding_returns(picks([["A.XSHE"], ["A.XSHE"]]), close,
                               last_session=DATES[-1])
    assert list(returns.index) == list(DATES[:2])
    assert returns.loc[DATES[1], "A.XSHE"] == pytest.approx(0.3)


# --------------------------------------------------------------------------- 图8/9

def test_contribution_gini_skips_periods_with_too_few_names():
    """研报 4.3 节：盈利或亏损股票不足 10 只的期不计算。"""
    returns = pd.DataFrame([[0.1, 0.2, -0.1, -0.2]], index=DATES[:1], columns=IDS)
    assert a.contribution_gini(returns).isna().all().all()


def test_contribution_gini_splits_winners_from_losers_by_sign():
    """赢家 [1,2,3,4] 阶梯 → 0.25；输家全部等额 → 0。都手算得出。"""
    winners = [0.01, 0.02, 0.03, 0.04] * 3               # 12 只，成比例的阶梯
    losers = [-0.05] * 12
    returns = pd.DataFrame([winners + losers], index=DATES[:1])
    got = a.contribution_gini(returns, min_names=10)
    assert got.loc[DATES[0], "盈利"] == pytest.approx(0.25)
    assert got.loc[DATES[0], "亏损"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- 图10

def test_cap_percentile_ranks_the_whole_cross_section_not_just_the_portfolio():
    percentile = a.cap_percentile(frame([[1e8, 2e8, 3e8, 4e8]] * 3))
    assert list(percentile.loc[DATES[0]]) == [0.25, 0.5, 0.75, 1.0]


def test_percentile_shift_is_the_change_in_the_portfolios_average_rank():
    caps = frame([[1e8, 2e8, 3e8, 4e8],                  # A=0.25, B=0.5
                  [4e8, 2e8, 3e8, 1e8],                  # A=1.00, B=0.5
                  [1e8, 2e8, 3e8, 4e8]])
    shift = a.percentile_shift(picks([["A.XSHE", "B.XSHE"], ["A.XSHE"]]),
                               a.cap_percentile(caps))
    # 期初 (0.25+0.5)/2 = 0.375，期末 (1.0+0.5)/2 = 0.75
    assert shift.loc[DATES[1]] == pytest.approx(0.375)


def test_percentile_shift_drops_a_name_from_both_ends_when_it_stops_quoting():
    """只在期末剔除会把差值系统性抬高——退市的正是期初分位数最低的那只。"""
    caps = frame([[1e8, 2e8, 3e8, 4e8],
                  [1e8, np.nan, 3e8, 4e8],
                  [1e8, 2e8, 3e8, 4e8]])
    shift = a.percentile_shift(picks([["A.XSHE", "B.XSHE"], ["A.XSHE"]]),
                               a.cap_percentile(caps))
    # B 两端都被剔掉，只剩 A：期初 0.25 → 期末 1/3
    assert shift.loc[DATES[1]] == pytest.approx(1 / 3 - 0.25)


# --------------------------------------------------------------------------- 图11–14

def test_price_percentile_is_one_all_the_way_up_a_monotone_rally():
    """每天都是历史新高 → 时序分位数恒为 1。"""
    prices = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-06", periods=4))
    assert list(a.price_percentile_series(prices)) == [1.0, 1.0, 1.0, 1.0]


def test_price_percentile_counts_the_days_at_or_below_todays_price():
    prices = pd.Series([10.0, 20.0, 15.0, 5.0],
                       index=pd.bdate_range("2020-01-06", periods=4))
    # 第 3 天：历史 {10,20,15} 中不高于 15 的有 2 天 → 2/3
    # 第 4 天：历史 {10,20,15,5} 中不高于 5 的只有自己 → 1/4
    assert list(a.price_percentile_series(prices)) == pytest.approx([1.0, 1.0, 2 / 3, 0.25])


def test_price_percentile_ignores_days_the_stock_had_no_quote():
    prices = pd.Series([10.0, np.nan, 20.0], index=pd.bdate_range("2020-01-06", periods=3))
    assert list(a.price_percentile_series(prices)) == [1.0, 1.0]


def test_event_percentiles_use_only_history_up_to_the_event_day():
    """卖出时刻的分位数不能偷看卖出之后的价格，否则「高卖」是循环论证。"""
    sessions = pd.bdate_range("2020-01-06", periods=4)
    close = pd.DataFrame({"A.XSHE": [10.0, 20.0, 15.0, 100.0]}, index=sessions)
    events = pd.DataFrame({"date": [sessions[2]], "order_book_id": ["A.XSHE"],
                           "event": [a.EXIT]})
    assert a.event_percentiles(events, close)["percentile"].iloc[0] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- 图15–18

def test_industry_weights_are_cap_weighted_not_headcount_weighted():
    caps = pd.Series({"A.XSHE": 90.0, "B.XSHE": 5.0, "C.XSHE": 5.0})
    industry = pd.Series({"A.XSHE": "机械", "B.XSHE": "医药", "C.XSHE": "医药"})
    weights = a.industry_weights(caps.index, caps, industry)
    assert weights["机械"] == pytest.approx(0.9)          # 只数占比只有 1/3
    assert weights["医药"] == pytest.approx(0.1)


def test_industry_weights_drop_names_with_no_industry_mapping():
    caps = pd.Series({"A.XSHE": 50.0, "B.XSHE": 50.0})
    industry = pd.Series({"A.XSHE": "机械"})
    assert a.industry_weights(caps.index, caps, industry).sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- 图19/20–24

def members(rows):
    return picks(rows)


def test_aggregate_pb_totals_first_and_divides_second():
    """整体法 = Σ市值/Σ净资产 = 300/150 = 2；个股 PB 的算术平均是 2.5。"""
    caps = frame([[100.0, 200.0, 0.0, 0.0]] * 3)
    pb = frame([[1.0, 4.0, 1.0, 1.0]] * 3)
    got = a.aggregate_pb(members([["A.XSHE", "B.XSHE"]]), caps, pb)
    assert got.loc[DATES[0]] == pytest.approx(2.0)


def test_aggregate_pb_ignores_names_whose_pb_is_missing_on_both_sides():
    caps = frame([[100.0, 200.0, 0.0, 0.0]] * 3)
    pb = frame([[1.0, np.nan, 1.0, 1.0]] * 3)
    got = a.aggregate_pb(members([["A.XSHE", "B.XSHE"]]), caps, pb)
    assert got.loc[DATES[0]] == pytest.approx(1.0)        # 只剩 A：100/100


def test_median_cap_takes_the_middle_holding_not_the_middle_of_the_market():
    caps = frame([[1e8, 2e8, 3e8, 9e8]] * 3)
    got = a.median_cap(members([["A.XSHE", "B.XSHE", "C.XSHE"]]), caps)
    assert got.loc[DATES[0]] == pytest.approx(2e8)


def test_aggregate_turnover_weights_by_free_float_via_the_turnover_identity():
    """成交额 [100,300]、换手率 [1%,2%] → 流通市值 [10000,15000]，
    整体法 400/25000 = 1.60%，等权平均则是 1.50%。"""
    amount = frame([[100.0, 300.0, 0.0, 0.0]] * 3)
    rate = frame([[1.0, 2.0, 1.0, 1.0]] * 3)
    got = a.aggregate_turnover(members([["A.XSHE", "B.XSHE"]]), amount, rate)
    assert got.loc[DATES[0]] == pytest.approx(0.016)


# --------------------------------------------------------------------------- 图25

def test_nominal_turnover_counts_names_in_and_out_over_portfolio_size():
    """两只的组合换掉一只 = 进 1 出 1 → 2/2 = 100%。"""
    got = a.nominal_turnover(picks([["A.XSHE", "B.XSHE"], ["A.XSHE", "C.XSHE"]]))
    assert got.loc[DATES[0]] == pytest.approx(1.0)        # 建仓：进 2 出 0
    assert got.loc[DATES[1]] == pytest.approx(1.0)


def test_nominal_turnover_is_zero_when_nothing_changes():
    got = a.nominal_turnover(picks([["A.XSHE"], ["A.XSHE"]]))
    assert got.loc[DATES[1]] == pytest.approx(0.0)


# --------------------------------------------------------------------------- 图27

def test_rolling_correlation_of_a_series_with_itself_is_one():
    sessions = pd.bdate_range("2020-01-06", periods=8)
    nav = pd.Series(np.linspace(1.0, 2.0, 8) ** 2, index=sessions)
    index_close = pd.DataFrame({"000852.XSHG": nav.to_numpy() * 3.0}, index=sessions)
    got = a.rolling_correlation(nav, index_close, window=4)
    assert got["000852.XSHG"].dropna().round(9).eq(1.0).all()


# --------------------------------------------------------------------------- 图28

def test_calendar_effect_is_measured_against_the_periods_own_average():
    """构造：只有周三多涨 1pp，其余四天相同 → 周三超额 +0.8pp，其余 −0.2pp。"""
    sessions = pd.bdate_range("2020-01-06", periods=5)     # 恰好周一到周五各一天
    returns = np.array([0.0, 0.0, 0.01, 0.0, 0.0])
    nav = pd.Series(np.concatenate([[1.0], np.cumprod(1 + returns)]),
                    index=pd.DatetimeIndex([pd.Timestamp("2020-01-03")]).append(sessions))
    got = a.calendar_effect(nav, {"全时段": (sessions[0], sessions[-1])})
    assert got.loc["周三", "全时段"] == pytest.approx(0.008, abs=1e-4)
    assert got.loc["周一", "全时段"] == pytest.approx(-0.002, abs=1e-4)


def test_calendar_effect_rows_are_labelled_monday_through_friday():
    sessions = pd.bdate_range("2020-01-06", periods=10)
    nav = pd.Series(np.linspace(1.0, 1.1, 10), index=sessions)
    got = a.calendar_effect(nav, {"全时段": (sessions[0], sessions[-1])})
    assert list(got.index) == ["周一", "周二", "周三", "周四", "周五"]


# --------------------------------------------------------------------------- 成员视图

def test_daily_holdings_carry_forward_until_the_next_rebalance():
    sessions = pd.bdate_range("2020-01-02", "2020-03-03")
    held = a.daily(picks([["A.XSHE"], ["B.XSHE"]]), sessions)
    assert held.loc["2020-01-31", "A.XSHE"] and not held.loc["2020-01-31", "B.XSHE"]
    assert held.loc["2020-02-28", "B.XSHE"] and not held.loc["2020-02-28", "A.XSHE"]


def test_days_before_the_first_rebalance_hold_nothing():
    sessions = pd.bdate_range("2019-12-02", "2020-01-31")
    held = a.daily(picks([["A.XSHE"]]), sessions)
    assert not held.loc["2019-12-31"].any()
    assert held.dtypes.eq(bool).all()


def test_index_members_leave_a_gap_where_the_index_has_no_constituents():
    """中证1000 成分股最早只到 2014-10-17，缺口要留着，不能用前值填出来。"""
    components = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-02-03"]),
        "index_id": ["000852.XSHG", "000852.XSHG", "000852.XSHG"],
        "order_book_id": ["A.XSHE", "B.XSHE", "A.XSHE"],
    })
    wide = a.index_members(components, "000852.XSHG")
    assert list(wide.index) == list(pd.to_datetime(["2020-01-02", "2020-02-03"]))
    assert not wide.loc["2020-02-03", "B.XSHE"]
    assert a.index_members(components, "000905.XSHG").empty
