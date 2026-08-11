"""净值引擎，用手算得出的合成价格逐条钉死。

这些用例存在的理由很具体：如果最终跑出来的年化不是 43.1%，我们要能立刻说出
「不是引擎算错了」。所以每条用例的期望值都是能在纸上算出来的数，不是把当前
实现的输出抄回来当基准。
"""
import numpy as np
import pandas as pd
import pytest

from smallcap import backtest as bt

SESSIONS = pd.bdate_range("2020-01-06", periods=6)     # 周一到下周一
IDS = ["A.XSHE", "B.XSHE"]
COST = 0.0015                                          # 单边，往返千分之三


def prices(closes, opens=None):
    close = pd.DataFrame(closes, index=SESSIONS, columns=IDS, dtype=float)
    if opens is None:
        opens = closes                                 # 开盘=收盘：两种口径应当一致
    open_ = pd.DataFrame(opens, index=SESSIONS, columns=IDS, dtype=float)
    return close, open_


def selection(rows):
    """rows: {日期序号: [持有的股票]}"""
    frame = pd.DataFrame(False, index=SESSIONS[sorted(rows)], columns=IDS)
    for i, held in rows.items():
        frame.loc[SESSIONS[i], held] = True
    return frame


def run(rows, closes, opens=None, mode="report_close", cost=COST):
    close, open_ = prices(closes, opens)
    return bt.run(selection(rows), close, open_, SESSIONS, mode=mode, cost_per_side=cost)


# --------------------------------------------------------------------------- 调仓日

def test_monthly_rebalance_takes_the_first_session_of_each_month():
    calendar = pd.bdate_range("2012-12-03", "2013-03-29")
    dates = bt.rebalance_dates(calendar, "monthly", "2012-12-03", "2013-03-29")
    assert list(dates.strftime("%Y-%m-%d")) == [
        "2012-12-03", "2013-01-01", "2013-02-01", "2013-03-01"
    ]


def test_monthly_rebalance_skips_to_the_next_session_when_the_first_is_a_holiday():
    """真实日历里 2013-01-01 休市，第一个交易日是 01-04。"""
    calendar = pd.DatetimeIndex(["2012-12-03", "2012-12-31", "2013-01-04", "2013-01-07"])
    dates = bt.rebalance_dates(calendar, "monthly", "2012-12-01", "2013-01-31")
    assert list(dates.strftime("%Y-%m-%d")) == ["2012-12-03", "2013-01-04"]


def test_weekly_rebalance_takes_the_first_session_of_each_week():
    dates = bt.rebalance_dates(SESSIONS, "weekly", SESSIONS[0], SESSIONS[-1])
    assert list(dates.strftime("%Y-%m-%d")) == ["2020-01-06", "2020-01-13"]


def test_daily_rebalance_is_every_session():
    assert list(bt.rebalance_dates(SESSIONS, "daily", SESSIONS[0], SESSIONS[-1])) == list(SESSIONS)


# --------------------------------------------------------------------------- 净值

def test_a_single_holding_tracks_its_own_price_net_of_the_entry_cost():
    closes = [[10, 10], [11, 10], [12, 10], [12, 10], [12, 10], [12, 10]]
    result = run({0: ["A.XSHE"]}, closes)

    assert result.nav.iloc[0] == 1.0                              # 期初资金
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * 1.2)  # 建仓费 + 12/10


def test_equal_weight_halves_the_move_of_a_two_stock_portfolio():
    closes = [[10, 10], [12, 10], [12, 10], [12, 10], [12, 10], [12, 10]]
    result = run({0: IDS}, closes)
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * 1.10)


def test_weights_drift_between_rebalances_instead_of_being_held_equal():
    """A 涨 100% 之后应占 2/3 仓位；若引擎偷偷再平衡，第二天涨幅会算错。"""
    closes = [[10, 10], [20, 10], [40, 10], [40, 10], [40, 10], [40, 10]]
    result = run({0: IDS}, closes)

    # 第 1 天：(1+1)/2 = 1.5 倍；第 2 天：A 占 2/3，再涨 100% → 1.5 * (1 + 2/3) = 2.5
    assert result.nav.iloc[1] == pytest.approx((1 - COST) * 1.5)
    assert result.nav.iloc[2] == pytest.approx((1 - COST) * 2.5)


def test_a_full_swap_costs_exactly_the_round_trip_rate():
    """整仓换股 = 卖光 + 买光 = sum|Δw| = 2 → 双边千分之三。"""
    closes = [[10, 10]] * 6
    result = run({0: ["A.XSHE"], 2: ["B.XSHE"]}, closes)

    assert result.turnover.iloc[0] == pytest.approx(1.0)          # 建仓只有买
    assert result.turnover.iloc[1] == pytest.approx(2.0)          # 全换
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * (1 - 2 * COST))


def test_holding_the_same_names_costs_nothing():
    closes = [[10, 10]] * 6
    result = run({0: IDS, 2: IDS, 4: IDS}, closes)
    assert result.turnover.iloc[1] == pytest.approx(0.0)
    assert result.nav.iloc[-1] == pytest.approx(1 - COST)


def test_turnover_counts_only_the_part_that_actually_changed():
    """两只换掉一只 = 半仓换手 → 双边换手率 1.0。"""
    closes = [[10, 10]] * 6
    result = run({0: IDS, 2: ["A.XSHE"]}, closes)
    assert result.turnover.iloc[1] == pytest.approx(1.0)


def test_a_halted_holding_contributes_zero_return_not_a_gap():
    """停牌日后复权收盘价沿用前值，持仓当天收益为 0。"""
    closes = [[10, 10], [10, 10], [10, 10], [10, 10], [10, 10], [20, 10]]
    result = run({0: ["A.XSHE"]}, closes)
    assert result.nav.iloc[3] == pytest.approx(1 - COST)
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * 2)


def test_a_delisted_holding_is_frozen_at_its_last_price_not_marked_to_zero():
    """退市后价格变 NaN。按 0 收益处理 = 以最后一个有效价清仓后持币。"""
    closes = [[10, 10], [10, 10], [np.nan, 10], [np.nan, 10], [np.nan, 10], [np.nan, 10]]
    result = run({0: ["A.XSHE"]}, closes)
    assert result.nav.iloc[-1] == pytest.approx(1 - COST)
    assert np.isfinite(result.nav).all()


# --------------------------------------------------------------------------- 两种成交口径

def test_next_open_is_report_close_shifted_by_exactly_one_session():
    """开盘=收盘时价格里没有隔夜跳空，两种口径的差别就只剩「晚一天成交」。

    信号错开一天之后两条净值必须逐点相等——这说明 next_open 的两腿拆分没有
    凭空多算或漏算收益，只是把成交推后了一个交易日。
    """
    closes = [[10, 10], [11, 10], [12, 10], [13, 10], [14, 10], [15, 10]]
    later = run({1: ["A.XSHE"]}, closes, mode="report_close")
    earlier = run({0: ["A.XSHE"]}, closes, mode="next_open")
    assert later.nav.round(12).tolist() == earlier.nav.round(12).tolist()


def test_next_open_misses_the_overnight_gap_that_report_close_captures():
    """T 日收盘选中、T+1 跳空高开：研报口径吃到这段，实盘口径吃不到。

    小市值策略里这段差价是系统性的——被选中的票往往刚跌过，隔夜常有反弹。
    """
    closes = [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    opens = [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    signal = {1: ["A.XSHE"]}                       # T=第 1 天收盘选中，T+1 跳空一倍

    report = run(signal, closes, opens, mode="report_close")
    live = run(signal, closes, opens, mode="next_open")

    assert report.nav.iloc[-1] == pytest.approx((1 - COST) * 2)
    assert live.nav.iloc[-1] == pytest.approx(1 - COST)     # 在跳空后的开盘价才买入


def test_next_open_splits_the_day_into_overnight_and_intraday_legs():
    """成交日当天：开盘前用旧权重，开盘后用新权重。"""
    closes = [[10, 10], [10, 10], [12, 20], [12, 20], [12, 20], [12, 20]]
    opens = [[10, 10], [10, 10], [10, 10], [12, 20], [12, 20], [12, 20]]
    result = run({0: ["B.XSHE"], 1: ["A.XSHE"]}, closes, opens, mode="next_open")

    # 第 0 天的信号在第 1 天开盘建仓 B；第 1 天的信号在第 2 天开盘换成 A。
    # 第 2 天开盘价与前收相同 → 隔夜腿无收益；换股付 2*COST；盘中 A 由 10 涨到 12。
    assert result.nav.iloc[2] == pytest.approx((1 - COST) * (1 - 2 * COST) * 1.2)


def test_a_signal_on_the_final_session_cannot_be_traded_next_open():
    closes = [[10, 10]] * 6
    result = run({0: ["A.XSHE"], 5: ["B.XSHE"]}, closes, mode="next_open")
    assert len(result.weights) == 1                 # 最后一天的信号作废，不是崩掉


def test_zero_cost_reproduces_the_gross_price_path():
    closes = [[10, 10], [20, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    result = run({0: ["A.XSHE"]}, closes, cost=0.0)
    assert result.nav.iloc[-1] == pytest.approx(2.0)


def test_recorded_weights_are_equal_and_sum_to_one():
    closes = [[10, 10]] * 6
    result = run({0: IDS}, closes)
    assert result.weights.iloc[0].tolist() == [0.5, 0.5]
    assert result.drifted.iloc[0].sum() == pytest.approx(0.0)     # 建仓前空仓
