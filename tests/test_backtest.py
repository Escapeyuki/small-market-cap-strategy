"""净值引擎，用手算得出的合成价格逐条钉死。

这些用例存在的理由很具体：如果最终跑出来的年化不对，我们要能立刻说出
「不是引擎算错了」。所以每条用例的期望值都是能在纸上算出来的数，不是把当前
实现的输出抄回来当基准。

**成交口径只有一种**：T 日收盘出信号，T+1 日开盘成交（grill.md Q19）。下面很多
用例把开盘价设成与收盘价相同，这样价格里没有跳空，口径的效果就只剩「晚一天
成交」这一件事，期望值好手算；真正要考察跳空的那几条会单独给出开盘价。
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
        opens = closes                                 # 开盘=收盘：价格里没有跳空
    open_ = pd.DataFrame(opens, index=SESSIONS, columns=IDS, dtype=float)
    return close, open_


def selection(rows):
    """rows: {日期序号: [持有的股票]}，日期序号是**信号日**。"""
    frame = pd.DataFrame(False, index=SESSIONS[sorted(rows)], columns=IDS)
    for i, held in rows.items():
        frame.loc[SESSIONS[i], held] = True
    return frame


def run(rows, closes, opens=None, cost=COST):
    close, open_ = prices(closes, opens)
    return bt.run(selection(rows), close, open_, SESSIONS, cost_per_side=cost)


# --------------------------------------------------------------------------- 信号日

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


# --------------------------------------------------------------------------- 成交日

def test_each_signal_trades_on_the_following_session():
    schedule = bt.trade_dates(SESSIONS[[0, 2]], SESSIONS)
    assert list(schedule.index) == [SESSIONS[0], SESSIONS[2]]
    assert list(schedule.to_numpy()) == [SESSIONS[1], SESSIONS[3]]


def test_a_signal_on_the_final_session_is_dropped_rather_than_traded_same_day():
    """最后一天出的信号没有下一个开盘可用。作废，而不是退回当天收盘成交
    ——退回去就正是这次要删掉的那个未来函数。"""
    assert list(bt.trade_dates(SESSIONS[[5]], SESSIONS).index) == []


def test_trade_dates_ignore_signal_days_that_are_not_sessions():
    stray = pd.DatetimeIndex(["2020-01-07", "2020-01-11"])      # 后者是周六
    assert list(bt.trade_dates(stray, SESSIONS).index) == [SESSIONS[1]]


# --------------------------------------------------------------------------- 净值

def test_the_first_session_is_flat_because_no_signal_can_trade_yet():
    """T 日的信号最早 T+1 开盘成交，所以首个交易日必然空仓持币。

    这不是特例，是口径的必然结果——顺带说明组合**没有**吃到 A 在第 0→1 天的翻倍。
    """
    closes = [[10, 10], [20, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    result = run({0: IDS}, closes)
    assert result.nav.iloc[0] == 1.0
    assert result.nav.iloc[1] == pytest.approx(1 - COST)


def test_a_single_holding_tracks_its_own_price_net_of_the_entry_cost():
    closes = [[10, 10], [10, 10], [12, 10], [12, 10], [12, 10], [12, 10]]
    result = run({0: ["A.XSHE"]}, closes)

    assert result.nav.iloc[1] == pytest.approx(1 - COST)           # 第 1 天开盘按 10 建仓
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * 1.2)  # 建仓费 + 12/10


def test_equal_weight_halves_the_move_of_a_two_stock_portfolio():
    closes = [[10, 10], [10, 10], [12, 10], [12, 10], [12, 10], [12, 10]]
    result = run({0: IDS}, closes)
    assert result.nav.iloc[-1] == pytest.approx((1 - COST) * 1.10)


def test_weights_drift_between_rebalances_instead_of_being_held_equal():
    """A 涨 100% 之后应占 2/3 仓位；若引擎偷偷再平衡，下一天的涨幅会算错。"""
    closes = [[10, 10], [10, 10], [20, 10], [40, 10], [40, 10], [40, 10]]
    result = run({0: IDS}, closes)

    # 第 1 天开盘按 (10,10) 各半建仓
    # 第 2 天：(1+1)/2 = 1.5 倍；第 3 天：A 占 2/3，再涨 100% → 1.5 * (1 + 2/3) = 2.5
    assert result.nav.iloc[2] == pytest.approx((1 - COST) * 1.5)
    assert result.nav.iloc[3] == pytest.approx((1 - COST) * 2.5)


def test_a_full_swap_costs_exactly_the_round_trip_rate():
    """整仓换股 = 卖光 + 买光 = sum|Δw| = 2 → 双边千分之三。"""
    closes = [[10, 10]] * 6
    result = run({0: ["A.XSHE"], 2: ["B.XSHE"]}, closes)

    assert list(result.turnover.index) == [SESSIONS[1], SESSIONS[3]]   # 按成交日索引
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


def test_zero_cost_reproduces_the_gross_price_path():
    closes = [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    result = run({0: ["A.XSHE"]}, closes, cost=0.0)
    assert result.nav.iloc[-1] == pytest.approx(2.0)


def test_recorded_weights_are_equal_and_sum_to_one():
    closes = [[10, 10]] * 6
    result = run({0: IDS}, closes)
    assert list(result.weights.index) == [SESSIONS[1]]            # 成交日，不是信号日
    assert result.weights.iloc[0].tolist() == [0.5, 0.5]
    assert result.drifted.iloc[0].sum() == pytest.approx(0.0)     # 建仓前空仓


# --------------------------------------------------------------------------- 隔夜 / 盘中两腿

def test_a_gap_up_between_the_signal_and_the_open_is_not_captured():
    """T 日收盘选中、T+1 跳空高开：这段差价**吃不到**，因为要到跳空后的开盘才买。

    这正是删掉研报口径所放弃的东西。小市值策略里这段差价是系统性的——被选中的
    票往往刚跌过，隔夜常有反弹——所以它必须以「没吃到」的形式出现在净值里。
    """
    closes = [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    opens = [[10, 10], [10, 10], [20, 10], [20, 10], [20, 10], [20, 10]]
    result = run({1: ["A.XSHE"]}, closes, opens)       # T=第 1 天收盘选中，T+1 跳空一倍

    assert result.nav.iloc[-1] == pytest.approx(1 - COST)      # 在跳空后的开盘价才买入


def test_the_trade_day_splits_into_an_overnight_leg_and_an_intraday_leg():
    """成交日当天：开盘前用旧权重，开盘后用新权重。"""
    closes = [[10, 10], [10, 10], [12, 20], [12, 20], [12, 20], [12, 20]]
    opens = [[10, 10], [10, 10], [10, 10], [12, 20], [12, 20], [12, 20]]
    result = run({0: ["B.XSHE"], 1: ["A.XSHE"]}, closes, opens)

    # 第 0 天的信号在第 1 天开盘建仓 B；第 1 天的信号在第 2 天开盘换成 A。
    # 第 2 天开盘价与前收相同 → 隔夜腿无收益；换股付 2*COST；盘中 A 由 10 涨到 12。
    assert result.nav.iloc[2] == pytest.approx((1 - COST) * (1 - 2 * COST) * 1.2)


def test_splitting_a_day_into_legs_leaves_a_zero_turnover_rebalance_untouched():
    """两腿乘回去必须正好是当天的收盘到收盘收益。

    构造两次跑：同一天在一次里是普通持有日（走收盘到收盘），在另一次里是成交日
    但目标持仓不变（走隔夜 × 盘中）。两条净值必须逐点相等——否则拆腿这一步就在
    凭空造出或吃掉一段收益，而那种错会均匀地渗进整条曲线，很难被发现。
    """
    closes = [[10, 10], [10, 10], [12, 10], [15, 10], [15, 10], [15, 10]]
    opens = [[10, 10], [10, 10], [11, 10], [15, 10], [15, 10], [15, 10]]
    plain = run({0: ["A.XSHE"]}, closes, opens)                    # 第 2 天是普通持有日
    again = run({0: ["A.XSHE"], 1: ["A.XSHE"]}, closes, opens)     # 第 2 天是成交日但不换股

    assert again.turnover.iloc[1] == pytest.approx(0.0)
    assert plain.nav.round(12).tolist() == again.nav.round(12).tolist()


def test_a_signal_on_the_final_session_does_not_crash_the_engine():
    closes = [[10, 10]] * 6
    result = run({0: ["A.XSHE"], 5: ["B.XSHE"]}, closes)
    assert len(result.weights) == 1                 # 最后一天的信号作废，不是崩掉
