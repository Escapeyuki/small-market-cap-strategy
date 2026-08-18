"""专题之二新增机制，用手算/不变量钉死：多频率、择时空仓、跌停惩罚、逐级选股、
容量顺延、以及专题二三个指标（胜率 / 策略·基准平均分位数）。

期望值要么能在纸上算出，要么是清晰的不变量（如「跌停当天不卖出 → 换手更低」）。
"""
import numpy as np
import pandas as pd
import pytest

from smallcap import backtest as bt, enhance as en, metrics as m, universe as u

COST = 0.0015


# --------------------------------------------------------------------------- 多频率

def test_new_frequencies_bimonthly_quarterly_monthend():
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    bim = bt.rebalance_dates(cal, "bimonthly", cal[0], cal[-1])
    assert list(bim.month) == [1, 3, 5, 7, 9, 11]                 # 隔月首个交易日
    q = bt.rebalance_dates(cal, "quarterly", cal[0], cal[-1])
    assert list(q.month) == [1, 4, 7, 10]
    me = bt.rebalance_dates(cal, "monthend", cal[0], cal[-1])
    assert 12 not in me.month and 3 not in me.month              # E12：除 12 月、3 月
    assert set(me.month) == {1, 2, 4, 5, 6, 7, 8, 9, 10, 11}


def test_biweekly_takes_every_other_week():
    cal = pd.bdate_range("2020-01-06", periods=40)               # 8 整周
    bw = bt.rebalance_dates(cal, "biweekly", cal[0], cal[-1])
    wk = bt.rebalance_dates(cal, "weekly", cal[0], cal[-1])
    assert list(bw) == list(wk[::2])                             # 隔周取周首日


# --------------------------------------------------------------------------- 择时空仓

def test_cash_dates_liquidate_to_flat_nav():
    """择时空仓：清仓后持币，净值走平——即便持仓股继续上涨（E9）。"""
    sessions = pd.bdate_range("2020-01-06", periods=6)
    ids = ["A.XSHE"]
    close = pd.DataFrame({"A.XSHE": [100, 100, 110, 121, 133.1, 146.41]},
                         index=sessions, dtype=float)
    sel = pd.DataFrame(False, index=[sessions[0], sessions[3]], columns=ids)
    sel.loc[sessions[0], "A.XSHE"] = True                        # 信号0 建仓 A
    # 信号3 空仓（cash_dates）→ 成交日 day4 清仓
    res = bt.run(sel, close, close, sessions, cost_per_side=COST,
                 cash_dates=[sessions[3]])
    nav = res.nav
    assert nav.iloc[2] < nav.iloc[3]                             # 清仓前 A 涨，净值涨
    assert nav.iloc[4] == pytest.approx(nav.iloc[5])            # 清仓后走平
    assert nav.iloc[5] == pytest.approx(nav.iloc[4])            # 尽管 day5 的 A 仍在涨


# ------------------------------------------------------------------- 日历月择时构造

def test_apply_timing_calendar_month_cashout_and_reentry():
    """apply_timing 日历月口径（E9）—— 接缝里最易错的一处，手算钉死：

    空仓月**首个交易日**清仓；**下一个非空仓月首个交易日**按最近一次原生选股 ffill 再建仓；
    空仓月内没有原生调仓也照样在下月初再入场（这正是粗频率不能「清仓持到下次调仓」的原因）。
    """
    sessions = pd.bdate_range("2019-12-01", "2020-05-31")        # 工作日历（周末剔除）
    ids = ["A", "B", "C"]
    # 原生选股只有一/四月两次（模拟季度级稀疏）：一月 {A,B}、四月 {B,C}。
    sel = pd.DataFrame(False, index=pd.DatetimeIndex(["2020-01-01", "2020-04-01"]), columns=ids)
    sel.loc["2020-01-01", ["A", "B"]] = True
    sel.loc["2020-04-01", ["B", "C"]] = True

    aug, cash_out = en.apply_timing(sel, sessions, cash_months=(1,))

    # 清仓信号 = 一月首个交易日（2020-01-01 是周三）
    assert list(cash_out) == [pd.Timestamp("2020-01-01")]
    # 增广索引 = {一月首日(清仓), 二月首日(再入场), 四月首日(原生)}——二月即便无原生调仓也再入场
    assert set(aug.index) == {pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-03"),
                              pd.Timestamp("2020-04-01")}
    assert not aug.loc["2020-01-01"].any()                       # 一月首日清仓：全 False
    # 二月首日按最近原生（一月 {A,B}）ffill 再建仓——而非空到四月
    assert set(aug.columns[aug.loc["2020-02-03"]]) == {"A", "B"}
    assert set(aug.columns[aug.loc["2020-04-01"]]) == {"B", "C"}


# --------------------------------------------------------------------------- 跌停惩罚

def test_limit_down_penalty_holds_the_stuck_exit_one_more_day():
    """跌停调出位当天卖不掉：换手比无惩罚时低，且净值路径不同（E7）。"""
    sessions = pd.bdate_range("2020-01-06", periods=6)
    ids = ["A.XSHE", "B.XSHE"]
    close = pd.DataFrame({"A.XSHE": [100, 100, 100, 90, 95, 95],
                          "B.XSHE": [100, 100, 100, 100, 100, 100]},
                         index=sessions, dtype=float)
    sel = pd.DataFrame(False, index=[sessions[0], sessions[2]], columns=ids)
    sel.loc[sessions[0]] = [True, True]                         # 信号0 持 A、B
    sel.loc[sessions[2], "B.XSHE"] = True                       # 信号2 只留 B → 调出 A
    # A 在成交日 day3 收盘跌停（day2→day3 −10%），day4 不再跌停
    ld = pd.DataFrame(False, index=sessions, columns=ids)
    ld.loc[sessions[3], "A.XSHE"] = True

    without = bt.run(sel, close, close, sessions, cost_per_side=COST)
    withp = bt.run(sel, close, close, sessions, cost_per_side=COST, limit_down=ld)

    # 成交日 day3 那次调仓：带惩罚时 A 冻结不卖 → 换手更低
    assert withp.turnover.loc[sessions[3]] < without.turnover.loc[sessions[3]]
    # 两条净值路径不同（惩罚版把 A 多持有到 day4 才卖）
    assert not np.allclose(withp.nav.values, without.nav.values)


# --------------------------------------------------------------------------- 逐级选股

def _all_eligible_panel(caps):
    """构造一个所有股票都合格的单日 Panel，只有市值不同，供 cascade 测试。"""
    date = pd.DatetimeIndex(["2020-06-01"])
    ids = [f"S{i}.XSHE" for i in range(len(caps))]
    one = lambda v: pd.DataFrame([v] * len(ids), index=date.repeat(1), columns=["x"])  # noqa
    frame = lambda vals: pd.DataFrame([vals], index=date, columns=ids)
    return u.Panel(
        market_cap=frame(caps),
        raw_close=frame([10.0] * len(ids)),
        limit_up=frame([20.0] * len(ids)),                     # 远离涨停
        volume=frame([1.0] * len(ids)),                        # 有成交
        st=frame([False] * len(ids)),
        listed_days=frame([9999] * len(ids)),                  # 上市够久
        registration=frame([False] * len(ids)),
    ), date, ids


def test_cascade_two_stage_smallest_cap_then_lowest_vol():
    panel, date, ids = _all_eligible_panel([5, 1, 4, 2, 3])     # 市值
    vol = pd.DataFrame([[0.9, 0.9, 0.1, 0.2, 0.9]], index=date, columns=ids)
    # 第一级：市值最小 3 只 = S1(1), S3(2), S4(3)。第二级：其中波动率最低 2 只
    #   S1 vol .9, S3 vol .2, S4 vol .9 → 取 S3、S1？ 不：最低两只是 S3(.2) 与 S1(.9)/S4(.9)
    #   .9 并列，rank method=first 取列序靠前的 S1 → 结果 {S1, S3}
    sel = u.cascade(panel, [("cap", True, 3), ("vol", True, 2)], {"vol": vol})
    chosen = set(sel.columns[sel.iloc[0]])
    assert chosen == {"S1.XSHE", "S3.XSHE"}


def test_cascade_single_stage_equals_smallest():
    panel, date, ids = _all_eligible_panel([5, 1, 4, 2, 3])
    casc = u.cascade(panel, [("cap", True, 2)])
    small = u.smallest(panel, 2, predicates=u.BUYABLE2)
    assert (casc.values == small.values).all()


# --------------------------------------------------------------------------- 容量顺延

def test_capacity_cap_slows_deployment_vs_unlimited():
    """5% 成交额封顶 → 大资金铺不满仓，涨行情里净值落后于无约束（E8）。"""
    sessions = pd.bdate_range("2020-01-06", periods=4)
    ids = ["A.XSHE"]
    close = pd.DataFrame({"A.XSHE": [100, 110, 121, 133.1]}, index=sessions, dtype=float)
    turn = pd.DataFrame({"A.XSHE": [1000.0] * 4}, index=sessions)   # 日成交额 1000 元
    sel = pd.DataFrame(True, index=[sessions[0]], columns=ids)
    capped = bt.run_with_capacity(sel, close, turn, sessions, 1000.0,
                                  max_participation=0.05, cost_per_side=COST)   # 每日限 50
    free = bt.run_with_capacity(sel, close, turn, sessions, 1000.0,
                                max_participation=1e9, cost_per_side=COST)      # 无约束
    assert capped.iloc[-1] < free.iloc[-1]                      # 封顶铺得慢，涨行情里落后
    assert free.iloc[-1] > 1.2                                  # 无约束≈满仓吃到 A 的涨幅


# --------------------------------------------------------------------------- 专题二指标

def test_win_rate_counts_periods_beating_benchmark():
    tds = pd.bdate_range("2020-01-06", periods=4)
    strat = pd.Series([1.0, 1.2, 1.1, 1.4], index=tds)         # 期收益 +.2, -.083, +.27
    bench = pd.Series([1.0, 1.1, 1.2, 1.3], index=tds)         # 期收益 +.1, +.09, +.083
    # 期1 strat>bench, 期2 strat<bench, 期3 strat>bench → 2/3
    assert m.win_rate(strat, bench, tds) == pytest.approx(2 / 3)


def test_win_rate_excludes_cash_periods():
    tds = pd.bdate_range("2020-01-06", periods=4)
    strat = pd.Series([1.0, 1.0, 0.9, 1.4], index=tds)         # 期0 现金(0%)、期1 亏、期2 赢
    bench = pd.Series([1.0, 1.1, 1.2, 1.3], index=tds)
    # 排除期0（起点 tds[0] 为 cash）后剩期1(strat −.1 < bench .09)、期2(strat +.56 > bench .083) → 1/2
    assert m.win_rate(strat, bench, tds, cash_trade_dates=[tds[0]]) == pytest.approx(1 / 2)


def test_avg_and_benchmark_percentile_rank_in_cross_section():
    # 需要 2 个成交日才有 1 个可测的持有期：signal 在 t0、t2，成交在 t1、t3，
    # 持有期 = [t1, t3]。t1→t3 收益：A +50%, B +40%, C +10%, D −10%。
    tds = pd.bdate_range("2020-01-06", periods=4)
    ids = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        {"A": [1, 1, 1, 1.5], "B": [1, 1, 1, 1.4], "C": [1, 1, 1, 1.1], "D": [1, 1, 1, 0.9]},
        index=tds, dtype=float)
    sel = pd.DataFrame(False, index=[tds[0], tds[2]], columns=ids)
    sel.loc[tds[0], ["A", "D"]] = True                         # 组合 = A、D（等权收益 +20%）
    sel.loc[tds[2], ["A"]] = True                              # 第二期无完整持有期，被跳过
    sched = pd.Series([tds[1], tds[3]], index=[tds[0], tds[2]], dtype="datetime64[ns]")
    # 组合收益 +20% 高于 C(.1)、D(-.1)，低于 A(.5)、B(.4) → 2/4 在其下 → 分位 0.5
    assert m.avg_percentile(sel, close, sched) == pytest.approx(0.5)
    # 基准收益 +20% → 同样 2/4 在其下 → 0.5。基准分位数按**成交日边界**算（与驱动一致），
    # 传入的是成交日 [t1, t3]，不是全部 session。
    bench = pd.Series([1, 1, 1, 1.2], index=tds, dtype=float)
    trade_bounds = pd.DatetimeIndex([tds[1], tds[3]])
    assert m.benchmark_percentile(bench, close, trade_bounds) == pytest.approx(0.5)
