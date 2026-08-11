"""拿本地指数缓存 + 指标口径去对研报本身。

表1、表2 在策略行旁边同时公布了中证1000 的数字。那些行是**纯指数序列**——不涉
股票池、不涉组合构造、不涉成本模型——所以一旦对不上，问题必定出在我们的数据或
口径上，而且可以在回测存在**之前**就解决掉。这正是本文件的全部意义：将来策略
若够不到 43.1%，这些用例让我们有底气说「基准这条链路不是原因」。

最初猜错的两条口径就是被它钉死的：

  * 年化收益率按**自然日**折算。若按 252 交易日折算，同一条指数序列会读出
    10.10% 而不是公布的 9.7%，因为 A 股一年只有约 243 个交易日。
  * 分年度那一列以上年最后一个交易日的收盘为基准，而不是从 1 月首个交易日起算。

与 tests/test_metrics.py 不同，这些用例要读 data/，缓存不在时会 skip。
但它们同样不消耗 rqdatac 额度。
"""
import pandas as pd
import pytest

from smallcap import data, metrics as m

BENCH = "000852.XSHG"

# 表1 的基准行，2012-12-03 ~ 2022-03-31。
T1_BENCH_RETURN = {2013: 31.6, 2014: 34.5, 2015: 76.1, 2016: -20.0, 2017: -17.4,
                   2018: -37.3, 2019: 25.7, 2020: 19.4, 2021: 20.5}
T1_BENCH_VOL = {2013: 23.9, 2014: 21.1, 2015: 46.2, 2016: 31.7, 2017: 16.2,
                2018: 24.8, 2019: 24.9, 2020: 27.3, 2021: 18.9}
# 基准最大回撤连同公布的峰谷日期——九组带日期的固定值。
T1_BENCH_MDD = {
    2013: (14.9, "2013-05-30", "2013-06-28"), 2014: (13.6, "2014-02-18", "2014-04-28"),
    2015: (53.1, "2015-06-12", "2015-09-15"), 2016: (26.9, "2016-01-06", "2016-01-28"),
    2017: (20.0, "2017-01-05", "2017-12-25"), 2018: (42.3, "2018-01-08", "2018-10-18"),
    2019: (22.2, "2019-04-04", "2019-08-09"), 2020: (15.7, "2020-02-25", "2020-04-01"),
    2021: (11.2, "2021-01-05", "2021-02-05"),
}

# 研报只公布到一位小数，且数据源是 Wind 而非 rqdatac，本来就不可能完全一致。
# 最差的年份是 2018，差 0.4pp。
RETURN_TOL, VOL_TOL = 0.6, 0.03


@pytest.fixture(scope="module")
def benchmark():
    try:
        px = data.wide("index_price", "close")
    except FileNotFoundError:
        pytest.skip("index_price cache absent; run scripts/01_fetch.py")
    return px[BENCH].dropna()


@pytest.fixture(scope="module")
def report_window(benchmark):
    return benchmark.loc["2012-12-03":"2022-03-31"]


def test_report_window_spans_the_expected_sessions(report_window):
    assert len(report_window) == 2267
    assert report_window.index[0] == pd.Timestamp("2012-12-03")
    assert report_window.index[-1] == pd.Timestamp("2022-03-31")


def test_full_period_matches_table1_and_table3(report_window):
    """全时段：年化 9.7% / 波动 27.6% / 最大回撤 72.3%（2015-06-12 → 2018-10-18）。"""
    ann = m.annualized_return(report_window)
    vol = m.annualized_vol(m.to_returns(report_window))
    mdd, peak, trough = m.max_drawdown(report_window)

    assert ann * 100 == pytest.approx(9.7, abs=0.1)
    assert vol * 100 == pytest.approx(27.6, rel=VOL_TOL)
    assert mdd * 100 == pytest.approx(72.3, abs=0.2)
    assert peak == pd.Timestamp("2015-06-12")
    assert trough == pd.Timestamp("2018-10-18")
    assert m.sharpe(ann, vol) == pytest.approx(0.28, abs=0.02)
    assert m.calmar(ann, mdd) == pytest.approx(0.134, abs=0.01)


def test_trading_day_annualisation_would_have_missed_it(report_window):
    """守的是这个**发现**而不是代码：按 252 天/年折算会高估约 0.4pp。"""
    n = len(report_window) - 1
    naive = (report_window.iloc[-1] / report_window.iloc[0]) ** (252 / n) - 1
    assert naive * 100 == pytest.approx(10.1, abs=0.1)
    assert abs(naive - m.annualized_return(report_window)) > 0.003


@pytest.fixture(scope="module")
def yearly(report_window):
    return m.performance_table(report_window, report_window)


@pytest.mark.parametrize("year", sorted(T1_BENCH_RETURN))
def test_yearly_return_measured_from_previous_year_end(yearly, year):
    """收益口径：以上年最后一个交易日的收盘为基准。"""
    assert yearly[str(year)]["基准年化收益率"] * 100 == pytest.approx(
        T1_BENCH_RETURN[year], abs=RETURN_TOL
    )


@pytest.mark.parametrize("year", sorted(T1_BENCH_VOL))
def test_yearly_risk_measured_inside_the_year_only(yearly, year):
    """2016 是判别性的那一年；其余年份两种口径都一样，但同样必须成立。"""
    assert yearly[str(year)]["基准年化波动率"] * 100 == pytest.approx(
        T1_BENCH_VOL[year], rel=VOL_TOL
    )


@pytest.mark.parametrize("year", sorted(T1_BENCH_MDD))
def test_yearly_drawdown_matches_table1_with_published_dates(yearly, year):
    mdd, peak, trough = T1_BENCH_MDD[year]
    col = yearly[str(year)]
    assert col["基准最大回撤"] * 100 == pytest.approx(mdd, abs=0.3)
    assert col["基准最大回撤起始"] == pd.Timestamp(peak)
    assert col["基准最大回撤终止"] == pd.Timestamp(trough)


def test_capacity_window_matches_table2_benchmark_column(benchmark):
    """表2：2019-01-07 ~ 2022-03-31，年化 12.8% / 波动 24.2% / 回撤 22.2%。

    这是一个独立的第二窗口，而且短得多——自然日口径必须在两个窗口上同时成立，
    否则就只是巧合。
    """
    window = benchmark.loc["2019-01-07":"2022-03-31"]
    mdd, peak, trough = m.max_drawdown(window)

    assert m.annualized_return(window) * 100 == pytest.approx(12.8, abs=0.1)
    assert m.annualized_vol(m.to_returns(window)) * 100 == pytest.approx(24.2, rel=VOL_TOL)
    assert mdd * 100 == pytest.approx(22.2, abs=0.1)
    assert peak == pd.Timestamp("2019-04-04")
    assert trough == pd.Timestamp("2019-08-09")
