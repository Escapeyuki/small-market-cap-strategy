"""拿研报自己公布的数字来校验指标公式。

这些用例不需要任何行情数据，也不消耗 rqdatac 额度。它们在回测存在之前就把公式
钉死，这样将来若对不上 43.1%，可以归因到**回测**，而不是「公式理解错了」。

表3 是研报里最强的一组固定值：10 个市值档，每档都公布了年化收益、波动率、
最大回撤、夏普、卡玛和信息比率。如果我们的公式是对的，这 10 列必须**同时**
对得平。
"""
import numpy as np
import pandas as pd
import pytest

from smallcap import metrics as m

RF = 0.02
BANDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# 表3 原样照抄（凡是百分比的都按百分数记）。各档的「基准」列完全相同：
# 年化 9.7、波动 27.6、最大回撤 72.3、夏普 0.28、卡玛 0.134。
T3 = pd.DataFrame(
    {
        "ann":         [43.1, 33.8, 31.2, 24.1, 20.3, 16.8, 14.9, 13.9, 14.3, 9.7],
        "vol":         [32.1, 31.4, 31.0, 30.9, 30.7, 30.9, 30.7, 30.4, 30.2, 30.2],
        "mdd":         [54.7, 55.1, 53.5, 52.5, 58.9, 60.2, 65.3, 69.7, 68.0, 71.9],
        "sharpe":      [1.28, 1.013, 0.942, 0.716, 0.596, 0.48, 0.419, 0.391, 0.408, 0.255],
        "calmar":      [0.788, 0.614, 0.583, 0.46, 0.345, 0.279, 0.228, 0.199, 0.211, 0.135],
        "excess_ann":  [33.4, 24.1, 21.5, 14.4, 10.6, 7.1, 5.2, 4.2, 4.6, 0.0],
        "excess_vol":  [12.6, 11.4, 11.0, 10.4, 10.0, 9.8, 9.4, 9.0, 8.6, 8.5],
        "excess_mdd":  [20.4, 16.8, 18.5, 17.6, 20.5, 25.1, 20.1, 16.7, 17.6, 24.9],
        "ir":          [2.652, 2.11, 1.944, 1.387, 1.058, 0.724, 0.551, 0.465, 0.534, -0.002],
        "excess_calmar": [1.637, 1.431, 1.163, 0.816, 0.515, 0.283, 0.256, 0.251, 0.261, -0.001],
    },
    index=BANDS,
)

BENCH_ANN, BENCH_VOL, BENCH_MDD = 9.7, 27.6, 72.3

# 表1 的策略分年度数字（2013-2021 加全时段）。
T1 = pd.DataFrame(
    {
        "ann":    [60.9, 76.7, 267.0, 22.2, -22.5, -17.1, 52.7, 16.4, 45.1, 43.1],
        "vol":    [24.5, 22.7, 56.8, 36.1, 24.6, 31.4, 25.9, 27.1, 22.2, 32.1],
        "mdd":    [14.3, 16.1, 54.7, 28.0, 29.7, 31.6, 15.5, 19.2, 18.9, 54.7],
        "sharpe": [2.402, 3.287, 4.665, 0.559, -0.992, -0.609, 1.958, 0.531, 1.937, 1.28],
        "calmar": [4.274, 4.777, 4.877, 0.792, -0.756, -0.541, 3.402, 0.854, 2.388, 0.788],
    },
    index=["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "全时段"],
)

# 输入本身只公布到一位小数，由它们算出的比值会带上 0.5% 量级的舍入误差。
# 2% 的相对容差是诚实的边界，再紧就是自欺欺人。
REL = 0.02


@pytest.mark.parametrize("band", BANDS)
def test_sharpe_reconciles_across_all_bands(band):
    row = T3.loc[band]
    assert m.sharpe(row["ann"] / 100, row["vol"] / 100, RF) == pytest.approx(
        row["sharpe"], rel=REL
    )


@pytest.mark.parametrize("band", BANDS)
def test_calmar_reconciles_across_all_bands(band):
    row = T3.loc[band]
    assert m.calmar(row["ann"] / 100, row["mdd"] / 100) == pytest.approx(
        row["calmar"], rel=REL
    )


@pytest.mark.parametrize("band", BANDS)
def test_excess_return_is_arithmetic_difference(band):
    """超额年化收益率 = 策略年化 − 基准年化，不是两条净值的比值。"""
    row = T3.loc[band]
    assert row["ann"] - BENCH_ANN == pytest.approx(row["excess_ann"], abs=0.05)


@pytest.mark.parametrize("band", BANDS)
def test_information_ratio_reconciles_across_all_bands(band):
    row = T3.loc[band]
    got = m.information_ratio(row["excess_ann"] / 100, row["excess_vol"] / 100)
    assert got == pytest.approx(row["ir"], rel=REL, abs=0.01)


@pytest.mark.parametrize("band", BANDS)
def test_excess_calmar_uses_excess_return_over_excess_drawdown(band):
    row = T3.loc[band]
    got = m.calmar(row["excess_ann"] / 100, row["excess_mdd"] / 100)
    assert got == pytest.approx(row["excess_calmar"], rel=REL, abs=0.01)


def test_benchmark_sharpe_and_calmar():
    assert m.sharpe(BENCH_ANN / 100, BENCH_VOL / 100, RF) == pytest.approx(0.28, rel=REL)
    assert m.calmar(BENCH_ANN / 100, BENCH_MDD / 100) == pytest.approx(0.134, rel=REL)


@pytest.mark.parametrize("period", T1.index)
def test_table1_yearly_sharpe_and_calmar(period):
    row = T1.loc[period]
    assert m.sharpe(row["ann"] / 100, row["vol"] / 100, RF) == pytest.approx(
        row["sharpe"], rel=REL
    )
    assert m.calmar(row["ann"] / 100, row["mdd"] / 100) == pytest.approx(
        row["calmar"], rel=REL
    )


def test_annualized_return_uses_calendar_days_not_observation_count():
    """两个自然年涨到 4 倍就是年化约 100%，中间摆多少个点都一样。

    按观测点个数折算的口径会对下面两条序列给出不同答案（3 个点 vs 200 个点），
    按自然日折算则不可能。表1 的基准行遵循的是后者——见 tests/test_benchmark.py。
    """
    coarse = pd.Series(
        [1.0, 2.0, 4.0],
        index=pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
    )
    dense_idx = pd.date_range("2020-01-01", "2022-01-01", periods=200)
    dense = pd.Series(np.exp(np.linspace(0, np.log(4), 200)), index=dense_idx)

    assert m.annualized_return(coarse) == pytest.approx(4 ** (365.25 / 731) - 1, rel=1e-12)
    assert m.annualized_return(coarse) == pytest.approx(1.0, abs=2e-3)
    assert m.annualized_return(dense) == pytest.approx(m.annualized_return(coarse), rel=1e-9)


def test_max_drawdown_finds_peak_and_trough():
    idx = pd.bdate_range("2020-01-01", periods=6)
    nav = pd.Series([1.0, 1.5, 1.2, 0.9, 1.1, 2.0], index=idx)
    mdd, peak, trough = m.max_drawdown(nav)
    assert mdd == pytest.approx(0.4)          # 1.5 → 0.9
    assert peak == idx[1]
    assert trough == idx[3]


def test_zero_volatility_series_has_no_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=10)
    nav = pd.Series(np.ones(10), index=idx)
    mdd, _, _ = m.max_drawdown(nav)
    assert mdd == pytest.approx(0.0)


def test_performance_end_to_end_matches_component_formulas():
    """一个跑赢平坦基准的策略：汇总出来的各行必须与单个公式算出来的一致。"""
    idx = pd.bdate_range("2015-01-01", periods=504)
    nav = pd.Series(1.0008 ** np.arange(504), index=idx)
    bench = pd.Series(1.0002 ** np.arange(504), index=idx)

    perf = m.performance(nav, bench, rf=RF)
    per_year = 503 * 365.25 / (idx[-1] - idx[0]).days      # 自然日口径

    assert perf["策略年化收益率"] == pytest.approx(1.0008 ** per_year - 1, rel=1e-6)
    assert perf["基准年化收益率"] == pytest.approx(1.0002 ** per_year - 1, rel=1e-6)
    assert perf["超额年化收益率"] == pytest.approx(
        perf["策略年化收益率"] - perf["基准年化收益率"], rel=1e-9
    )
    # 单调上升的序列处处无回撤，所以卡玛是「未定义」而不是「无穷大」。
    assert perf["策略最大回撤"] == pytest.approx(0.0)
    assert np.isnan(perf["策略卡玛比率"])


def test_performance_table_has_a_column_per_year_plus_full_period():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    rng = np.random.default_rng(0)
    nav = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, len(idx))), index=idx)
    bench = pd.Series(np.cumprod(1 + rng.normal(0.0002, 0.008, len(idx))), index=idx)

    table = m.performance_table(nav, bench)

    assert list(table.columns) == ["2019", "2020", "2021", "全时段"]
    assert list(table.index) == m.ROWS
