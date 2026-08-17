"""波动率因子，用手算得出的合成价格钉死（专题之二 E4）。

期望值都能在纸上算出来：波动率 = 过去 window 个交易日**日收益率的样本标准差**
（pandas rolling(window).std() 默认 ddof=1）。
"""
import pandas as pd
import pytest

from smallcap import factors as fa

DATES = pd.bdate_range("2020-01-01", periods=6)


def test_volatility_is_trailing_daily_return_sample_std():
    # A：日收益率 [.1, -.1, .1, -.1, .1] —— 价格 100→110→99→108.9→98.01→107.811
    a = [100.0]
    for r in [0.1, -0.1, 0.1, -0.1, 0.1]:
        a.append(a[-1] * (1 + r))
    close = pd.DataFrame({"A": a}, index=DATES)

    vol = fa.volatility(close, DATES, window=3)

    # 窗口 3 需要 3 个收益率；最早两天回看不足 → NaN
    assert vol["A"].iloc[:3].isna().all()
    # 第 4 天（index 3）：trailing 3 收益 = [.1, -.1, .1]，样本 std：
    #   mean = 1/30，方差 = Σ(r-mean)²/2 = 0.013333…，std = 0.115470…
    assert vol["A"].iloc[3] == pytest.approx(pd.Series([0.1, -0.1, 0.1]).std())
    assert vol["A"].iloc[3] == pytest.approx(0.1154700538, abs=1e-9)


def test_constant_return_has_zero_volatility():
    close = pd.DataFrame({"A": [100.0 * 1.05 ** i for i in range(6)]}, index=DATES)
    vol = fa.volatility(close, DATES, window=4)
    assert abs(vol["A"].iloc[-1]) < 1e-12          # 恒定 5% 日涨 → 收益率无波动


def test_volatility_samples_only_requested_dates_and_ids():
    close = pd.DataFrame({"A": range(1, 7), "B": range(2, 8)}, index=DATES, dtype=float)
    vol = fa.volatility(close, DATES[4:], window=2, ids=["A"])
    assert list(vol.index) == list(DATES[4:])
    assert list(vol.columns) == ["A"]
