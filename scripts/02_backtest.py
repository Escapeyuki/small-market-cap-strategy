"""验收 —— 小市值100 月频回测，对照研报表1 / 表3 / 图7。

**先验假设在此一次列清，跑完如实报告，不为凑 43.1% 回头调参**（grill.md Q14）。
对不上时一次只动一个变量，把「哪个假设、推动了多少」记成一张映射表——那张表
比对上数字本身更值钱。

⚠️ **本脚本跑出来的不是研报那个数。** 成交已统一为 T+1 开盘（grill.md Q19）。
研报**从未交代它在什么价上成交**；本项目原按「T 日收盘成交」重建它，那是未来
函数，已删除。所以下面每一处「本文 vs 研报」的差值里都含一项口径差，不再是
纯粹的复现误差。

    python scripts/02_backtest.py            # 主口径，约 1 分钟
    python scripts/02_backtest.py --full     # 再加频率对比与市值分档

全程离线，只读 data/ 下的 Parquet，不消耗 rqdatac 额度。
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import backtest as bt, data, metrics as m, universe as u
from smallcap.config import CFG, ROOT

P, B = CFG["periods"], CFG["backtest"]
START, END = str(P["report_start"]), str(P["report_end"])
BENCH = B["benchmark"]
OUTPUT = ROOT / "output"

# 研报表1 策略行，用于并排对照（基准行已在 tests/test_benchmark.py 里对完）。
T1 = pd.DataFrame(
    {
        "2013": [60.9, 24.5, 14.3], "2014": [76.7, 22.7, 16.1],
        "2015": [267.0, 56.8, 54.7], "2016": [22.2, 36.1, 28.0],
        "2017": [-22.5, 24.6, 29.7], "2018": [-17.1, 31.4, 31.6],
        "2019": [52.7, 25.9, 15.5], "2020": [16.4, 27.1, 19.2],
        "2021": [45.1, 22.2, 18.9], "全时段": [43.1, 32.1, 54.7],
    },
    index=["年化收益率", "年化波动率", "最大回撤"],
)
T3_ANNUAL = [43.1, 33.8, 31.2, 24.1, 20.3, 16.8, 14.9, 13.9, 14.3, 9.7]


def load_market(sessions):
    prices = data.series_many("price_post", ["close", "open"], START, END)
    close = prices["close"].reindex(sessions)
    return close, prices["open"].reindex(sessions)


def compare_table1(nav, benchmark):
    table = m.performance_table(nav, benchmark, rf=B["rf"])
    rows = ["策略年化收益率", "策略年化波动率", "策略最大回撤"]
    years = [c for c in T1.columns if c in table.columns]

    print(f"\n{'':16s}" + "".join(f"{y:>10s}" for y in years))
    for row, published in zip(rows, T1.index):
        ours = [table.loc[row, y] * 100 for y in years]
        theirs = [T1.loc[published, y] for y in years]
        print(f"  {published:12s}本文" + "".join(f"{v:>10.1f}" for v in ours))
        print(f"  {'':12s}研报" + "".join(f"{v:>10.1f}" for v in theirs))
        print(f"  {'':12s}差  " + "".join(f"{a - b:>+10.1f}" for a, b in zip(ours, theirs)))
    return table


def exit_counts(selection):
    """每期被调出组合的股票只数，对应研报 4.7 节统计到的 2401 次。"""
    held = [set(row[row].index) for _, row in selection.iterrows()]
    counts = [len(a - b) for a, b in zip(held, held[1:])]
    return pd.Series(counts, index=selection.index[1:])


def structural_checks(table, result, selection):
    """研报里那些「必须成立」的定性结论，比头条数字更能说明复现对不对。

    换手率同时报两种口径。研报图25 说 2018 年后双边换手率稳定在 30%–40%，而
    它自己 4.7 节又统计出 2401 次调出（111 次调仓，合 21.6 只/期，折成价值口径
    约 43%）。两个数字互不相容，除非图25 用的是**名义口径**——进出只数除以组合
    只数。两者的差有明确机制：**被调出的正是涨上去的票**，它们的权重已经漂移到
    平均之上，所以 sum|Δw| 必然高于名义换手率。这里按名义口径与图25 对照。
    """
    full = table["全时段"]
    peak, trough = full["策略最大回撤起始"], full["策略最大回撤终止"]
    exits = exit_counts(selection)
    late = exits[exits.index >= "2018-01-01"]
    nominal = late.median() * 2 / CFG["portfolio"]["size"]
    by_value = result.turnover[result.turnover.index >= "2018-01-01"].median()

    checks = [
        ("最大回撤落在 2015-06-12 → 2015-07-08",
         f"{peak:%Y-%m-%d} → {trough:%Y-%m-%d}",
         peak == pd.Timestamp("2015-06-12") and trough == pd.Timestamp("2015-07-08")),
        ("2017 年为负",
         f"{table.loc['策略年化收益率', '2017'] * 100:.1f}%",
         table.loc["策略年化收益率", "2017"] < 0),
        ("2018 年为负",
         f"{table.loc['策略年化收益率', '2018'] * 100:.1f}%",
         table.loc["策略年化收益率", "2018"] < 0),
        ("2020 年跑输基准（研报表1 唯二的落后年份之一）",
         f"超额 {table.loc['超额年化收益率', '2020'] * 100:+.1f}%",
         table.loc["超额年化收益率", "2020"] < 0),
        ("调出组合总次数 ≈ 2401（研报 4.7 节）",
         f"{exits.sum()} 次，{exits.mean():.1f} 只/期",
         abs(exits.sum() / 2401 - 1) < 0.05),
        ("2018 年后名义双边换手率落在 30%–40%（图25）",
         f"名义 {nominal * 100:.1f}%  /  价值口径 {by_value * 100:.1f}%",
         0.30 <= nominal <= 0.40),
    ]
    print("\n结构性检查")
    for label, got, ok in checks:
        print(f"  [{'通过' if ok else '不符'}] {label:44s} {got}")
    return checks


def first_monday_rebalances(calendar, start, end):
    """每月第一个周一（遇休市顺延）—— 调仓日规则的另一个候选。

    研报表2 容量测试的起点 2019-01-07 不是当月第一个交易日（那是 01-02），
    却是当月第一个周一；主区间起点 2012-12-03 两者都满足。
    """
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    picked = []
    for period in sorted(set(sessions.to_period("M"))):
        first = period.start_time
        monday = first + pd.Timedelta(days=(7 - first.weekday()) % 7)
        later = sessions[sessions >= monday]
        if len(later):
            picked.append(later[0])
    return pd.DatetimeIndex(sorted(set(picked)))


def calendar_day_listing(panel, instruments):
    """把「上市满 20 个交易日」换成「上市满 20 个自然日」的 Panel 变体。"""
    listed = pd.to_datetime(
        instruments.set_index("order_book_id")["listed_date"], errors="coerce"
    ).reindex(panel.ids)
    elapsed = (panel.dates.values[:, None] - listed.values[None, :])
    days = elapsed.astype("timedelta64[D]").astype(float) + 1
    return replace(panel, listed_days=pd.DataFrame(days, index=panel.dates, columns=panel.ids))


def sensitivity(calendar, sessions, close, open_, base_panel, base_annual):
    """Q14 的交付物：一次只动一个变量，记录它把结果推动了多少。

    这张表比对上 43.1% 本身更有价值——它说明这个复现有多脆弱，以及研报没写清
    的哪些口径其实无关紧要。
    """
    size = CFG["portfolio"]["size"]

    def annual(selection, cost=B["cost_per_side"], suspended=None):
        result = bt.run(selection, close, open_, sessions, cost_per_side=cost,
                        suspended=suspended)
        return m.annualized_return(result.nav)

    instruments = data.load("instruments")
    monday_dates = first_monday_rebalances(calendar, START, END)
    no_limit = [p for p in u.BUYABLE if p is not u.not_limit_up]
    no_st = [p for p in u.BUYABLE if p is not u.not_st]
    # 停牌持仓改为「持有到复牌」而非按冻结价卖出（volume == 0 即停牌，见 grill.md）。
    suspended = (data.series("price_raw", "volume", START, END).reindex(index=sessions) == 0)

    # 「成交改为 T+1 开盘」这一行没了——它现在就是基准本身。那次实测的结论
    # （月频只值 −0.09pp）留在 grill.md，代码删掉不影响它成立。
    variants = [
        ("调仓日改为每月第一个周一",
         lambda: annual(u.smallest(u.panel(monday_dates), size))),
        ("次新改为上市满 20 个自然日",
         lambda: annual(u.smallest(calendar_day_listing(base_panel, instruments), size))),
        ("不剔除调仓日涨停股",
         lambda: annual(u.smallest(base_panel, size, predicates=no_limit))),
        ("不剔除 ST 股",
         lambda: annual(u.smallest(base_panel, size, predicates=no_st))),
        ("持有停牌持仓（不按冻结价卖出）",
         lambda: annual(u.smallest(base_panel, size), suspended=suspended)),
        ("手续费改为 0",
         lambda: annual(u.smallest(base_panel, size), cost=0.0)),
    ]

    print("\n" + "=" * 78)
    print("Q14 单变量敏感性 —— 每行只改一个假设，其余保持基准")
    print("=" * 78)
    print(f"  {'基准（研报口径）':32s} 年化 {base_annual * 100:7.2f}%")
    rows = {}
    for label, compute in variants:
        got = compute()
        rows[label] = got
        print(f"  {label:32s} 年化 {got * 100:7.2f}%   {(got - base_annual) * 100:+6.2f}pp")
    pd.Series(rows).to_csv(OUTPUT / "sensitivity.csv")
    return rows


def main(full):
    OUTPUT.mkdir(exist_ok=True)
    calendar = u.trading_calendar()
    sessions = calendar[(calendar >= pd.Timestamp(START)) & (calendar <= pd.Timestamp(END))]
    benchmark = data.wide("index_price", "close", START, END)[BENCH].reindex(sessions)

    print(__doc__.split("\n")[0])
    print(f"\n先验假设（跑之前固定）")
    print(f"  区间          {START} ~ {END}（{len(sessions)} 个交易日）")
    print(f"  调仓          每月第一个交易日出信号")
    print(f"  成交          **T+1 开盘**（grill.md Q19）—— 研报未交代成交价，"
          f"原重建的 T 日收盘属未来函数，已删")
    print(f"  排序因子      market_cap 总市值，升序取最小 {CFG['portfolio']['size']} 只，等权")
    print(f"  选股范围      非 ST、上市满 {CFG['universe']['min_listed_days']} 个交易日；"
          f"信号日剔除涨停与停牌")
    print(f"  成本          单边 {B['cost_per_side'] * 100:.2f}%，按 sum|Δw| 计")
    print(f"  基准          {BENCH}   年化口径 自然日折算")

    rebalances = bt.rebalance_dates(calendar, "monthly", START, END)
    panel = u.panel(rebalances)
    selection = u.smallest(panel, CFG["portfolio"]["size"])
    close, open_ = load_market(sessions)

    print(f"\n  调仓 {len(rebalances)} 次，"
          f"每期合格股票 {u.eligible(panel).sum(axis=1).median():.0f} 只（中位数），"
          f"选中过 {int(selection.any().sum())} 只不同股票")

    main_result = bt.run(selection, close, open_, sessions)
    print(f"  年化 {m.annualized_return(main_result.nav) * 100:.2f}%   "
          f"终值 {main_result.nav.iloc[-1]:.2f}")

    print("\n" + "=" * 78)
    print("表1 对照 —— 小市值100 月频，T+1 开盘成交    单位：%")
    print("=" * 78)
    print("研报没说它在什么价上成交（已查证）。差值里含一项口径差，不是纯复现误差。")
    table = compare_table1(main_result.nav, benchmark)
    structural_checks(table, main_result, selection)

    table.to_csv(OUTPUT / "table1_monthly.csv")
    main_result.nav.to_csv(OUTPUT / "nav_monthly.csv")
    main_result.turnover.to_csv(OUTPUT / "turnover_monthly.csv")
    benchmark.to_csv(OUTPUT / "nav_benchmark.csv")

    if not full:
        print(f"\n结果写入 {OUTPUT.relative_to(ROOT)}/。加 --full 跑频率对比与市值分档。")
        return

    print("\n" + "=" * 78)
    print("图7 —— 调仓频率对比（研报：周月差异很小，日频显著跑输）")
    print("=" * 78)
    print("这三行同时是「研报用了 T 日收盘」这个推断的证据：研报的日频结论只在那个")
    print("口径下复现得出来，换成 T+1 开盘后方向相反。详见 grill.md Q19。")
    for freq in B["frequencies"]:
        dates = bt.rebalance_dates(calendar, freq, START, END)
        picked = u.smallest(u.panel(dates), CFG["portfolio"]["size"])
        result = bt.run(picked, close, open_, sessions)
        annual = m.annualized_return(result.nav)
        print(f"  {freq:8s} 调仓 {len(dates):>4} 次   年化 {annual * 100:7.2f}%   "
              f"终值 {result.nav.iloc[-1]:7.2f}   双边换手率中位数 "
              f"{result.turnover.median() * 100:5.1f}%")
        result.nav.to_csv(OUTPUT / f"nav_{freq}.csv")

    print("\n" + "=" * 78)
    print("表3 / 图30 —— 市值分档（研报：单调递减，43.1% → 9.7%）")
    print("=" * 78)
    size = CFG["portfolio"]["size"]
    bands = {}
    for i, band in enumerate(CFG["portfolio"]["bands"]):
        picked = u.smallest(panel, size, skip=band - size)
        result = bt.run(picked, close, open_, sessions)
        annual = m.annualized_return(result.nav)
        bands[band] = annual
        print(f"  小市值{band:<5d} 年化 {annual * 100:7.2f}%   研报 {T3_ANNUAL[i]:6.1f}%   "
              f"差 {annual * 100 - T3_ANNUAL[i]:+6.1f}")
    # 研报只说「几乎单调递减」，它自己的表3 在 800→900 处也是往上翘的
    # （13.9% → 14.3%）。所以这里比的是**形状**：唯一的例外该出现在同一处。
    ours = list(bands.values())
    breaks = [CFG["portfolio"]["bands"][i] for i in range(len(ours) - 1) if ours[i] < ours[i + 1]]
    published = [CFG["portfolio"]["bands"][i]
                 for i in range(len(T3_ANNUAL) - 1) if T3_ANNUAL[i] < T3_ANNUAL[i + 1]]
    print(f"\n  非单调点  本文 {breaks}   研报 {published}   "
          f"{'同处' if breaks == published else '不一致'}")
    print(f"  两端      本文 {ours[0] * 100:.1f}% → {ours[-1] * 100:.1f}%   "
          f"研报 {T3_ANNUAL[0]}% → {T3_ANNUAL[-1]}%")
    pd.Series(bands).to_csv(OUTPUT / "bands_annual_return.csv")
    sensitivity(calendar, sessions, close, open_, panel,
                m.annualized_return(main_result.nav))
    print(f"\n结果写入 {OUTPUT.relative_to(ROOT)}/。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="加跑频率对比、市值分档与单变量敏感性")
    main(ap.parse_args().full)
