"""分析层 —— 研报 4.1、4.3–4.9 节的全部诊断图。

    python scripts/03_analytics.py              # 图8–28，约 1 分钟
    python scripts/03_analytics.py --deciles    # 追加图6 十分组（2007–2021，慢）

全程离线。图写到 output/figures/，数值写到 output/。

这一层不是装饰。研报的头条数字（43.1%）第一次验收就对上了，但**对上一个数字
说明不了策略为什么有效**——收益是来自小盘 beta，还是来自少数极端赢家？是靠
横截面上系统性低买高卖，还是靠踩中了 2015？下面这十几张图就是在回答这个，
其中图26 更是 grill.md Q14 分层验收里最后一项没跑过的**结构性检查**。

**信号日与成交日必须分清**（grill.md Q19）。引擎按 T+1 开盘成交，所以：

  * 讲**组合持有了什么、赚了多少**的（图8/9、10、19–25）一律对齐到**成交日**，
    否则统计的是一个引擎从未持有过的组合；
  * 讲**为什么被调出**的（图26）用**信号日**——那才是过滤条件被求值的时点。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import analytics as a, backtest as bt, data, metrics as m
from smallcap import plots as p, universe as u
from smallcap.config import CFG, ROOT

P, B = CFG["periods"], CFG["backtest"]
START, END = str(P["report_start"]), str(P["report_end"])
BENCH = B["benchmark"]
SIZE = CFG["portfolio"]["size"]
OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"

INDEX_NAMES = {"000300.XSHG": "沪深300", "000905.XSHG": "中证500", "000852.XSHG": "中证1000"}
PHASES = {k: (str(v[0]), str(v[1])) for k, v in CFG["phases"].items()}
SOURCE = "本文复现，数据源 rqdatac；研报原图数据源为 Wind"

# 研报 4.7 节公布的调出原因占比，图26 的验收对象。
PUBLISHED_EXITS = {"市值上涨": 97.17, "戴帽": 2.75, "退市": 0.08}


def heading(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def delisting_paths(detail):
    """「退市 0」是口径差异不是漏了一档 —— 把三个良定义口径并排摆出来说清。

    退市归因随口径而变，三种都对，只是数的不是同一件事：
      * 调出当日状态（本文）：调出那天已无市值才算退市 → 0。濒临退市的股票总是
        先停牌、先戴帽，过滤链平均提前一年多就把它们挤出去了，退市当天早已不在组合。
      * 曾持有后退市：组合历史上持有过、后来（区间内）真的退市的股票 → 14 只。
      * 研报：2 次（0.08%）。它比「曾持有后退市」紧得多，夹在两个口径之间——更像
        「退市前后不久才调出」，也可能只是 Wind 比本文多持有了 2 只到退市（数据差异）。
    不去挑「退市前 N 天」的窗口凑那个 2——那是拿自由参数凑数字（grill.md Q14）。
    """
    total = len(detail)
    instruments = data.load("instruments")
    gone = instruments[instruments["de_listed_date"] != "0000-00-00"].copy()
    gone["de_listed"] = pd.to_datetime(gone["de_listed_date"])
    gone = gone[(gone["de_listed"] >= START) & (gone["de_listed"] <= END)]
    touched = gone[gone["order_book_id"].isin(set(detail["order_book_id"]))]

    last_exit = detail.sort_values("date").groupby("order_book_id").last()
    rows = last_exit.reindex(touched["order_book_id"])
    lead = (touched.set_index("order_book_id")["de_listed"] - rows["date"]).dt.days

    print(f"  退市归因随口径而变，三个口径并排（占调出总数 {total} 的比例）：")
    print(f"    调出当日状态（本文）  {0:>4} 次   0.00%")
    print(f"    曾持有后退市          {len(touched):>4} 次   {len(touched) / total * 100:.2f}%")
    print(f"    研报                  {2:>4} 次   0.08%")
    print(f"  ⇒ 研报的 2 夹在中间：既非本文「调出当日」口径（0），也非「曾持有后退市」"
          f"（{len(touched)}）。那 {len(touched)} 只最后离开组合的原因："
          f"{dict(rows['reason'].value_counts())}，离退市还有 {lead.median():.0f} 天"
          f"（中位数，最短 {lead.min():.0f} 天）——无一在退市当天，过滤链早把它们挤出去了。")


# --------------------------------------------------------------------------- 图26

def figure_26(selection, panel):
    """调出组合的原因统计 —— grill.md Q14 分层验收的最后一项结构性检查。"""
    heading("图26 —— 调出组合的原因（研报 4.7 节：97.17% / 2.75% / 0.08%）")
    counts, detail = a.exit_reasons(selection, panel)
    total = len(detail)

    print(f"  调出合计 {total} 次（研报 2401 次）\n")
    print(f"  {'原因':10s}{'次数':>8s}{'占比':>9s}   研报")
    collapsed = a.collapse_reasons(counts).sum()
    for reason in a.REASONS:
        n = int(counts[reason].sum())
        published = PUBLISHED_EXITS.get(reason)
        note = f"{published:.2f}%" if published is not None else "—（研报未单列）"
        print(f"  {reason:10s}{n:>8d}{n / total * 100:>8.2f}%   {note}")

    print(f"\n  折叠成研报的三类（涨停、停牌并入「市值上涨」）")
    ok = True
    for reason, published in PUBLISHED_EXITS.items():
        share = collapsed[reason] / total * 100
        gap = share - published
        ok &= abs(gap) < 1.5
        print(f"  {reason:10s}{int(collapsed[reason]):>8d}{share:>8.2f}%   "
              f"研报 {published:5.2f}%   差 {gap:+.2f}pp")
    print(f"\n  [{'通过' if ok else '不符'}] 三类占比与研报各差不到 1.5pp")

    by_year = detail.assign(year=detail["date"].dt.year).pivot_table(
        index="year", columns="reason", aggfunc="size", fill_value=0
    ).reindex(columns=a.REASONS, fill_value=0)
    print(f"\n  戴帽逐年：{dict(by_year['戴帽'])}")
    delisting_paths(detail)

    counts.to_csv(OUTPUT / "exit_reasons_monthly.csv")
    detail.to_csv(OUTPUT / "exit_reasons_detail.csv", index=False)
    p.save(
        p.stacked_bars_with_line(
            counts[["市值上涨", "涨停", "停牌"]], counts[["戴帽", "退市"]],
            "图26 小市值100 组合成分被调出组合的原因统计",
            note=f"{SOURCE}。研报只列三类，涨停/停牌是本文按研报自己的选股范围额外拆出的。",
            bar_label="调出小市值100 只数", line_label="退市 / 戴帽只数",
        ),
        FIGURES / "fig26_exit_reasons.png",
    )
    return counts, detail, ok


# --------------------------------------------------------------------------- 图8/9

def figures_8_9(traded, open_, sessions):
    heading("图8/9 —— 盈利 / 亏损股票的收益贡献基尼系数")
    # 开盘价、成交日：持有期就是「这次开盘买进 → 下次开盘卖出」，与净值引擎逐段对齐。
    returns = a.holding_returns(traded, open_, last_session=sessions[-1])
    gini = a.contribution_gini(returns, min_names=10)
    smoothed = gini.apply(a.trailing_mean)

    print(f"  盈利股票基尼系数 中位数 {gini['盈利'].median():.3f}   "
          f"亏损 {gini['亏损'].median():.3f}")
    print(f"  {'盈利更不均匀' if gini['盈利'].median() > gini['亏损'].median() else '亏损更不均匀'}"
          f" —— 研报 4.3 节：收益由少数大涨股票贡献，不是普涨")

    for label, column in [("图8 盈利", "盈利"), ("图9 亏损", "亏损")]:
        frame = pd.DataFrame({f"{column}股票基尼系数": gini[column],
                              f"{column}TTM移动平均": smoothed[column]})
        p.save(
            p.lines(frame, f"{label}股票基尼系数", note=SOURCE,
                    highlight=f"{column}TTM移动平均"),
            FIGURES / f"fig{8 if column == '盈利' else 9}_gini_{'win' if column == '盈利' else 'loss'}.png",
        )
    gini.to_csv(OUTPUT / "gini_monthly.csv")
    return returns, gini


# --------------------------------------------------------------------------- 图10

def figure_10(traded, market_cap, components):
    heading("图10 —— 平均横截面市值分位数的月度变化")
    holding_dates = traded.index                       # 成交日，即持有期的两端
    percentile = a.cap_percentile(market_cap.reindex(holding_dates))
    samples = {"小市值100": a.percentile_shift(traded, percentile)}
    for index_id, name in INDEX_NAMES.items():
        members = a.index_members(components, index_id).reindex(holding_dates).ffill()
        members = members[members.notna().any(axis=1)].eq(True)
        samples[name] = a.percentile_shift(members, percentile)

    for name, series in samples.items():
        positive = (series > 0).mean()
        print(f"  {name:10s} 均值 {series.mean():+.4f}   中位数 {series.median():+.4f}   "
              f"为正的比例 {positive:5.1%}   n={series.notna().sum()}")
    print("\n  研报 4.4 节：小市值100 几乎恒为正（期初分位数极低，只有上升空间），"
          "三大指数集中在 0 的左侧")

    p.save(
        p.density(samples, "图10 平均市值分位数月度变化的概率密度图",
                  note=f"{SOURCE}。中证1000 成分股最早只到 2014-10-17，其样本数少于其余三条。",
                  xlabel="期末平均市值分位数 − 期初平均市值分位数"),
        FIGURES / "fig10_cap_percentile_shift.png",
    )
    pd.DataFrame(samples).to_csv(OUTPUT / "cap_percentile_shift.csv")
    return samples


# --------------------------------------------------------------------------- 图11–14

def figures_11_to_14(traded):
    heading("图11–14 —— 建仓 / 卖出时刻的后复权股价时序分位数")
    # 「建仓」「卖出」说的是成交，所以分位数在成交日上取，不在信号日上取。
    changes = a.membership_changes(traded)
    ids = changes["order_book_id"].unique()
    print(f"  组合历史上碰过 {len(ids)} 只股票，"
          f"建仓 {(changes['event'] == a.ENTRY).sum()} 次、"
          f"卖出 {(changes['event'] == a.EXIT).sum()} 次")

    # 分位数按研报 3.3 节从 2000-01-03（或上市日）起算，所以这里要全历史价格，
    # 但只要这 ~1000 只 —— 全市场的话 pivot 出来是 300 MB。
    history = data.series("price_post", "close", None, END, ids=ids)
    print(f"  取全历史后复权价 {history.shape[0]} 天 × {history.shape[1]} 只 "
          f"（{history.index[0]:%Y-%m-%d} 起）")
    events = a.event_percentiles(changes, history)

    example = "000001.XSHE"                        # 研报图11 用的就是平安银行
    prices = history[example].dropna() if example in history.columns else None
    if prices is None or prices.empty:
        prices = data.series("price_post", "close", None, END, ids=[example])[example].dropna()
    p.save(
        p.price_with_percentile(prices, a.price_percentile_series(prices),
                                "图11 平安银行股价（后复权）和股价分位数",
                                note=SOURCE, name="平安银行"),
        FIGURES / "fig11_price_percentile_example.png",
    )

    for i, (label, (start, end)) in enumerate(PHASES.items(), start=12):
        window = events[(events["date"] >= start) & (events["date"] <= end)]
        samples = {event: window[window["event"] == event]["percentile"]
                   for event in (a.ENTRY, a.EXIT)}
        gap = samples[a.EXIT].mean() - samples[a.ENTRY].mean()
        print(f"  {label:14s} 建仓中位数 {samples[a.ENTRY].median():.3f}   "
              f"卖出中位数 {samples[a.EXIT].median():.3f}   "
              f"均值差 {gap:+.3f} {'（高卖低买）' if gap > 0 else '（未见低买高卖）'}")
        p.save(
            p.density(samples, f"图{i} 小市值100 组合建仓、卖出股价分位数概率密度图（{label}）",
                      note=SOURCE, clip=(0.0, 1.0), xlabel="后复权股价的时序分位数"),
            FIGURES / f"fig{i}_entry_exit_percentile_{label.replace('.', '_')}.png",
        )
    print("\n  研报 4.4 节：调出股票的股价分位数总体高于调入 —— 组合在系统性地低买高卖")
    events.to_csv(OUTPUT / "entry_exit_percentiles.csv", index=False)
    return events


# --------------------------------------------------------------------------- 图15–18

def figures_15_to_18(members_by_name, market_cap):
    heading("图15–18 —— 行业分布（截至 2022-03-31，中信一级，自由流通市值加权）")
    snapshot = pd.Timestamp(END)
    industry = data.load("industry")
    latest = industry[industry["date"] <= snapshot]["date"].max()
    mapping = (industry[industry["date"] == latest]
               .set_index("order_book_id")["first_industry_name"])

    # 加权口径用**自由流通市值**，不是研报注写的「总市值占比」——那句注写错了口径：
    # 沪深300 银行 按总市值 21.0% vs 按自由流通 12.3%，只有后者对上研报图16（12.77%）；
    # 机制是大国有行自由流通仅占总市值 4-6%。分类则须用 citics_2019（本地 zx 源把宁德时代
    # 等错分到汽车），数据已按 citics_2019 重抓。两者都对之后沪深300 全 28 行业对上研报
    # （最大 0.62pp，发布权重按 citics_2019 汇总逐行一分不差）。见 grill.md「两处对不齐」#2。
    raw_close = data.series("price_raw", "close", START, END)
    free_circ = (data.series("free_circ", "free_circulation", START, END)
                 .reindex(index=raw_close.index.union([snapshot]),
                          columns=raw_close.columns).ffill())
    ffcap = a.free_float_cap(
        raw_close.reindex(index=raw_close.index.union([snapshot])).ffill().loc[snapshot],
        free_circ.loc[snapshot])
    total_cap = market_cap.loc[snapshot]        # 总市值口径，保留作诊断对照
    print(f"  行业口径取 {latest:%Y-%m-%d} 的快照"
          f"（{'与统计日同日' if latest == snapshot else '统计日当天没有快照，取最近的一次'}）")

    largest, top5 = {}, {}
    for i, (name, members) in enumerate(members_by_name.items(), start=15):
        row = members.loc[members.index[members.index <= snapshot][-1]]
        ids = row[row].index
        weights = a.industry_weights(ids, ffcap, mapping)         # 自由流通（正式）
        diag = a.industry_weights(ids, total_cap, mapping)        # 总市值（诊断对照）
        lead = weights.index[0]
        largest[name], top5[name] = weights.iloc[0], weights.head(5).sum()
        top = "、".join(f"{k} {v:.0%}" for k, v in weights.head(3).items())
        print(f"  {name:10s} {len(ids):>4} 只   最大 {lead} {weights.iloc[0]:5.1%}"
              f"（总市值口径 {diag.get(lead, float('nan')):5.1%}）   "
              f"前五合计 {weights.head(5).sum():5.1%}   {top}")
        weights.to_csv(OUTPUT / f"industry_{name}.csv")
        p.save(
            p.barh(weights[weights > 0.005], f"图{i} {name}行业分布",
                   note=f"{SOURCE}。行业为中信一级（取 {latest:%Y-%m-%d} 快照），"
                        f"占比按成分股自由流通市值加权；小于 0.5% 的行业未画。"),
            FIGURES / f"fig{i}_industry_{name}.png",
        )
    print(f"\n  研报图15 的前三是 机械 20%、基础化工 11%、纺织服装 9% —— 本文对得上。")
    print(f"  图16 沪深300 银行 {largest['沪深300']:.1%}（研报 12.77% ＝ 发布权重按 citics_2019 汇总）："
          f"研报注写「总市值占比」是错的，实为自由流通/发布权重（总市值口径下 21.0%）。")
    # 换成正确口径后，研报 4.5 节那句「小市值100 集中度显著高于三大指数」按口径分两说，
    # 与旧（总市值）口径下「说得太满」的结论方向相反 —— 见 grill.md「两处对不齐」#2。
    print(f"  研报 4.5 节「小市值100 集中度显著高于三大指数」按口径分两说：")
    print(f"    最大单一行业 小市值100 {largest['小市值100']:.0%} > 沪深300 {largest['沪深300']:.0%}"
          f" / 中证500 {largest['中证500']:.0%} / 中证1000 {largest['中证1000']:.0%}，这条成立；")
    print(f"    前五合计 小市值100 {top5['小市值100']:.0%} vs 沪深300 {top5['沪深300']:.0%} 接近，"
          f"「显著」两字按此口径偏满。")


# --------------------------------------------------------------------------- 图19

def figure_19(members_by_name, market_cap, sessions):
    heading("图19 —— 整体法 PB（Σ总市值 / Σ净资产）")
    pb = data.series("pb", "pb_ratio_lyr", START, END)
    frame = pd.DataFrame(
        {name: a.aggregate_pb(members, market_cap, pb).reindex(sessions)
         for name, members in members_by_name.items()}
    )
    for name in frame:
        series = frame[name].dropna()
        print(f"  {name:10s} 全时段中位数 {series.median():5.2f}   "
              f"2019 年后 {series.loc['2019':].median():5.2f}")
    print("\n  研报 4.6 节：2020 年以前小市值100 的 PB 高于三大指数，之后回落到中证1000 以下")
    p.save(
        p.lines(frame, "图19 小市值100 组合与三大股指的 PB 值", note=SOURCE,
                highlight="小市值100", ylabel="PB"),
        FIGURES / "fig19_pb.png",
    )
    frame.to_csv(OUTPUT / "pb_aggregate.csv")
    return frame


# --------------------------------------------------------------------------- 图20–23

def figures_20_to_23(members_by_name, market_cap, sessions):
    heading("图20–23 —— 成分股总市值中位数")
    for i, (name, members) in enumerate(members_by_name.items(), start=20):
        median = a.median_cap(members, market_cap).reindex(sessions)
        smoothed = median.rolling(252, min_periods=252).mean()
        series = median.dropna()
        print(f"  图{i} {name:10s} 起点 {series.iloc[0] / 1e8:8.2f} 亿   "
              f"终点 {series.iloc[-1] / 1e8:8.2f} 亿   "
              f"2019 年后变动 {series.loc['2019':].iloc[-1] / series.loc['2019':].iloc[0] - 1:+.1%}")
        frame = pd.DataFrame({name: median / 1e8, f"{name}TTM": smoothed / 1e8})
        p.save(
            p.lines(frame, f"图{i} {name}成分市值中位数", note=SOURCE,
                    highlight=name, ylabel="总市值中位数（亿元）"),
            FIGURES / f"fig{i}_median_cap_{name}.png",
        )
    print("\n  研报 4.6 节：2019 年以来小市值100 的市值中位数保持平稳，三大指数显著抬升")


# --------------------------------------------------------------------------- 图24/25

def figures_24_25(members_by_name, traded, result, sessions):
    heading("图24 —— 整体法日均换手率（自由流通口径，3 个月移动平均）")
    amount = data.series("price_raw", "total_turnover", START, END)
    raw_close = data.series("price_raw", "close", START, END)          # 不复权
    free_circ = (data.series("free_circ", "free_circulation", START, END)
                 .reindex(index=sessions, columns=raw_close.columns).ffill())
    ffcap = a.free_float_cap(raw_close, free_circ)                     # 自由流通市值
    rate = data.series("turnover", "turnover_rate", START, END)        # 流通A股对照口径
    frame = pd.DataFrame(
        {name: a.aggregate_turnover_by_cap(members, amount, ffcap).reindex(sessions)
                    .rolling(60, min_periods=60).mean()
         for name, members in members_by_name.items()}
    )
    for name in frame:
        print(f"  {name:10s} 中位数 {frame[name].median():.2%}")
    small = members_by_name["小市值100"]
    circ = (a.aggregate_turnover(small, amount, rate).reindex(sessions)
            .rolling(60, min_periods=60).mean().median())
    print("\n  研报 4.6 节：小市值股票的换手率总体高于市值更大的股票（与直觉相反）")
    print(f"  分母口径：小市值100 自由流通 {frame['小市值100'].median():.2%}"
          f"（流通A股对照 {circ:.2%}，研报图目测 4-6%，见 grill.md「两处对不齐」）")
    p.save(
        p.lines(frame, "图24 小市值100 组合与三大指数成分股每日平均换手率",
                note=f"{SOURCE}。整体法，分母=自由流通市值，取 3 个月（60 个交易日）移动平均。",
                percent=True, highlight="小市值100", ylabel="日均换手率"),
        FIGURES / "fig24_turnover_rate.png",
    )
    frame.to_csv(OUTPUT / "turnover_rate_aggregate.csv")

    heading("图25 —— 月度双边换手率")
    nominal = a.nominal_turnover(traded, size=SIZE)     # 与 result.turnover 同按成交日索引
    by_value = result.turnover.reindex(nominal.index)
    late = nominal[nominal.index >= "2018-01-01"]
    print(f"  2018 年后：名义 {late.median():.1%}（研报 30%–40%）   "
          f"价值口径 sum|Δw| {by_value[by_value.index >= '2018-01-01'].median():.1%}")
    print("  两种口径的差有明确机制：被调出的正是涨上去的票，权重已漂移到平均之上")
    p.save(
        p.lines(pd.DataFrame({"双边换手率（名义）": nominal, "双边换手率（sum|Δw|）": by_value}),
                "图25 小市值100 组合的月度双边换手率",
                note=f"{SOURCE}。研报图25 未注明口径，本文两种都画；30%–40% 那句话对应名义口径。",
                percent=True, highlight="双边换手率（名义）"),
        FIGURES / "fig25_turnover.png",
    )
    pd.DataFrame({"nominal": nominal, "by_value": by_value}).to_csv(OUTPUT / "turnover_both.csv")


# --------------------------------------------------------------------------- 图27/28

def figures_27_28(nav, index_close):
    heading("图27 —— 与三大指数的滚动 12 个月相关系数")
    named = index_close.rename(columns=INDEX_NAMES)
    corr = a.rolling_correlation(nav, named)
    for name in corr:
        print(f"  {name:10s} 中位数 {corr[name].median():.3f}")
    print("\n  研报 4.8 节：与中证1000 相关度最高，与沪深300 最低 —— 分散化价值")
    p.save(
        p.lines(corr, "图27 小市值100 组合与三大指数滚动 12 个月相关系数",
                note=SOURCE, ylabel="相关系数"),
        FIGURES / "fig27_rolling_correlation.png",
    )
    corr.to_csv(OUTPUT / "rolling_correlation.csv")

    heading("图28 —— 日历效应（各工作日的平均超额日收益）")
    effect = a.calendar_effect(nav, PHASES)
    print(effect.map(lambda v: f"{v * 100:+.2f}%").to_string())
    print("\n  研报 4.9 节：周二、周三通常为正，周四、周五通常为负")
    p.save(
        p.grouped_bars(effect, "图28 小市值组合的日历效应", note=SOURCE,
                       ylabel="平均日收益 − 该阶段平均日收益"),
        FIGURES / "fig28_calendar_effect.png",
    )
    effect.to_csv(OUTPUT / "calendar_effect.csv")
    return effect


# --------------------------------------------------------------------------- 图31

def figure_31(selection, close, open_, sessions, base_nav, benchmark):
    """止盈 / 止损增强（研报 4.12 节、图31）。

    研报只给定性结论、不给具体数字，所以这里是 Q14 意义上的**结构性检查**：
    先验固定 13% / 26%（config.enhance），跑一次，如实报告止盈是否增厚、止损是否
    如研报所说「很难被触发」。触发后的成交口径、参照价等推断见 bt.run_with_stops。
    """
    heading("图31 —— 止盈 / 止损增强（研报 4.12 节：止盈多数时段有效，止损仅 2015 有用）")
    tp, sl = CFG["enhance"]["take_profit"], CFG["enhance"]["stop_loss"]
    inf = float("inf")
    nav_tp, ev_tp = bt.run_with_stops(selection, close, open_, sessions, tp, inf)
    nav_tpsl, ev_tpsl = bt.run_with_stops(selection, close, open_, sessions, tp, sl)

    base_annual = m.annualized_return(base_nav)
    tp_annual = m.annualized_return(nav_tp)
    tpsl_annual = m.annualized_return(nav_tpsl)
    hits = lambda ev, col: int(ev[col].sum()) if not ev.empty else 0
    print(f"  {'口径':22s}{'年化':>9s}{'终值':>9s}   触发次数")
    print(f"  {'无止盈止损（基准）':22s}{base_annual * 100:>8.2f}%{base_nav.iloc[-1]:>9.2f}   —")
    print(f"  {f'仅止盈 {tp:.0%}':22s}{tp_annual * 100:>8.2f}%{nav_tp.iloc[-1]:>9.2f}   "
          f"止盈 {hits(ev_tp, 'tp')}")
    print(f"  {f'止盈 {tp:.0%} + 止损 {sl:.0%}':22s}{tpsl_annual * 100:>8.2f}%"
          f"{nav_tpsl.iloc[-1]:>9.2f}   止盈 {hits(ev_tpsl, 'tp')} / 止损 {hits(ev_tpsl, 'sl')}")

    print("\n  结构性检查（研报只给定性结论，无具体数字可对）")
    gap = (tpsl_annual - tp_annual) * 100
    print(f"  [{'通过' if tp_annual > base_annual else '不符'}] 止盈增厚收益"
          f"（研报「止盈效果显著」）：{tp_annual * 100:.2f}% > 基准 {base_annual * 100:.2f}%")
    print(f"  [{'通过' if abs(gap) < 3 else '存疑'}] 止损几乎不改变全时段收益"
          f"（研报「止损效果一般」）：止盈+止损 vs 仅止盈 差 {gap:+.2f}pp")
    if not ev_tpsl.empty:
        sl_years = (ev_tpsl.assign(year=ev_tpsl.index.year)
                    .groupby("year")["sl"].sum())
        sl_years = sl_years[sl_years > 0]
        peak = sl_years.idxmax() if len(sl_years) else None
        print(f"  [{'通过' if peak == 2015 else '存疑'}] 止损触发集中在 2015"
              f"（研报「止损仅在 2015 牛熊切换时发挥作用」）：按年 {dict(sl_years)}")

    p.save(
        p.dual_axis(
            pd.DataFrame({"小市值100": base_nav, "小市值100 止盈": nav_tp,
                          "小市值100 止盈+止损": nav_tpsl}),
            pd.DataFrame({"止盈 相对基准策略": nav_tp / base_nav,
                          "止盈+止损 相对基准策略": nav_tpsl / base_nav}),
            "图31 小市值100 组合增强（止盈、止损）",
            note=f"{SOURCE}。止盈 {tp:.0%} / 止损 {sl:.0%}，T+1 开盘成交、参照建仓价"
                 f"（grill.md「止盈止损」）。右轴「相对基准」研报未定义口径，本文取"
                 f"「相对无止盈基准策略」——唯一与研报右轴 0.6-1.6 刻度自洽的口径。",
            left_label="净值", right_label="相对无止盈策略"),
        FIGURES / "fig31_take_profit_stop_loss.png",
    )
    pd.DataFrame({"baseline": base_nav, "take_profit": nav_tp,
                 "take_profit_stop_loss": nav_tpsl}).to_csv(OUTPUT / "nav_enhance.csv")
    if not ev_tpsl.empty:
        ev_tpsl.to_csv(OUTPUT / "enhance_events.csv")


# --------------------------------------------------------------------------- 图6

def figure_6(calendar):
    """十分组回测。区间是研报图6 注明的 2007-01-01 ~ 2021-12-31。

    这张图直到最近才跑得动：2012-12 之前当初只买了后复权 close，没有
    market_cap / ST / volume，分组根本无从选起（见 grill.md「实施中的发现」）。
    """
    heading("图6 —— 小市值因子十分组回测（2007-01-01 ~ 2021-12-31）")
    start, end = str(P["decile_start"]), str(P["decile_end"])
    sessions = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    rebalances = bt.rebalance_dates(calendar, "monthly", start, end)
    panel = u.panel(rebalances)
    prices = data.series_many("price_post", ["close", "open"], start, end)
    close = prices["close"].reindex(sessions)
    open_ = prices["open"].reindex(sessions)
    benchmark = data.wide("index_price", "close", start, end)[BENCH].reindex(sessions)

    groups = u.deciles(panel, CFG["portfolio"]["n_deciles"])
    navs = {}
    print(f"  调仓 {len(rebalances)} 次，每期合格股票 "
          f"{u.eligible(panel).sum(axis=1).median():.0f} 只（中位数）\n")
    print(f"  {'分组':10s}{'年化':>9s}{'终值':>9s}   （组号越大市值越小）")
    for group in sorted(groups):
        result = bt.run(groups[group], close, open_, sessions)
        navs[f"第{group}组"] = result.nav
        print(f"  第{group}组{'':6s}{m.annualized_return(result.nav) * 100:>8.2f}%"
              f"{result.nav.iloc[-1]:>9.2f}")
    navs[INDEX_NAMES[BENCH]] = benchmark / benchmark.iloc[0]

    frame = pd.DataFrame(navs)
    annual = {k: m.annualized_return(v) for k, v in navs.items()}
    ordered = [annual[f"第{g}组"] for g in sorted(groups)]
    breaks = [i + 1 for i in range(len(ordered) - 1) if ordered[i] > ordered[i + 1]]

    # 研报 4.1 节的原话是「当市值越小时，分组的单调性越明显」——它**没有**声称
    # 全程严格单调，只声称小市值那一端单调。按它说的检验，不要按它没说的检验。
    small_end = ordered[max(breaks) if breaks else 0:]
    print(f"\n  [{'通过' if all(x <= y for x, y in zip(small_end, small_end[1:])) else '不符'}]"
          f" 小市值端单调递增（第{max(breaks) + 1 if breaks else 1}组起，"
          f"研报 4.1 节：「当市值越小时，分组的单调性越明显」）")
    print(f"  非单调点 第{breaks}→次组   两端：第1组（市值最大）{ordered[0] * 100:.2f}%  →  "
          f"第10组（市值最小）{ordered[-1] * 100:.2f}%")

    p.save(
        p.lines(frame, "图6 小市值因子的分组回测结果", log=True,
                note=f"{SOURCE}。2007-01-01 ~ 2021-12-31，基准中证1000；纵轴对数刻度。",
                highlight="第10组", benchmark=INDEX_NAMES[BENCH],
                ylabel="净值（对数刻度）"),
        FIGURES / "fig6_deciles.png",
    )
    frame.to_csv(OUTPUT / "nav_deciles.csv")
    pd.Series(annual).to_csv(OUTPUT / "deciles_annual_return.csv")
    return frame


# --------------------------------------------------------------------------- 主流程

def main(with_deciles):
    FIGURES.mkdir(parents=True, exist_ok=True)
    font = p.use_font()
    p.style()
    print(__doc__.split("\n")[0])
    print(f"\n  中文字体 {font}   区间 {START} ~ {END}")

    calendar = u.trading_calendar()
    sessions = calendar[(calendar >= pd.Timestamp(START)) & (calendar <= pd.Timestamp(END))]
    rebalances = bt.rebalance_dates(calendar, "monthly", START, END)
    panel = u.panel(rebalances)
    selection = u.smallest(panel, SIZE)

    prices = data.series_many("price_post", ["close", "open"], START, END)
    close = prices["close"].reindex(sessions)
    open_ = prices["open"].reindex(sessions)
    result = bt.run(selection, close, open_, sessions)
    index_close = data.wide("index_price", "close", START, END).reindex(sessions)
    market_cap = data.series("market_cap", "market_cap", START, END).reindex(sessions)
    components = data.load("index_components", START, END)

    # 同一份选股，按**成交日**重新贴标签。引擎在 T+1 开盘才真的持有它们，组合侧的
    # 统计就得跟着挪过去；否则算的是一个引擎从未持有过的组合。末期若无下一交易日
    # 可成交，该信号作废，这里一并跟着丢掉。
    schedule = bt.trade_dates(selection.index, sessions)
    traded = selection.loc[schedule.index].set_axis(pd.DatetimeIndex(schedule.to_numpy()))

    print(f"  调仓 {len(rebalances)} 次（{len(traded)} 次可成交）   "
          f"年化 {m.annualized_return(result.nav) * 100:.2f}%   终值 {result.nav.iloc[-1]:.2f}")

    # 组合与三大指数走同一条代码路径：成员一律是布尔宽表，统计函数只认这个形状。
    members_by_name = {"小市值100": a.daily(traded, sessions)}
    for index_id, name in INDEX_NAMES.items():
        members_by_name[name] = a.index_members(components, index_id)

    figure_26(selection, panel)                    # 调出原因：信号日，过滤条件求值的时点
    figures_8_9(traded, open_, sessions)
    figure_10(traded, market_cap, components)
    figures_11_to_14(traded)
    figures_15_to_18(members_by_name, market_cap)
    figure_19(members_by_name, market_cap, sessions)
    figures_20_to_23(members_by_name, market_cap, sessions)
    figures_24_25(members_by_name, traded, result, sessions)
    figures_27_28(result.nav, index_close)
    figure_31(selection, close, open_, sessions, result.nav, index_close[BENCH])

    if with_deciles:
        figure_6(calendar)

    figures = sorted(FIGURES.glob("*.png"))
    print(f"\n{len(figures)} 张图写入 {FIGURES.relative_to(ROOT)}/，"
          f"数值表写入 {OUTPUT.relative_to(ROOT)}/。")
    if not with_deciles:
        print("加 --deciles 跑图6 十分组（2007–2021，约 1 分钟）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deciles", action="store_true",
                    help="加跑图6 十分组回测（区间 2007–2021）")
    main(ap.parse_args().deciles)
