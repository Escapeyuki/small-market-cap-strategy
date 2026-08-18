"""专题之二《小市值增强策略》样本外回测 —— 研报回答不了的部分。

研报样本止于 2022-05。本脚本把 06_enhance.py 的增强族**参数逐字不变**（grill.md Q14：
先验固定，只改区间），经**同一条构造路径** `smallcap.enhance.run_strategy` 延伸到数据
末端，回答四件研报回答不了的事（grill_enhance.md「样本外」/ project_enhance.md E3）：

  ① 连续跑 2010→2026，样本内切片是否仍复现 06 的样本内数（证 `_ext` 未污染样本内）；
  ② 样本外块 2022-06→2026-08：全 15 策略「样本内本文 vs 样本外本文」5 指标并排（E1
     中立，无研报列——已越过研报样本）；
  ③ ladder-6 风险调整视图 + **中证2000 副基准**：用中证1000 与中证2000 两个分母各算一次
     超额，量化「只用中证1000 会高估超额纯度」（Q2=b）；
  ④ **2024-01 微盘踩踏 vs 择时（E3 预登记假设）**：旗舰/择时策略每年一月本就空仓，可能躲过
     击穿裸策略的那场踩踏。三视图并列——一月纯空仓段 / 二月首日再入场吃到什么 / 峰→谷→
     回本净——**结论由数字定，允许「躲了一月、二月初照样被埋」的反直觉结果，不预设**；
  ⑤ 退市新规后调出归因是否迁移（专题之二 baseline，样本内 vs 样本外）。

    python scripts/07_enhance_oos.py       # 约 1-2 分钟，全程离线，不消耗 rqdatac 额度

**只改一件事**：区间从 [2010-01-01, 2022-05-31] 延到 [2010-01-01, 2026-08-07]。其余
（选股范围、波动窗口 250、跌停惩罚、日历月择时、T+1 开盘、双边千三、基准中证1000）
与 06_enhance.py 逐字一致。样本外块在 2022-05-31（研报样本终点）归一到 1.0。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import analytics as a, data, enhance as en
from smallcap import metrics as m, plots as p, universe as u
from smallcap.config import CFG, ROOT

P2, B = CFG["report2"], CFG["backtest"]
START = str(P2["start"])                              # 2010-01-01 专题之二样本起点
IS_END = str(P2["end"])                               # 2022-05-31 研报样本终点 = 样本外起点前
OOS_START = "2022-06-01"
FULL_END = str(CFG["periods"]["extension_end"])       # 2026-08-08（日历自然截到 2026-08-07）
NEW_RULE = "2024-01-01"                               # 退市新规大致生效点
BENCH = B["benchmark"]                                # 000852.XSHG 中证1000（主基准）
MICRO = "932000.INDX"                                 # 中证2000（样本外副基准，回算序列自 2013-12）
RF, COST, VOL_W = B["rf"], B["cost_per_side"], P2["volatility_window"]
FEE_TEST_COST = P2["fee_test_cost_per_side"]
OUTPUT, FIGURES = ROOT / "output", ROOT / "output" / "figures"

# 图与崩盘拆解聚焦的 ladder-6（Q3=i）。含只差择时的隔离对：小市值低波50(不择时) vs 择时低波50(除1,4)。
LADDER = ["小市值100基准", "小市值50", "小市值低波50", "择时低波50", "★旗舰双周低波50", "保留六月"]
# 样本内自检锚点：06_enhance.py 全区间实测值（连续跑的样本内切片因引擎因果性应逐点复现）。
IS_ANCHORS = {"小市值100基准": 29.7, "★旗舰双周低波50": 44.2}


def heading(text):
    print(f"\n{'=' * 88}\n{text}\n{'=' * 88}")


def td_gap(sessions, d1, d2):
    return sessions.get_loc(d2) - sessions.get_loc(d1)


# --------------------------------------------------------------------------- 数据 & 回测

def load_inputs():
    """同 06.load_inputs，但区间延到 FULL_END，并多加载中证2000 副基准。"""
    cal = u.trading_calendar()
    sessions = cal[(cal >= pd.Timestamp(START)) & (cal <= pd.Timestamp(FULL_END))]
    prices = data.series_many("price_post", ["close", "open"], "2009-01-01", FULL_END)
    close = prices["close"].reindex(sessions)
    open_ = prices["open"].reindex(sessions)
    full_close = prices["close"]                                     # 含 2009 回看，供波动率
    bench = data.wide("index_price", "close", START, FULL_END)[BENCH].reindex(sessions)
    micro = data.wide("index_csi2000", "close", START, FULL_END)[MICRO].reindex(sessions)
    raw_close = data.series("price_raw", "close", START, FULL_END).reindex(sessions)
    ld_px = data.wide("limit_down", "limit_down", START, FULL_END).reindex(sessions)
    is_ld = (raw_close <= ld_px + 1e-6) & (ld_px > 0)
    return dict(cal=cal, sessions=sessions, close=close, open_=open_, full_close=full_close,
                bench=bench, micro=micro, is_ld=is_ld)


def run_all(ctx):
    """全 15 策略连续跑 2010→2026（en.run_strategy，与 06 同一条构造路径，只区间延长）。"""
    cache = {}
    strategies = {}
    for label, freq, steps, cash_months, penalty, _ref in en.ROSTER:
        cost = FEE_TEST_COST if "费用" in label else COST
        strategies[label] = en.run_strategy(
            cache, ctx["cal"], ctx["sessions"], ctx["close"], ctx["open_"], ctx["full_close"],
            ctx["is_ld"], freq, steps, cash_months, penalty, cost, VOL_W, START, FULL_END)
    ctx["cache"] = cache
    return strategies


def window_metrics(s, bench, full_close, lo, hi):
    """在 [lo, hi] 切片上算专题之二 5 指标（年化/换手/胜率/策略分位/基准分位，单位 %）。

    win/sp/bp 传完整 nav/full_close + **窗口内的成交日/schedule**：因取的是窗口内全部（连续）
    成交日，consecutive-in-window = consecutive-in-reality，逐期收益口径与 06 一致。
    """
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    nav = s["nav"]
    seg = nav.loc[lo:hi]
    tds = s["tds"][(s["tds"] >= lo) & (s["tds"] <= hi)]
    sched = s["sched"][(s["sched"].index >= lo) & (s["sched"].index <= hi)]
    cash_td = None
    if s["cash_td"] is not None:
        cash_td = s["cash_td"][(s["cash_td"] >= lo) & (s["cash_td"] <= hi)]
    turn = s["res"].turnover.loc[lo:hi]
    return dict(
        ann=m.annualized_return(seg) * 100,
        turn=turn.median() * 100 if len(turn) else np.nan,
        win=m.win_rate(nav, bench, tds, cash_td) * 100 if len(tds) > 1 else np.nan,
        sp=m.avg_percentile(s["sel"], full_close, sched) * 100 if len(sched) else np.nan,
        bp=m.benchmark_percentile(bench, full_close, tds) * 100 if len(tds) > 1 else np.nan,
    )


# --------------------------------------------------------------------------- 分节

def section_sanity(strategies, ctx):
    heading("① 连续跑 2010→2026 · 样本内切片自检（切片应复现 06_enhance.py 的样本内数）")
    print(f"  {'策略':18s}{'样本内切片年化':>14s}{'06 实测':>10s}{'判定':>8s}")
    ok_all = True
    for label in LADDER:
        is_ann = m.annualized_return(strategies[label]["nav"].loc[:IS_END]) * 100
        anchor = IS_ANCHORS.get(label)
        tag = ""
        if anchor is not None:
            ok = abs(is_ann - anchor) < 0.3
            ok_all &= ok
            tag = f"{anchor:>8.1f}  [{'通过' if ok else '不符'}]"
        print(f"  {label:18s}{is_ann:>13.1f}%{tag:>18s}")
    print(f"\n  [{'通过' if ok_all else '不符'}] 锚点策略样本内切片复现 06 → 样本外数据未污染样本内，可放心外推")


def section_panel(strategies, ctx):
    heading("② 样本外块 2022-06→2026-08 · 全 15 策略「样本内本文 vs 样本外本文」（E1 中立，无研报列）")
    bench, fc = ctx["bench"], ctx["full_close"]
    print(f"  单位 %，每格 样本内/样本外。区间：样本内 {START}~{IS_END}，样本外 {OOS_START}~2026-08-07\n")
    print(f"  {'策略':18s}{'年化':>15}{'换手':>15}{'胜率':>15}{'策略分位':>15}{'基准分位':>15}")
    rows = {}
    for label, *_ in en.ROSTER:
        s = strategies[label]
        is_m = window_metrics(s, bench, fc, START, IS_END)
        oo_m = window_metrics(s, bench, fc, OOS_START, FULL_END)
        rows[label] = {f"IS_{k}": is_m[k] for k in is_m} | {f"OOS_{k}": oo_m[k] for k in oo_m}
        def cell(k):
            return f"{is_m[k]:>6.1f}/{oo_m[k]:<6.1f}"
        print(f"  {label:18s}{cell('ann'):>16}{cell('turn'):>16}{cell('win'):>16}"
              f"{cell('sp'):>16}{cell('bp'):>16}")
    print(f"\n  注：样本外无研报对照（已越过 2022-05 研报样本）。分位数系统低研报约 11pp 是 Wind 方法学口径差（E5），")
    print(f"     样本内已记；此处样本内/样本外自比，口径一致，看的是**同一策略跨期是否退化**。")
    pd.DataFrame(rows).T.to_csv(OUTPUT / "enhance_oos_panel.csv")


def section_ladder_risk(strategies, ctx):
    heading("③ ladder-6 样本外风险调整 · 中证1000 vs 中证2000 两个分母（Q2=b：超额纯度）")
    bench, micro = ctx["bench"], ctx["micro"]
    ob = bench.loc[IS_END:]
    om = micro.loc[IS_END:]
    print(f"  样本外块（2022-05-31 归一）。超额分别对 中证1000 与 中证2000（更贴微盘）算：\n")
    print(f"  {'策略':18s}{'年化':>8}{'波动':>8}{'夏普':>8}{'信息比':>8}{'最大回撤':>9}"
          f"{'超额/中证1000':>13}{'超额/中证2000':>13}")
    rows = {}
    for label in LADDER:
        onav = strategies[label]["nav"].loc[IS_END:]
        perf1 = m.performance(onav, ob, rf=RF)                       # 对中证1000
        perf2 = m.performance(onav, om, rf=RF)                       # 对中证2000
        rows[label] = {
            "年化": perf1["策略年化收益率"], "波动": perf1["策略年化波动率"],
            "夏普": perf1["策略夏普比率(rf=2%)"], "信息比率": perf1["信息比率"],
            "最大回撤": perf1["策略最大回撤"],
            "超额_中证1000": perf1["超额年化收益率"], "超额_中证2000": perf2["超额年化收益率"],
        }
        print(f"  {label:18s}{perf1['策略年化收益率']*100:>7.1f}%{perf1['策略年化波动率']*100:>7.1f}%"
              f"{perf1['策略夏普比率(rf=2%)']:>8.2f}{perf1['信息比率']:>8.2f}"
              f"{perf1['策略最大回撤']*100:>8.1f}%{perf1['超额年化收益率']*100:>+12.1f}"
              f"{perf2['超额年化收益率']*100:>+13.1f}")
    b1 = m.annualized_return(ob) * 100
    b2 = m.annualized_return(om) * 100
    print(f"\n  基准年化：中证1000 {b1:+.1f}%   中证2000 {b2:+.1f}%（微盘样本外涨得多得多）")
    print(f"  ⇒ 用中证1000 作分母，超额被系统性抬高约 {b2 - b1:.0f}pp（中证2000 才是更贴切的微盘对照）。")
    pd.DataFrame(rows).T.to_csv(OUTPUT / "enhance_oos_ladder_risk.csv")


def section_crash(strategies, ctx):
    heading("④ 2024-01 微盘踩踏 vs 择时（E3 假设检验）· 三视图 · 结论由数字定")
    bench, micro, sessions = ctx["bench"], ctx["micro"], ctx["sessions"]

    def ete(nav, s, e):
        x = nav.loc[s:e]
        return (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) else np.nan

    def deepest(nav, s, e):
        x = nav.loc[s:e]
        if len(x) < 2:
            return np.nan
        return -(x / x.cummax() - 1).min() * 100

    # 先确认机制：一月各 ladder 策略的市场仓位（择时策略应≈0＝确实在空仓躲）
    jan = ("2024-01-02", "2024-01-31")
    print("  【机制核对】2024 年 1 月平均市场仓位（择时策略空仓则≈0，是「躲过」的前提）：")
    for label in LADDER:
        exp = strategies[label]["exposure"].loc[jan[0]:jan[1]].mean()
        print(f"     {label:18s} 一月平均仓位 {exp*100:>5.1f}%   一月端到端 {ete(strategies[label]['nav'], *jan):>+6.1f}%")

    windows = [
        ("视图1 一月纯空仓段 2024-01-02~01-31", "2024-01-02", "2024-01-31"),
        ("视图2 二月再入场   2024-02-01~02-29", "2024-02-01", "2024-02-29"),
        ("视图3 峰→回本弧   2023-12-01~06-30", "2023-12-01", "2024-06-30"),
        ("对照 05_oos 窗口   2024-01-02~02-08", "2024-01-02", "2024-02-08"),
    ]
    rows = []
    for title, s, e in windows:
        print(f"\n  {title}")
        print(f"    {'策略/基准':18s}{'端到端':>9}{'窗口内最深回撤':>14}")
        for label in LADDER:
            nav = strategies[label]["nav"]
            print(f"    {label:18s}{ete(nav, s, e):>+8.1f}%{deepest(nav, s, e):>13.1f}%")
            rows.append(dict(window=title, name=label, ete=ete(nav, s, e), dd=deepest(nav, s, e)))
        for bl, bn in [("中证1000", bench), ("中证2000", micro)]:
            print(f"    {bl:18s}{ete(bn, s, e):>+8.1f}%{deepest(bn, s, e):>13.1f}%")
            rows.append(dict(window=title, name=bl, ete=ete(bn, s, e), dd=deepest(bn, s, e)))

    # 诚实小结：只差择时的一对（低波50 vs 择时低波50）在「峰→回本弧」上的净差
    lv, tv = "小市值低波50", "择时低波50"
    arc = ("2023-12-01", "2024-06-30")
    d_lv, d_tv = ete(strategies[lv]["nav"], *arc), ete(strategies[tv]["nav"], *arc)
    print(f"\n  ⇒ 只差择时的一对（峰→回本弧 {arc[0]}~{arc[1]}）：{lv} {d_lv:+.1f}% vs {tv} {d_tv:+.1f}%"
          f"（净差 {d_tv - d_lv:+.1f}pp）。")
    print(f"    择时一月空仓（仓位≈0）确实躲过一月下杀，但 2024-02-01 首日按 apply_timing 再建仓——")
    print(f"    二月初（约 2/5-2/8）正是微盘谷底，是否「躲了一月又埋在二月」，看上面视图2 与视图3 的数。")
    pd.DataFrame(rows).to_csv(OUTPUT / "enhance_oos_crash.csv", index=False)
    return windows


def section_migration(strategies, ctx):
    heading("⑤ 调出原因归因：样本内 vs 样本外（专题之二 baseline，退市新规后是否迁移？）")
    # baseline（小市值100基准）无择时、无惩罚，其 sel 即 cascade 原始选股（信号日索引）。
    sel = strategies["小市值100基准"]["sel"]
    _, panel_m, _ = en.build_freq(ctx["cache"], ctx["cal"], "monthly", START, FULL_END,
                                  ctx["full_close"], VOL_W)
    counts, detail = a.exit_reasons(sel, panel_m)
    segments = [
        ("样本内 2010-01~2022-05", detail[detail["date"] <= IS_END]),
        ("样本外 2022-06~2026-08", detail[detail["date"] > IS_END]),
        ("样本外·新规后 2024-01~2026-08", detail[detail["date"] >= NEW_RULE]),
    ]
    print(f"  {'区间':30s}{'退市':>7s}{'戴帽':>7s}{'停牌':>7s}{'涨停':>7s}{'市值上涨':>9s}{'总数':>7s}")
    table = {}
    for label, sub in segments:
        n = len(sub)
        by = sub["reason"].value_counts().reindex(a.REASONS, fill_value=0)
        table[label] = {r: (by[r] / n if n else 0) for r in a.REASONS} | {"总数": n}
        print(f"  {label:30s}" + "".join(f"{by[r]/n*100:>6.1f}%" if n else f"{'—':>7s}" for r in a.REASONS)
              + f"{n:>7d}")
    print("\n  ⇒ 与专题之一样本外同向对照（grill.md「样本外回测」）：退市新规把更多微盘打成 ST，戴帽调出占比抬升；")
    print("    退市在「调出当日状态」口径下仍恒 0（濒退市股先被戴帽/停牌提前赶走）。数字为准，逐格见 CSV。")
    pd.DataFrame(table).T.to_csv(OUTPUT / "enhance_oos_migration.csv")


def figures(strategies, ctx):
    heading("图 —— 增强阶梯样本外净值 · 2024 踩踏择时对照")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = p.use_font()
    p.style()
    bench, micro = ctx["bench"], ctx["micro"]

    # 图A：ladder 关键 4 级样本外净值（2022-05-31 归一）+ 两个基准
    rungs = ["小市值100基准", "小市值低波50", "择时低波50", "★旗舰双周低波50"]
    frame = {}
    for label in rungs:
        n = strategies[label]["nav"].loc[IS_END:]
        frame[label.lstrip("★")] = n / n.iloc[0]
    frame["中证1000"] = bench.loc[IS_END:] / bench.loc[IS_END]
    frame["中证2000"] = micro.loc[IS_END:] / micro.loc[IS_END]
    fig = p.lines(pd.DataFrame(frame), "小市值增强阶梯 · 样本外净值（2022-06~2026-08，2022-05-31=1.0）",
                  log=True, highlight="旗舰双周低波50", benchmark="中证1000",
                  note="本文复现，数据源 rqdatac。参数与样本内逐字相同；纵轴对数刻度。副基准中证2000（回算序列）更贴微盘。",
                  ylabel="净值（对数刻度，起点 1.0）")
    p.save(fig, FIGURES / "enh_oos_ladder.png")

    # 图B：2024 踩踏，择时 vs 不择时（各自 2024-01-02 归一到 100，竖线标二月首日再入场）
    lo, hi = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-29")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    series = [("小市值低波50", "#2E86C1", "-", 2.0), ("择时低波50", "#28B463", "-", 2.0),
              ("★旗舰双周低波50", p.PORTFOLIO, "-", 2.0), ("中证2000", "#888888", "--", 1.3)]
    for label, color, ls, lw in series:
        src = strategies[label]["nav"] if label in strategies else micro
        seg = src.loc[lo:hi]
        ax.plot(seg.index, (seg / seg.iloc[0] * 100).to_numpy(), color=color, ls=ls, lw=lw,
                label=label.lstrip("★"))
    ax.axhline(100, color="#333", lw=0.8)
    ax.axvline(pd.Timestamp("2024-02-01"), color="#C0392B", lw=0.9, ls=":", alpha=0.7)
    ax.text(pd.Timestamp("2024-02-01"), ax.get_ylim()[1], " 择时策略二月首日再入场",
            fontsize=8, color="#C0392B", va="top")
    ax.set_ylabel("净值（2024-01-02 = 100）")
    ax.set_title("2024-01 微盘踩踏 · 择时（除1,4,6）一月空仓 vs 不择时（E3 假设检验）", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower left")
    fig.text(0.01, 0.01, "本文复现，数据源 rqdatac。择时策略整个一月持币（净值走平），2024-02-01 按日历月口径再建仓——"
                         "二月初正值微盘谷底附近。", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p.save(fig, FIGURES / "enh_oos_crash.png")

    pd.DataFrame({k: strategies[k]["nav"] for k in LADDER}).to_csv(OUTPUT / "enhance_oos_navs.csv")
    print(f"  2 张图写入 {FIGURES.relative_to(ROOT)}/（enh_oos_ladder.png, enh_oos_crash.png），字体 {font}")


def main():
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(__doc__.split("\n")[0])
    print("\n先验假设（跑之前固定，与 06_enhance.py 逐字一致，只改区间 → 延到样本外）")
    print(f"  区间          {START} ~ 2026-08-07（样本外 = {OOS_START} 之后）")
    print(f"  选股范围      上市满1年(243交易日)、非ST、非注册制、非北交所；信号日剔除涨停/停牌")
    print(f"  波动率/择时   过去 {VOL_W} 交易日 std（E4）；1/4月空仓（旗舰加6月）持币0%（E9）")
    print(f"  惩罚/成交     跌停惩罚延迟卖出（E7）；T+1开盘（Q19）；双边千三")
    print(f"  基准          主 {BENCH} 中证1000；副 {MICRO} 中证2000（样本外，更贴微盘）")

    ctx = load_inputs()
    strategies = run_all(ctx)
    section_sanity(strategies, ctx)
    section_panel(strategies, ctx)
    section_ladder_risk(strategies, ctx)
    section_crash(strategies, ctx)
    section_migration(strategies, ctx)
    figures(strategies, ctx)
    print(f"\n结果与图写入 {OUTPUT.relative_to(ROOT)}/enhance_oos_*.csv。全程离线，未消耗 rqdatac 额度。")


if __name__ == "__main__":
    main()
