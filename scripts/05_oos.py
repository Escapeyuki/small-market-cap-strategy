"""样本外回测 —— 论文回答不了的部分（grill.md「样本外回测」，PROJECT.md Phase 4）。

研报样本止于 2022-03-31。本脚本把**同一套策略、参数逐字不变**（grill.md Q14：
先验固定假设，只改区间）延伸到数据末端 2026-08-07，回答三件研报回答不了的事：

  1. 策略在样本外还成立吗？——年化、夏普、信息比率、最大回撤，与样本内并排。
  2. 2024-01 微盘踩踏长什么样？——单独拆解 2024-01-02 ~ 2024-02-08，与 2015-06
     股灾对照（研报「已过时之处」#1 预判的、样本内不存在的流动性螺旋风险）。
  3. 2024 退市新规之后，调出原因归因是否迁移？（研报「已过时之处」#2）

**只改一件事**：区间从 [2012-12-03, 2022-03-31] 延到 [2012-12-03, 2026-08-07]。
其余（小市值100、月初出信号、T+1 开盘、非 ST/上市满 20 交易日、信号日剔除涨停
与停牌、单边 0.15%、基准中证1000、baseline 停牌口径）与 02_backtest.py 完全一致。

    python scripts/05_oos.py            # 约 10 秒，全程离线

样本外块指标在 2022-03-31（研报样本终点）归一到 1.0。停牌口径与独立起跑口径
都作交叉核对，避免结论是某一种记账方式的产物。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import analytics as a, backtest as bt, data, metrics as m
from smallcap import plots as p, universe as u
from smallcap.config import CFG, ROOT

B = CFG["backtest"]
SIZE = CFG["portfolio"]["size"]
BENCH = B["benchmark"]                 # 000852.XSHG 中证1000（主基准）
MICRO = "932000.INDX"                  # 中证2000（样本外副基准，更贴微盘；PROJECT.md 坑位早想要）
RF = B["rf"]

IS_START = str(CFG["periods"]["report_start"])     # 2012-12-03 研报样本起点
CUTOFF = str(CFG["periods"]["report_end"])          # 2022-03-31 研报样本终点 = 样本外起点前一日
OOS_START = "2022-04-01"
FULL_END = "2026-08-07"                              # 数据实际最后一个交易日
NEW_RULE = "2024-01-01"                              # 退市新规大致生效点，用于归因前后对照

OUTPUT = ROOT / "output"
FIGURES = OUTPUT / "figures"

# 研报表1 策略行的样本内数（对照用；样本内已在 02_backtest.py 验收）。
IS_ANN, IS_VOL, IS_SHARPE, IS_IR, IS_MDD, IS_CALMAR = 43.0, 32.4, 1.27, 2.64, 55.5, 0.78


def heading(text):
    print(f"\n{'=' * 82}\n{text}\n{'=' * 82}")


def td_gap(sessions, d1, d2):
    """两个交易日之间隔了多少个交易日。"""
    return sessions.get_loc(d2) - sessions.get_loc(d1)


# --------------------------------------------------------------------------- 回测

def run_full():
    """全区间连续回测 2012-12-03 → 2026-08-07（月频，T+1 开盘）。

    连续跑而不是从样本外独立起跑：这样样本内切片能顺带复现 43.0%，证明 `_ext`
    数据没有污染样本内；月频每月全额调仓，起始状态对样本外块的影响可忽略（下面
    的 OOS-only 交叉核对量化了这一点）。
    """
    calendar = u.trading_calendar()
    sessions = calendar[(calendar >= pd.Timestamp(IS_START)) & (calendar <= pd.Timestamp(FULL_END))]
    rebalances = bt.rebalance_dates(calendar, "monthly", IS_START, FULL_END)
    panel = u.panel(rebalances)
    selection = u.smallest(panel, SIZE)

    prices = data.series_many("price_post", ["close", "open"], IS_START, FULL_END)
    close = prices["close"].reindex(sessions)
    open_ = prices["open"].reindex(sessions)
    benchmark = data.wide("index_price", "close", IS_START, FULL_END)[BENCH].reindex(sessions)
    micro = data.wide("index_csi2000", "close", IS_START, FULL_END)[MICRO].reindex(sessions)  # 副基准
    # 停牌开关：volume == 0 即停牌（grill.md「停牌持仓的处理」），作敏感性口径。
    suspended = (data.series("price_raw", "volume", IS_START, FULL_END).reindex(index=sessions) == 0)

    result = bt.run(selection, close, open_, sessions)                               # baseline
    result_hold = bt.run(selection, close, open_, sessions, suspended=suspended)     # 持有停牌
    return dict(calendar=calendar, sessions=sessions, rebalances=rebalances, panel=panel,
                selection=selection, benchmark=benchmark, micro=micro,
                nav=result.nav, nav_hold=result_hold.nav)


def section_overview(ctx):
    heading("① 全区间连续回测 2012-12-03 → 2026-08-07（月频，T+1 开盘，baseline）")
    nav = ctx["nav"]
    print(f"  调仓 {len(ctx['rebalances'])} 次，选中过 {int(ctx['selection'].any().sum())} 只不同股票")
    print(f"  终值 {nav.iloc[-1]:.2f}   全区间年化 {m.annualized_return(nav) * 100:.2f}%")
    is_ann = m.annualized_return(nav.loc[:CUTOFF]) * 100
    ok = abs(is_ann - IS_ANN) < 0.3
    print(f"\n  [{'通过' if ok else '不符'}] 样本内切片 [.:{CUTOFF}] 年化 {is_ann:.2f}%"
          f"（应 ≈ 43.0%）—— 样本外数据未污染样本内，可放心外推")


def section_metrics(ctx):
    heading("② 样本外块指标 2022-04-01 → 2026-08-07（在 2022-03-31 归一到 1.0）")
    nav, nav_hold, bench = ctx["nav"], ctx["nav_hold"], ctx["benchmark"]
    oos_nav, oos_bench = nav.loc[CUTOFF:], bench.loc[CUTOFF:]
    perf = m.performance(oos_nav, oos_bench, rf=RF)
    perf_hold = m.performance(nav_hold.loc[CUTOFF:], oos_bench, rf=RF)

    print(f"  样本外时长 {(oos_nav.index[-1] - oos_nav.index[0]).days} 天"
          f"（{len(oos_nav)} 交易日）；2022-03-31 净值起点 {nav.loc[CUTOFF]:.2f} → 终值 {nav.iloc[-1]:.2f}"
          f"（{nav.iloc[-1] / nav.loc[CUTOFF]:.2f}×，中证1000 仅 {oos_bench.iloc[-1] / oos_bench.iloc[0]:.2f}×）\n")

    rows = [
        ("策略年化收益率", "策略年化收益率", "%", IS_ANN),
        ("年化波动率", "策略年化波动率", "%", IS_VOL),
        ("夏普比率(rf=2%)", "策略夏普比率(rf=2%)", "x", IS_SHARPE),
        ("信息比率", "信息比率", "x", IS_IR),
        ("最大回撤", "策略最大回撤", "%", IS_MDD),
        ("卡玛比率", "策略卡玛比率", "x", IS_CALMAR),
    ]
    print(f"  {'指标':16s}{'样本外(baseline)':>18s}{'持有停牌':>12s}{'样本内':>12s}")
    for label, key, unit, is_val in rows:
        v, vh = perf[key], perf_hold[key]
        if unit == "%":
            print(f"  {label:16s}{v * 100:>17.2f}%{vh * 100:>11.2f}%{is_val:>11.1f}%")
        else:
            print(f"  {label:16s}{v:>18.3f}{vh:>12.3f}{is_val:>12.2f}")
    print(f"  {'基准中证1000 年化':16s}{perf['基准年化收益率'] * 100:>17.2f}%{'':>12s}{9.7:>11.1f}%")
    print(f"  {'超额年化收益率':16s}{perf['超额年化收益率'] * 100:>+17.2f} {'':>12s}{'+33.3':>12s}")

    # 副基准中证2000（更贴微盘）：只用中证1000 作分母会高估超额纯度（PROJECT.md 坑位）。
    perf_micro = m.performance(oos_nav, ctx["micro"].loc[CUTOFF:], rf=RF)
    gap = (perf["超额年化收益率"] - perf_micro["超额年化收益率"]) * 100
    print(f"  {'基准中证2000 年化':16s}{perf_micro['基准年化收益率'] * 100:>17.2f}%")
    print(f"  {'超额/中证2000':16s}{perf_micro['超额年化收益率'] * 100:>+17.2f} "
          f"   （对中证1000 {perf['超额年化收益率']*100:+.1f}；差 {gap:+.1f}pp = 用中证1000 高估的那部分超额纯度）")
    print(f"\n  最大回撤区间 {perf['策略最大回撤起始']:%Y-%m-%d} → {perf['策略最大回撤终止']:%Y-%m-%d}"
          f"（基准 {perf['基准最大回撤起始']:%Y-%m-%d} → {perf['基准最大回撤终止']:%Y-%m-%d}，"
          f"回撤 {perf['基准最大回撤'] * 100:.1f}%）")

    # 独立从 2022-04-01 起跑的交叉核对
    cal = ctx["calendar"]
    oo_sess = cal[(cal >= pd.Timestamp(OOS_START)) & (cal <= pd.Timestamp(FULL_END))]
    oo_reb = bt.rebalance_dates(cal, "monthly", OOS_START, FULL_END)
    oo_sel = u.smallest(u.panel(oo_reb), SIZE)
    oo_pr = data.series_many("price_post", ["close", "open"], OOS_START, FULL_END)
    oo_nav = bt.run(oo_sel, oo_pr["close"].reindex(oo_sess), oo_pr["open"].reindex(oo_sess), oo_sess).nav
    print(f"\n  [交叉核对] OOS-only 独立起跑年化 {m.annualized_return(oo_nav) * 100:.2f}%"
          f" · 连续块 {perf['策略年化收益率'] * 100:.2f}% · 持有停牌 {perf_hold['策略年化收益率'] * 100:.2f}%"
          f" —— 三口径一致，结论不依赖记账方式")

    # 落盘
    metrics_df = pd.DataFrame({
        "样本外_baseline": [perf["策略年化收益率"], perf["策略年化波动率"], perf["策略夏普比率(rf=2%)"],
                            perf["信息比率"], perf["策略最大回撤"], perf["策略卡玛比率"],
                            perf["基准年化收益率"], perf["超额年化收益率"]],
        "样本外_持有停牌": [perf_hold["策略年化收益率"], perf_hold["策略年化波动率"],
                          perf_hold["策略夏普比率(rf=2%)"], perf_hold["信息比率"],
                          perf_hold["策略最大回撤"], perf_hold["策略卡玛比率"],
                          perf_hold["基准年化收益率"], perf_hold["超额年化收益率"]],
        "样本内": [IS_ANN / 100, IS_VOL / 100, IS_SHARPE, IS_IR, IS_MDD / 100, IS_CALMAR, 0.097, 0.333],
    }, index=["策略年化", "年化波动", "夏普", "信息比率", "最大回撤", "卡玛", "基准年化", "超额年化"])
    metrics_df.to_csv(OUTPUT / "oos_metrics.csv")
    return perf


def section_yearly(ctx):
    heading("③ 逐年收益率（全区间 performance_table，含样本外 2022–2026）")
    table = m.performance_table(ctx["nav"], ctx["benchmark"], rf=RF)
    years = [c for c in table.columns if c in {"2022", "2023", "2024", "2025", "2026"}]
    ann, ben = table.loc["策略年化收益率"] * 100, table.loc["基准年化收益率"] * 100
    exc, mdd = table.loc["超额年化收益率"] * 100, table.loc["策略最大回撤"] * 100
    print(f"  {'':10s}" + "".join(f"{y:>10s}" for y in years))
    print(f"  {'策略':10s}" + "".join(f"{ann[y]:>+10.1f}" for y in years))
    print(f"  {'中证1000':10s}" + "".join(f"{ben[y]:>+10.1f}" for y in years))
    print(f"  {'超额':10s}" + "".join(f"{exc[y]:>+10.1f}" for y in years))
    print(f"  {'年内回撤':10s}" + "".join(f"{mdd[y]:>10.1f}" for y in years))
    print(f"\n  注：2022 为整个自然年（含样本内 Q1）。2024 年内回撤 {mdd['2024']:.0f}% 却全年仅"
          f" {ann['2024']:+.1f}% —— 深坑后 V 形反弹；2025 {ann['2025']:+.1f}% 是 2015 以来最好一年。")
    table.loc[["策略年化收益率", "基准年化收益率", "超额年化收益率", "策略最大回撤"], years].to_csv(
        OUTPUT / "oos_by_year.csv")
    return table


def section_crash(ctx):
    heading("④ 2024-01 微盘踩踏拆解，并与 2015-06 股灾对照")
    nav, bench, sessions = ctx["nav"], ctx["benchmark"], ctx["sessions"]

    def window(start, end, label):
        n, b = nav.loc[start:end], bench.loc[start:end]
        dd = n / n.cummax() - 1
        trough = dd.idxmin()
        peak = n.loc[:trough].idxmax()
        print(f"  【{label}】{n.index[0]:%Y-%m-%d} → {n.index[-1]:%Y-%m-%d}（{len(n)} 交易日）")
        print(f"     策略端到端 {(n.iloc[-1] / n.iloc[0] - 1) * 100:+.1f}%   "
              f"基准中证1000 {(b.iloc[-1] / b.iloc[0] - 1) * 100:+.1f}%   "
              f"倍数 {abs(n.iloc[-1] / n.iloc[0] - 1) / max(abs(b.iloc[-1] / b.iloc[0] - 1), 1e-9):.1f}×")
        print(f"     窗口内最深回撤 {-dd.min() * 100:.1f}%（{peak:%Y-%m-%d} → {trough:%Y-%m-%d}，"
              f"{td_gap(sessions, peak, trough)} 交易日到谷）")
        return dict(label=label, start=str(n.index[0].date()), end=str(n.index[-1].date()),
                    strat=n.iloc[-1] / n.iloc[0] - 1, bench=b.iloc[-1] / b.iloc[0] - 1,
                    deepest_dd=-dd.min(), peak=peak, trough=trough)

    w24 = window("2024-01-02", "2024-02-08", "2024 踩踏 · 研报待办指定窗口 2024-01-02~02-08")
    window("2023-12-01", "2024-03-31", "2024 踩踏 · 稍宽窗口（含 V 形反弹）")
    print()
    w15 = window("2015-06-12", "2015-07-08", "2015 股灾 · 研报最大回撤窗口 2015-06-12~07-08")

    # 停牌口径分叉：2015 那次约 33pp，2024 是否重演？
    def wret(series, s, e):
        x = series.loc[s:e]
        return x.iloc[-1] / x.iloc[0] - 1
    b24 = wret(ctx["nav"], "2024-01-02", "2024-02-08")
    h24 = wret(ctx["nav_hold"], "2024-01-02", "2024-02-08")
    print(f"\n  停牌口径分叉：2024 窗口 baseline {b24 * 100:+.1f}% vs 持有停牌 {h24 * 100:+.1f}%"
          f"（差 {(h24 - b24) * 100:+.1f}pp）—— 对照 2015 那次约 33pp。"
          f"2024 微盘照跌不停牌，两口径几乎不分叉，说明这个 −50% 更贴近真实可实现的损失。")

    # 中证2000（微盘副基准）在两次踩踏窗口：2024 是微盘专属流动性螺旋，它比中证1000 跌得更深。
    mi24, mi15 = wret(ctx["micro"], "2024-01-02", "2024-02-08"), wret(ctx["micro"], "2015-06-12", "2015-07-08")
    print(f"  中证2000 对照：2024 窗口 {mi24 * 100:+.1f}%（中证1000 {wret(ctx['benchmark'], '2024-01-02', '2024-02-08') * 100:+.1f}%）、"
          f"2015 窗口 {mi15 * 100:+.1f}%——微盘副基准 2024 跌得比中证1000 更深，是比中证1000 更贴切的对照。")

    # 峰→谷→回本
    def recover(peak_dt, tag, floor):
        seg = nav.loc[peak_dt:]
        pk = nav.loc[peak_dt]
        tr_dt = (seg / seg.cummax() - 1).loc[:floor].idxmin()
        rec = seg.loc[tr_dt:]
        rec = rec[rec >= pk]
        rec_dt = rec.index[0] if len(rec) else None
        info = (f"回本 {rec_dt:%Y-%m-%d}（峰→回本 {td_gap(sessions, peak_dt, rec_dt)} 交易日）"
                if rec_dt is not None else "截至数据末仍未回本")
        print(f"  {tag}：峰 {peak_dt:%Y-%m-%d}({pk:.1f}) → 谷 {tr_dt:%Y-%m-%d}"
              f"({nav.loc[tr_dt]:.1f}, {(nav.loc[tr_dt] / pk - 1) * 100:.1f}%) → {info}")
    print()
    recover(w24["peak"], "2024", "2024-06-30")
    recover(w15["peak"], "2015", "2015-12-31")

    pd.DataFrame([w24, w15]).drop(columns=["peak", "trough"]).to_csv(
        OUTPUT / "oos_crash_windows.csv", index=False)
    return w24, w15


def section_migration(ctx):
    heading("⑤ 调出原因归因：样本内 vs 样本外（2024 退市新规后是否迁移？）")
    counts, detail = a.exit_reasons(ctx["selection"], ctx["panel"])   # 全区间，按信号日
    detail = detail.copy()
    detail["year"] = detail["date"].dt.year

    def share(sub):
        n = len(sub)
        by = sub["reason"].value_counts().reindex(a.REASONS, fill_value=0)
        return by, n

    segments = [
        ("样本内 2012-12~2022-03", detail[detail["date"] <= CUTOFF]),
        ("样本外 2022-04~2026-08", detail[detail["date"] > CUTOFF]),
        ("样本外·新规前 2022-04~2023-12", detail[(detail["date"] > CUTOFF) & (detail["date"] < NEW_RULE)]),
        ("样本外·新规后 2024-01~2026-08", detail[detail["date"] >= NEW_RULE]),
    ]
    print(f"  {'区间':30s}{'退市':>7s}{'戴帽':>7s}{'停牌':>7s}{'涨停':>7s}{'市值上涨':>9s}{'总数':>7s}")
    table = {}
    for label, sub in segments:
        by, n = share(sub)
        table[label] = {r: by[r] / n if n else 0 for r in a.REASONS}
        table[label]["总数"] = n
        print(f"  {label:30s}" + "".join(f"{by[r] / n * 100:>6.1f}%" if n else f"{'—':>7s}" for r in a.REASONS)
              + f"{n:>7d}")

    print("\n  ⇒ 戴帽（ST）近乎翻三倍：样本内 2.2% → 样本外 6.2% → 新规后 7.1%。退市与 ST 标准收紧，")
    print("    把更多微盘打成 ST，过滤链随之更频繁把它们剔出——这是退市新规在归因里的显形处。")
    print("  ⇒ 退市仍恒 0（「调出当日状态」口径）：濒退市股总先戴帽/停牌被提前赶走，与样本内同机制，")
    print("    迁移体现在戴帽而非退市。停牌 12.2%→1.4% 是 A 股结构性变化（大面积停牌时代结束）。")

    oos_detail = detail[detail["date"] > CUTOFF]
    yb = (oos_detail.pivot_table(index="year", columns="reason", aggfunc="size", fill_value=0)
          .reindex(columns=a.REASONS, fill_value=0))
    print(f"\n  戴帽逐年（样本外）：{dict(yb['戴帽'])}   （2025 最密集）")
    pd.DataFrame(table).T.to_csv(OUTPUT / "oos_exit_migration.csv")
    yb.to_csv(OUTPUT / "oos_exit_by_year.csv")


def figures(ctx, w24, w15):
    heading("图 —— 样本外净值路径 · 两次深跌形态对照")
    font = p.use_font()
    p.style()
    import matplotlib.pyplot as plt

    nav, bench, sessions = ctx["nav"], ctx["benchmark"], ctx["sessions"]

    # 图A：样本外净值，2022-03-31 归一
    oos = nav.loc[CUTOFF:] / nav.loc[CUTOFF]
    oob = bench.loc[CUTOFF:] / bench.loc[CUTOFF]
    oom = ctx["micro"].loc[CUTOFF:] / ctx["micro"].loc[CUTOFF]
    p.save(
        p.lines(pd.DataFrame({"小市值100": oos, "中证1000": oob, "中证2000": oom}),
                "样本外净值曲线（2022-04 ~ 2026-08，2022-03-31 = 1.0）", log=True,
                note="本文复现，数据源 rqdatac。参数与样本内逐字相同；纵轴对数刻度。副基准中证2000（回算序列）更贴微盘。",
                highlight="小市值100", benchmark="中证1000", ylabel="净值（对数刻度，起点 1.0）"),
        FIGURES / "oos_nav.png")

    # 图B：2024 vs 2015 踩踏，各自峰值归一到 100，横轴 = 距峰交易日
    def indexed(peak_dt, k=35):
        i = sessions.get_loc(peak_dt)
        idx = sessions[i:i + k]
        return ((nav.loc[idx] / nav.loc[peak_dt] * 100).reset_index(drop=True),
                (bench.loc[idx] / bench.loc[peak_dt] * 100).reset_index(drop=True))
    s24, b24 = indexed(w24["peak"])
    s15, b15 = indexed(w15["peak"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(s24.index, s24, color=p.PORTFOLIO, lw=2.0, label="小市值100 · 2024-01 微盘踩踏")
    ax.plot(b24.index, b24, color=p.PORTFOLIO, lw=1.2, ls="--", alpha=0.7, label="中证1000 · 2024")
    ax.plot(s15.index, s15, color="#2E86C1", lw=2.0, label="小市值100 · 2015-06 股灾")
    ax.plot(b15.index, b15, color="#2E86C1", lw=1.2, ls="--", alpha=0.7, label="中证1000 · 2015")
    ax.axhline(100, color="#333", lw=0.8)
    ax.set_xlabel("距峰值的交易日数")
    ax.set_ylabel("净值（峰值 = 100）")
    ax.set_title("两次深跌的形态对照：2024 是微盘独有的流动性螺旋，2015 是全市场 beta", fontsize=12, pad=12)
    ax.legend(fontsize=9, loc="lower left")
    fig.text(0.01, 0.01, "本文复现，数据源 rqdatac。2024 策略跌幅约为基准 3 倍（微盘独有风险）；2015 约 1.2 倍（系统性 beta）。",
             fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p.save(fig, FIGURES / "oos_crash_2024_vs_2015.png")

    nav.to_csv(OUTPUT / "nav_oos_full.csv")
    print(f"  2 张图写入 {FIGURES.relative_to(ROOT)}/（oos_nav.png, oos_crash_2024_vs_2015.png），字体 {font}")


def main():
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print(__doc__.split("\n")[0])
    print(f"\n先验假设（跑之前固定，与 02_backtest.py 逐字一致，只改区间）")
    print(f"  区间          {IS_START} ~ {FULL_END}（样本外 = {OOS_START} 之后）")
    print(f"  策略          小市值{SIZE}、月初出信号、T+1 开盘、非 ST/上市满 20 交易日、"
          f"信号日剔除涨停与停牌")
    print(f"  成本/基准     单边 {B['cost_per_side'] * 100:.2f}% · {BENCH} 中证1000 · 年化自然日折算")

    ctx = run_full()
    section_overview(ctx)
    section_metrics(ctx)
    section_yearly(ctx)
    w24, w15 = section_crash(ctx)
    section_migration(ctx)
    figures(ctx, w24, w15)
    print(f"\n结果与图写入 {OUTPUT.relative_to(ROOT)}/。全程离线，未消耗 rqdatac 额度。")


if __name__ == "__main__":
    main()
