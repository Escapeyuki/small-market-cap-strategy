"""专题之二《小市值增强策略》主驱动 —— 第 3 部分增强族 + 旗舰，样本内 2010-2022.5。

**先验假设在此一次列清，跑完如实报告，不为凑 26.7% / 50.9% 调参**（grill.md Q14，
grill_enhance.md）。口径见 grill_enhance.md 的 E 系列；两处最重要的实测发现：
  · 波动率窗口取 250 交易日（≈1年）——先验 60 太短，250 同时对上低波50/择时低波50（E4）；
  · 跌停惩罚在可执行口径下只值 −0.4pp（非研报 −2.8pp），差额是研报自身的执行口径，
    与 Q19 同源，故带惩罚策略系统性高研报约 2pp（E7）。

    python scripts/06_enhance.py           # 15 个策略 + 表19/20 + 结构性检查
    python scripts/06_enhance.py --vol-sweep  # 追加 E4 波动率窗口敏感性

全程离线，只读 data/ 下的 Parquet，不消耗 rqdatac 额度。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import backtest as bt, data, enhance as en, factors as fa, metrics as m, universe as u
from smallcap.config import CFG, ROOT

P2, B = CFG["report2"], CFG["backtest"]
START, END = str(P2["start"]), str(P2["end"])
BENCH = B["benchmark"]
COST = B["cost_per_side"]
VOL_W = P2["volatility_window"]
OUTPUT = ROOT / "output"

# 策略清单 ROSTER（17 策略构造参数 + 研报对照值）与 apply_timing 已移到
# smallcap/enhance.py，与样本外驱动 07_enhance_oos.py 共用同一份，两个脚本走**同一条
# 构造路径**（en.run_strategy），避免各写一份走岔（grill.md Q14）。本文用 en.ROSTER。


def load_inputs():
    cal = u.trading_calendar()
    sessions = cal[(cal >= pd.Timestamp(START)) & (cal <= pd.Timestamp(END))]
    prices = data.series_many("price_post", ["close", "open"], "2009-01-01", END)
    close = prices["close"].reindex(sessions)
    open_ = prices["open"].reindex(sessions)
    full_close = prices["close"]                                  # 含 2009 回看，供波动率
    bench = data.wide("index_price", "close", START, END)[BENCH].reindex(sessions)
    # 跌停标记：不复权收盘 <= 不复权跌停价（E7）
    raw_close = data.series("price_raw", "close", START, END).reindex(sessions)
    ld_px = data.wide("limit_down", "limit_down", START, END).reindex(sessions)
    is_ld = (raw_close <= ld_px + 1e-6) & (ld_px > 0)
    return cal, sessions, close, open_, full_close, bench, is_ld


def make_figures(navs, bench, exposures):
    """中立图集（E15，Q4）：增强阶梯净值 + 择时仓位暴露 + 表19/20 月度均值。
    不画带论点的口径归因图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from smallcap import plots
    plots.use_font()
    plots.style()                    # 与 03/05 一致的版式（figsize/网格），此前漏调导致图偏小
    figdir = OUTPUT / "figures"
    figdir.mkdir(exist_ok=True)

    ladder_keys = ["小市值100基准", "小市值50", "小市值低波50", "择时低波50", "★旗舰双周低波50"]
    ladder = pd.DataFrame({k: navs[k] for k in ladder_keys}).dropna()
    b = (bench.reindex(ladder.index).ffill()); ladder["中证1000"] = b / b.iloc[0]
    fig = plots.lines(ladder, "小市值增强阶梯净值（本项目，T+1 开盘，2010-2022.5）",
                      note="口径见 grill_enhance.md：成交 T+1 开盘、波动窗口 250、跌停惩罚；"
                           "双周旗舰含 Q19 频率口径差", log=True)
    plots.save(fig, figdir / "enh_ladder.png")

    # 择时仓位暴露（对应阶梯图，E9）：与阶梯同 5 条策略，画每条的市场仓位随时间。
    # 彩条 = 满仓持股，空白 = 择时空仓月持币（收益记 0%）。前三条无择时始终满仓，
    # 择时低波50 每年空一/四月，旗舰加空六月——直观看到增强阶梯越往上、空仓越多。
    full_x = exposures[ladder_keys[0]].index
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, k in enumerate(ladder_keys):
        exp = exposures[k]
        invested = exp.to_numpy() > 0.5
        flagship = k.startswith("★")
        ax.fill_between(exp.index, i, i + 0.8, step="post", color="#E5E8E8")     # 底：空仓灰
        ax.fill_between(exp.index, i, i + 0.8, where=invested, step="post",
                        color=plots.PORTFOLIO if flagship else "#28B463",
                        alpha=0.9 if flagship else 0.7)
    ax.set_yticks([i + 0.4 for i in range(len(ladder_keys))],
                  [k.lstrip("★") for k in ladder_keys], fontsize=10)
    ax.set_ylim(-0.05, len(ladder_keys))
    ax.set_xlim(full_x.min(), full_x.max())
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title("小市值增强阶梯 · 择时仓位暴露（本项目，2010-2022.5）", fontsize=12, pad=12)
    ax.legend(handles=[Patch(color="#28B463", alpha=0.7, label="满仓持股"),
                       Patch(color=plots.PORTFOLIO, alpha=0.9, label="旗舰满仓"),
                       Patch(color="#E5E8E8", label="择时空仓（持币，收益 0%）")],
              loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, frameon=False)
    fig.text(0.01, 0.005, "择时低波50 每年空一、四月；旗舰加空六月（E9；六月研报自述无经济逻辑）。"
                          "前三条无择时、始终满仓。", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plots.save(fig, figdir / "enh_position.png")

    base = navs["小市值100基准"]
    mret = base.resample("ME").last().pct_change().dropna()
    bret = bench.reindex(base.index).ffill().resample("ME").last().pct_change().dropna()
    mret, bret = mret.align(bret, join="inner")
    grid = pd.DataFrame({"绝对": mret.groupby(mret.index.month).mean() * 100,
                         "超额": (mret - bret).groupby(mret.index.month).mean() * 100})
    grid.index = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
    fig = plots.grouped_bars(grid, "小市值100 月度收益均值（表19/20）：一/四/六月为负",
                             note="择时依据：一/四月空仓（旗舰加六月）", percent=False, ylabel="月均收益 %")
    plots.save(fig, figdir / "enh_monthly.png")
    print(f"  图已写入 {figdir.relative_to(ROOT)}/enh_ladder.png, enh_position.png, enh_monthly.png")


def main(vol_sweep):
    OUTPUT.mkdir(exist_ok=True)
    cal, sessions, close, open_, full_close, bench, is_ld = load_inputs()

    print(__doc__.split("\n")[0])
    print("\n先验假设（跑之前固定，grill_enhance.md E 系列）")
    print(f"  区间          {START} ~ {END}（{len(sessions)} 个交易日）")
    print(f"  选股范围      上市满1年(243交易日)、非ST、非注册制、非北交所；信号日剔除涨停/停牌")
    print(f"  波动率        过去 {VOL_W} 交易日日收益率标准差（E4：先验60→实测采250）")
    print(f"  择时空仓      1/4月（旗舰加6月）持币0%（E9）")
    print(f"  跌停惩罚      冻结跌停调出位、首个非跌停日收盘卖出、均分再分配（E7）")
    print(f"  成交/成本     T+1开盘（Q19）；双边千三；基准 {BENCH}")
    print(f"\n⚠️ 带惩罚策略系统性高研报约2pp（E7：研报-2.8pp惩罚多是其执行口径，本项目不复制）；")
    print(f"   策略平均分位数系统性低研报约11pp（E5：Wind分位数方法学差异）。年化与胜率对得上。")

    # 频率 → (rebalances, panel, vol) 缓存，避免重复建面板（en.build_freq 持有此 dict）
    cache = {}

    def evaluate(freq, steps, cash_months, penalty, cost=COST):
        # 构造走 en.run_strategy（与样本外 07_enhance_oos.py 同一条路径，Q14）；
        # 指标在此按**全区间**算——样本外驱动 07 则分样本内/样本外两段各算一遍。
        s = en.run_strategy(cache, cal, sessions, close, open_, full_close, is_ld,
                            freq, steps, cash_months, penalty, cost, VOL_W, START, END)
        nav = s["nav"]
        # sel 已含 cash-out 空行，avg_percentile 自动跳过空仓期（E9）；
        # 仓位暴露（exposure）= 引擎实际目标权重之和的日度 ffill：择时空仓月清仓→≈0，其余≈1（E9）
        return nav, dict(
            ann=m.annualized_return(nav) * 100,
            turn=s["res"].turnover.median() * 100,
            win=m.win_rate(nav, bench, s["tds"], s["cash_td"]) * 100,
            sp=m.avg_percentile(s["sel"], full_close, s["sched"]) * 100,
            bp=m.benchmark_percentile(bench, full_close, s["tds"]) * 100,
        ), s["exposure"]

    print("\n" + "=" * 92)
    print("表26 汇总对照 —— 专题之二增强族（本文 vs 研报）    年化/换手/胜率/策略分位/基准分位，单位 %")
    print("=" * 92)
    print(f"{'策略':<18}{'年化':>14}{'换手':>13}{'胜率':>13}{'策略分位':>13}{'基准分位':>13}")
    rows = {}
    navs = {}
    exposures = {}
    for label, freq, steps, cash_months, penalty, ref in en.ROSTER:
        cost = P2["fee_test_cost_per_side"] if "费用" in label else COST
        nav, r, exp = evaluate(freq, steps, cash_months, penalty, cost)
        navs[label] = nav
        rows[label] = r
        exposures[label] = exp
        def cell(ours, theirs):
            return f"{ours:>6.1f}/{theirs:<5.1f}"
        print(f"{label:<18}{cell(r['ann'],ref[0]):>15}{cell(r['turn'],ref[1]):>14}"
              f"{cell(r['win'],ref[2]):>14}{cell(r['sp'],ref[3]):>14}{cell(r['bp'],ref[4]):>14}")

    pd.DataFrame(rows).T.to_csv(OUTPUT / "enhance_table26.csv")
    pd.DataFrame(navs).to_csv(OUTPUT / "enhance_navs.csv")

    monthly_distribution(navs["小市值100基准"], bench)
    structural_checks(rows, navs)
    capacity_test(cal, sessions, full_close, en.apply_timing)
    analyst_test(cal, sessions, close, open_, full_close, bench, is_ld, en.apply_timing)
    make_figures(navs, bench, exposures)

    if vol_sweep:
        print("\n" + "=" * 60)
        print("E4 波动率窗口敏感性（月频，低波50 / 择时低波50）")
        print("=" * 60)
        for w in P2["volatility_sweep"]:
            reb, panel, _ = en.build_freq(cache, cal, "monthly", START, END, full_close, VOL_W)
            vol = fa.volatility(full_close, reb, w, ids=panel.ids)
            lv = u.cascade(panel, [("cap", True, 100), ("vol", True, 50)], {"vol": vol}, predicates=u.BUYABLE2)
            cash = pd.DatetimeIndex([d for d in reb if d.month in (1, 4)])
            a0 = m.annualized_return(bt.run(lv, close, open_, sessions).nav) * 100
            a1 = m.annualized_return(bt.run(lv, close, open_, sessions, cash_dates=cash).nav) * 100
            print(f"  窗口 {w:>3d} 交易日   低波50 {a0:6.2f}%   择时低波50 {a1:6.2f}%")
        print("  研报         低波50  32.8%   择时低波50  39.5%")

    print(f"\n结果写入 {OUTPUT.relative_to(ROOT)}/enhance_*.csv。容量测试(表22)/分析师覆盖(表24)见后续步骤。")


def capacity_test(cal, sessions, full_close, apply_timing):
    """表22 容量测试（研报 p.22 §3.4.2，E8）：择时双周低波100，2019-2022.6，
    每票每日限成交当日成交额 5%，未成交顺延。用 performance 出完整指标。"""
    CS, CE = str(P2["capacity_start"]), str(P2["capacity_end"])
    cap_sessions = sessions[(sessions >= pd.Timestamp(CS)) & (sessions <= pd.Timestamp(CE))]
    close = full_close.reindex(cap_sessions)
    turn_yuan = data.series("price_raw", "total_turnover", CS, CE).reindex(cap_sessions)
    bench = data.wide("index_price", "close", CS, CE)[BENCH].reindex(cap_sessions)
    reb = bt.rebalance_dates(cal, "biweekly", CS, CE)
    panel = u.panel(reb)
    vol = fa.volatility(full_close, reb, VOL_W, ids=panel.ids)
    sel = u.cascade(panel, [("cap", True, 200), ("vol", True, 100)], {"vol": vol}, predicates=u.BUYABLE2)
    sel, cash = apply_timing(sel, cap_sessions, (1, 4, 6))

    print("\n" + "=" * 78)
    print("表22 容量测试 —— 择时双周低波100，每票每日≤当日成交额5%，2019-2022.6")
    print("=" * 78)
    cols = [("简单回测", 1e9, 54.2), ("容量1亿", 1e8, 45.5), ("容量5亿", 5e8, 29.9), ("容量10亿", 1e9, 22.8)]
    rows = {}
    for name, capital, ref_ann in cols:
        part = 1e9 if name == "简单回测" else 0.05
        nav = bt.run_with_capacity(sel, close, turn_yuan, cap_sessions, capital,
                                   max_participation=part, cost_per_side=COST, cash_dates=cash)
        perf = m.performance(nav, bench, rf=B["rf"])
        rows[name] = perf
        print(f"  {name:8s} 年化 {perf['策略年化收益率']*100:6.2f}% (研报{ref_ann})  "
              f"波动 {perf['策略年化波动率']*100:5.1f}%  夏普 {perf['策略夏普比率(rf=2%)']:.2f}  "
              f"回撤 {perf['策略最大回撤']*100:.1f}%")
    print("  研报：年化 54.2/45.5/29.9/22.8   波动 20.5/19.5/17.8/16.8（随资金递减是容量约束的签名）")
    pd.DataFrame(rows).to_csv(OUTPUT / "enhance_capacity.csv")


def analyst_test(cal, sessions, close, open_, full_close, bench, is_ld, apply_timing):
    """表24 分析师覆盖（研报 p.23 §3.4.4，E14）：择时双周低波50，选股先要求「有分析师
    覆盖」。覆盖用 consensus 目标价存在性作代理（E14）：某股在调仓日的前 180 交易日内
    出现过 one_year_target_price 即视为被覆盖。"""
    cov = data.load("analyst_coverage")
    cov_wide = cov.pivot_table(index="date", columns="order_book_id", values="covered", aggfunc="first")
    covered = cov_wide.reindex(sessions).notna()
    covered = covered.where(covered).ffill(limit=180).fillna(False)   # 目标价有效窗 ~180交易日
    reb = bt.rebalance_dates(cal, "biweekly", START, END)
    panel = u.panel(reb)
    vol = fa.volatility(full_close, reb, VOL_W, ids=panel.ids)
    cov_at_reb = covered.reindex(index=reb, columns=panel.ids).fillna(False)
    sel = u.cascade(panel, [("cap", True, 100), ("vol", True, 50)], {"vol": vol},
                    predicates=u.BUYABLE2, mask=cov_at_reb)
    sel, cash = apply_timing(sel, sessions, (1, 4, 6))
    res = bt.run(sel, close, open_, sessions, cost_per_side=COST, cash_dates=cash, limit_down=is_ld)
    sched = bt.trade_dates(sel.index, sessions)
    tds = pd.DatetimeIndex(sched.values)
    cash_td = pd.DatetimeIndex(sched.reindex(cash).dropna().values)
    med_names = int(sel.sum(axis=1)[sel.sum(axis=1) > 0].median())
    print("\n" + "=" * 78)
    print("表24 分析师覆盖 —— 择时双周低波50 ∩ 有覆盖（2010-2022.5，E14 代理）")
    print("=" * 78)
    print(f"  年化 {m.annualized_return(res.nav)*100:.1f}%（研报39.7）  "
          f"胜率 {m.win_rate(res.nav,bench,tds,cash_td)*100:.1f}  "
          f"每期实选中位 {med_names} 只（覆盖稀疏，E14：consensus 目标价 2012 峰、后递减）")
    print(f"  ⇒ 要求分析师覆盖显著抬高选股池市值、拉低收益（研报同向：39.7 < 旗舰 50.9）")
    res.nav.to_csv(OUTPUT / "enhance_analyst.csv")


def monthly_distribution(base_nav, bench):
    """表19（绝对）/表20（超额）月度收益分布，及一/四/六月异象（研报 p.20 §3.3）。"""
    def monthly(nav):
        mret = nav.resample("ME").last().pct_change().dropna()
        return mret
    s = monthly(base_nav)
    b = monthly(bench.reindex(base_nav.index).ffill())
    s, b = s.align(b, join="inner")
    grid = pd.DataFrame({"绝对": s * 100, "超额": (s - b) * 100})
    grid["year"] = grid.index.year
    grid["month"] = grid.index.month
    abs_avg = grid.groupby("month")["绝对"].mean()
    exc_avg = grid.groupby("month")["超额"].mean()
    print("\n" + "=" * 78)
    print("表19/20 —— 小市值100 月度收益分布：一/四/六月异象（择时依据，研报 p.20）")
    print("=" * 78)
    names = ["一","二","三","四","五","六","七","八","九","十","十一","十二"]
    print("  " + "".join(f"{n:>7}" for n in names))
    print("  绝对均值" + "".join(f"{abs_avg.get(i, float('nan')):>7.1f}" for i in range(1, 13)))
    print("  超额均值" + "".join(f"{exc_avg.get(i, float('nan')):>7.1f}" for i in range(1, 13)))
    print(f"  研报绝对：一月-5.1 四月-1.2 六月-2.2（其余为正）；超额：仅一月为负-1.1（择时=一四月空仓）")
    neg = [names[i-1] for i in range(1,13) if abs_avg.get(i,0) < 0]
    print(f"  ⇒ 本文绝对为负的月份：{neg}（研报：一、四、六）")
    grid.to_csv(OUTPUT / "enhance_monthly_distribution.csv")


def structural_checks(rows, navs):
    """研报的定性结论（比头条数字硬）。E2：分位数/胜率为主，年化为头条。"""
    def ann(k): return rows[k]["ann"]
    checks = [
        ("择时(1/4空仓)相对基准 +≈10pp（表10 vs 表1）",
         f"{ann('择时小市值100') - ann('小市值100基准'):+.1f}pp",
         ann("择时小市值100") - ann("小市值100基准") >= 8),
        ("频率形态 双周>单周>月>双月>季度（表15-18）",
         f"双周{ann('双周+惩罚择时100'):.0f}>单周{ann('单周+惩罚择时100'):.0f}>"
         f"月{ann('跌停惩罚择时100'):.0f}>双月{ann('双月+惩罚择时100'):.0f}>季{ann('季度+惩罚择时100'):.0f}",
         ann("双周+惩罚择时100") > ann("单周+惩罚择时100") > ann("跌停惩罚择时100")
         > ann("双月+惩罚择时100") > ann("季度+惩罚择时100")),
        ("旗舰年化 >50%（表21）",
         f"{ann('★旗舰双周低波50'):.1f}%",
         ann("★旗舰双周低波50") > 50),
        ("旗舰 > 保留六月 > 择时低波50月频（表21/25/12）",
         f"{ann('★旗舰双周低波50'):.1f} > {ann('保留六月'):.1f} > {ann('择时低波50'):.1f}",
         ann("★旗舰双周低波50") > ann("保留六月") > ann("择时低波50")),
    ]
    print("\n结构性检查（研报定性结论）")
    for label, got, ok in checks:
        print(f"  [{'通过' if ok else '不符'}] {label:44s} {got}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-sweep", action="store_true", help="追加 E4 波动率窗口敏感性")
    main(ap.parse_args().vol_sweep)
