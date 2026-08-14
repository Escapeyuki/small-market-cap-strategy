"""一次性 rqdatac 抓取。

**按价值排序，而不是按时间顺序**（grill.md Q13）：中途失败时，手里留下的应该是
一份完整可编码的复现数据集，而不是 26 年份的单一字段。每一步都可续传——已缓存
的块直接跳过，连网都不碰。

    python scripts/01_fetch.py --smoke    # 只探测与校验，几 MB，安全
    python scripts/01_fetch.py            # 真跑，约 600 MB

先跑 --smoke。它会验证抓取依赖的每一个字段名、形状和 dtype，并打印预算推算，
好让不可逆的那一次是从**实测**开始，而不是从我的心算开始。
"""
import argparse
import time
import sys
from pathlib import Path

import pandas as pd
import rqdatac as rq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smallcap import data
from smallcap.config import CFG

P = CFG["periods"]
BENCH = CFG["backtest"]["benchmark"]
INDICES = ["000300.XSHG", "000905.XSHG", "000852.XSHG"]     # 沪深300 / 中证500 / 中证1000

REPORT_START, REPORT_END = str(P["report_start"]), str(P["report_end"])
BACK_START = str(P["price_percentile_base"])
EXT_END = str(P["extension_end"])
BACK_END = "2012-12-02"
EXT_START = "2022-04-01"
# 图6 十分组回测从 2007-01-01 起，比研报主区间早六年，而 market_cap / ST /
# volume 这些选股必需的字段最早也只回溯到 2007 —— 正好够用。
DECILE_START = str(P["decile_start"])

# 字段按复权方式分开：收益要用后复权价，涨停判定与容量测算要用不复权价。
# 混在同一次 get_price 里取会得到自相矛盾的结果（实测：`close` 返回的是复权价，
# 而 `prev_close` 不是）。
POST_FIELDS = ["open", "close"]
RAW_FIELDS = ["close", "limit_up", "volume", "total_turnover"]


def live_ids(date):
    """截至该日在市的 A 股，**包含日后退市的标的**。"""
    return list(rq.all_instruments(type="CS", date=date).order_book_id)


def all_ids():
    return list(rq.all_instruments(type="CS").order_book_id)


_INST = None


def instruments():
    global _INST
    if _INST is None:
        _INST = rq.all_instruments(type="CS")
    return _INST


def block_ids(start, end):
    """在 [start, end] 内任一时点上市过的全部股票。

    **绝不能用 live_ids(end)**：块内退市的股票不在块末的股票池里，按那份名单抓
    会把它静默丢掉，等于把本项目要避免的幸存者偏差又请了回来。
    """
    inst = instruments()
    listed = pd.to_datetime(inst.listed_date, errors="coerce")
    delisted = pd.to_datetime(
        inst.de_listed_date.replace("0000-00-00", "2099-12-31"), errors="coerce"
    )
    live = (listed <= pd.Timestamp(end)) & (delisted >= pd.Timestamp(start))
    return list(inst.loc[live, "order_book_id"])


def components(start, end):
    """三大指数的日度成分股，合成一张长表。

    日期区间形式每个指数只要一次调用就能返回 {日期: [股票]}，正是这一点才让
    日度成分变得付得起——逐日调用每次约 250 KB。中证1000 在 2014-10-17 之前
    直接返回空，这个缺口是真实的，**在图上标注而不是回补**。
    """
    rows = []
    for index_id in INDICES:
        members = rq.index_components(index_id, start_date=start, end_date=end)
        rows += [(d, index_id, s) for d, ids in (members or {}).items() for s in ids]
    return pd.DataFrame(rows, columns=["date", "index_id", "order_book_id"])


def turnover(start, end):
    """get_turnover_rate 的索引名是 `tradedate`，而其余数据集统一用 `date`。

    它也是 get_factor 唯一伺候不了的流动性字段——向 get_factor 要
    `turnover_rate` 会**静默返回一整列 NaN**，而不是报错。
    """
    df = rq.get_turnover_rate(block_ids(start, end), start, end, fields="today")
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["order_book_id", "date", "turnover_rate"])
    return df.reset_index().rename(columns={"tradedate": "date", "today": "turnover_rate"})


def melt_flags(df, name):
    """is_st_stock / is_suspended 返回的是「日期 × 股票」宽表，这里摊成长表存。

    只保留 True 行——两个标记都很稀疏，所以「不在表里」就等于 False，
    这一步把约 1500 万行的稠密布尔网格变成一张很小的事件表。
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "order_book_id", name])
    out = df.stack().rename(name).reset_index()
    out.columns = ["date", "order_book_id", name]
    out = out[out[name]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


# --------------------------------------------------------------------------- 探测

def smoke():
    """校验抓取所需的每一个接口，然后推算预算。"""
    data.init()
    start_quota = data.quota_used()
    print(f"quota at start: {start_quota / 1e6:.1f} MB used "
          f"of {CFG['data']['quota_limit_bytes'] / 1e6:.0f} MB\n")

    s, e = "2021-03-01", "2021-03-31"
    ids = live_ids(e)
    ndays = len(rq.get_trading_dates(s, e))
    cells = len(ids) * ndays
    print(f"smoke window {s}..{e}: {len(ids):,} stocks x {ndays} days = {cells:,} cells\n")

    print("-- metadata --")
    inst, _, _ = data.measure("all_instruments (full)", lambda: rq.all_instruments(type="CS"))
    need = {"order_book_id", "listed_date", "de_listed_date", "symbol"}
    missing = need - set(inst.columns)
    assert not missing, f"all_instruments missing {missing}"
    print(f"     -> point-in-time universe derivable from listed/de_listed dates: "
          f"{len(inst):,} instruments, no per-year snapshots needed")

    # index_components 若支持日期区间，就是 1 次调用而不是约 164 次。
    try:
        comps, _, _ = data.measure(
            "index_components (range)",
            lambda: rq.index_components(BENCH, start_date="2021-01-01", end_date="2021-06-30"),
        )
        ranged = isinstance(comps, dict)
        print(f"     -> date-range form supported: {ranged}"
              f"{f' ({len(comps)} dates in one call)' if ranged else ''}")
    except Exception as exc:                                  # noqa: BLE001
        ranged = False
        print(f"     -> date-range form NOT supported ({type(exc).__name__}); "
              f"falls back to per-date calls at ~250 KB each")

    st, st_cost, _ = data.measure("is_st_stock (all, 1mo)", lambda: rq.is_st_stock(ids, s, e))
    susp, susp_cost, _ = data.measure("is_suspended (all, 1mo)", lambda: rq.is_suspended(ids, s, e))
    ind, _, _ = data.measure(
        "get_instrument_industry (citics_2019)",
        lambda: rq.get_instrument_industry(ids, source="citics_2019", level=1, date=e)
    )
    shares, _, _ = data.measure(
        "get_shares (free_circulation)", lambda: rq.get_shares(ids, s, e, fields=["free_circulation"])
    )
    assert shares is not None and "free_circulation" in shares.columns, "free_circulation missing"
    print("     -> free_circulation 可用；图24 自由流通市值分母 = 不复权收盘 × 它（rqdatac 无现成因子）")

    # 单次调用的增量不可信（计量有延迟），所以把整块框起来测，再除以
    # field-cell 数，而不是相信任何一次调用的读数。
    print("\n-- price / factor --")
    before = data.quota_used()
    t0 = time.time()
    post = rq.get_price(ids, s, e, fields=POST_FIELDS, adjust_type="post")
    raw = rq.get_price(ids, s, e, fields=RAW_FIELDS, adjust_type="none")
    mc = rq.get_factor(ids, ["market_cap"], s, e)
    time.sleep(data.SETTLE_SECONDS)
    spent = data.quota_used() - before
    n_fields = len(POST_FIELDS) + len(RAW_FIELDS) + 1
    per_field_cell = spent / (cells * n_fields)
    print(f"  3 calls, {n_fields} fields x {cells:,} cells -> {spent / 1e6:.2f} MB "
          f"({per_field_cell:.1f} B per field-cell, {time.time() - t0:.0f}s)")

    for name, df, cols in [("post", post, POST_FIELDS), ("raw", raw, RAW_FIELDS)]:
        assert df is not None and len(df), f"{name} price frame empty"
        assert set(cols) <= set(df.columns), f"{name} missing {set(cols) - set(df.columns)}"
    assert mc is not None and "market_cap" in mc.columns, "market_cap missing"

    # limit_up == 0 表示「无可用的涨跌幅参照」，含两种截然不同的情形，
    # 都已在本窗口上验证过：
    #   * 当日停牌      -> volume == 0，收盘价沿用前值
    #   * 注册制新股上市初期的无涨跌幅窗口 -> volume > 0，上市 <= 约 5 日
    # 两者都已被策略自身的过滤器排除（停牌 / 上市不满 20 日），所以这条不变量
    # 只在**真正存在涨跌幅限制**的行上断言。
    has_limit = raw["limit_up"] > 0
    assert (raw.loc[has_limit, "close"] <= raw.loc[has_limit, "limit_up"] + 1e-6).all(), \
        "raw close exceeds limit_up where a limit applies"
    halted = ~has_limit & (raw["volume"] == 0)
    debut = ~has_limit & (raw["volume"] > 0)
    print(f"     -> close<=limit_up holds on all {has_limit.sum():,} limited rows; "
          f"limit_up==0 on {halted.sum():,} halted + {debut.sum():,} post-IPO rows "
          f"(free 停牌 flag, cross-checks is_suspended)")

    flat = data.flatten(post)
    assert {"order_book_id", "date"} <= set(flat.columns)
    assert pd.api.types.is_datetime64_any_dtype(flat["date"])
    print("     -> flatten() yields tidy (order_book_id, date, ...) with datetime dates")

    print("\n-- projected harvest (measured B/field-cell x point-in-time universe) --")
    eras = [
        ("report  2012-12..2022-03", REPORT_START, REPORT_END, n_fields),
        ("back    2000-01..2012-12", BACK_START, BACK_END, 1),      # 只有 close
        ("extend  2022-04..today  ", EXT_START, EXT_END, n_fields),
    ]
    total = 0.0
    for label, a, b, fields in eras:
        listed = inst[(inst.listed_date <= b) & ((inst.de_listed_date == "0000-00-00")
                                                 | (inst.de_listed_date >= a))]
        n_days = len(rq.get_trading_dates(a, b))
        n_cells = len(listed) * n_days
        cost = n_cells * fields * per_field_cell
        total += cost
        print(f"  {label}  ~{len(listed):,} stk x {n_days:,} d x {fields}f"
              f" = {n_cells * fields / 1e6:6.1f}M field-cells  ~{cost / 1e6:6.0f} MB")

    overhead = 40 * 250_000
    misc = st_cost + susp_cost + 30e6                    # 标记 + 特征 + 指数
    print(f"  {'metadata / flags / characteristics':41s} ~{misc / 1e6:6.0f} MB")
    print(f"  {'call overhead (~40 calls @ 250 KB)':41s} ~{overhead / 1e6:6.0f} MB")

    projected = total + overhead + misc
    used = data.quota_used()
    print(f"\n  projected harvest ~{projected / 1e6:.0f} MB")
    print(f"  already used       {used / 1e6:.0f} MB")
    print(f"  headroom left     ~{(CFG['data']['quota_limit_bytes'] - projected - used) / 1e6:.0f} MB")
    print(f"\nsmoke spent {(used - start_quota) / 1e6:.1f} MB")


# --------------------------------------------------------------------------- 抓取

def harvest():
    """真正的抓取，按价值排序。可以反复重跑，已缓存的块会被跳过。"""
    data.init()
    print(f"quota: {data.quota_used() / 1e6:.0f} MB used, "
          f"{data.quota_remaining() / 1e6:.0f} MB left\n")

    print("[1/13] metadata")
    if not data.exists("instruments", "all"):
        data.write("instruments", "all", instruments().reset_index(drop=True))
    if not data.exists("calendar", "all"):
        dates = rq.get_trading_dates(BACK_START, EXT_END)
        data.write("calendar", "all", pd.DataFrame({"date": pd.to_datetime(dates)}))

    print("[2/13] post-adjusted open+close — report window")
    data.harvest("price_post", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=POST_FIELDS, adjust_type="post"), REPORT_START, REPORT_END)

    print("[3/13] market_cap — report window")
    data.harvest("market_cap", lambda a, b: rq.get_factor(
        block_ids(a, b), ["market_cap"], a, b), REPORT_START, REPORT_END)

    print("[4/13] raw close + limit_up + volume + turnover — report window")
    data.harvest("price_raw", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=RAW_FIELDS, adjust_type="none"), REPORT_START, REPORT_END)

    print("[5/13] ST / suspension flags")
    data.harvest("st", lambda a, b: melt_flags(
        rq.is_st_stock(block_ids(a, b), a, b), "is_st"), REPORT_START, EXT_END, label="is_st_stock")
    data.harvest("suspended", lambda a, b: melt_flags(
        rq.is_suspended(block_ids(a, b), a, b), "is_suspended"),
        REPORT_START, EXT_END, label="is_suspended")

    print("[6/13] back-history post-adjusted close (price percentile baseline)")
    data.harvest("price_post_back", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=["close"], adjust_type="post"), BACK_START, BACK_END)

    print("[7/13] extension to today")
    data.harvest("price_post_ext", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=POST_FIELDS, adjust_type="post"), EXT_START, EXT_END)
    data.harvest("market_cap_ext", lambda a, b: rq.get_factor(
        block_ids(a, b), ["market_cap"], a, b), EXT_START, EXT_END)
    data.harvest("price_raw_ext", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=RAW_FIELDS, adjust_type="none"), EXT_START, EXT_END)

    print("[8/13] index prices")
    data.harvest("index_price", lambda a, b: rq.get_price(
        INDICES, a, b, fields=["close"]), BACK_START, EXT_END, label="index_price")

    print("[9/13] index components — 图19 PB, 图22-24 市值中位数与换手率")
    data.harvest("index_components", components, REPORT_START, EXT_END,
                 label="index_components")

    print("[10/13] pb_ratio_lyr — 图19, 整体法 PB = sum(market_cap) / sum(net assets)")
    data.harvest("pb", lambda a, b: rq.get_factor(
        block_ids(a, b), ["pb_ratio_lyr"], a, b), REPORT_START, REPORT_END)
    data.harvest("pb_ext", lambda a, b: rq.get_factor(
        block_ids(a, b), ["pb_ratio_lyr"], a, b), EXT_START, EXT_END)

    print("[11/13] turnover_rate + free_circulation — 图24 换手率的两种分母口径")
    data.harvest("turnover", turnover, REPORT_START, REPORT_END)
    data.harvest("turnover_ext", turnover, EXT_START, EXT_END)
    # 换手率分母：turnover_rate 反推出来的是**流通A股市值**（实测 重建流通市值/(收盘×流通A股)
    # = 0.997），而研报图24 的水平更像**自由流通**口径。rqdatac 没有自由流通市值因子
    # （6 个候选因子标定后 3 个=总市值、3 个=流通市值，无一是自由流通），只能抓自由流通
    # 股本，自由流通市值 = 不复权收盘价 × free_circulation（见 analytics.free_float_cap）。
    data.harvest("free_circ", lambda a, b: rq.get_shares(
        block_ids(a, b), a, b, fields=["free_circulation"]), REPORT_START, REPORT_END)
    data.harvest("free_circ_ext", lambda a, b: rq.get_shares(
        block_ids(a, b), a, b, fields=["free_circulation"]), EXT_START, EXT_END)

    print("[12/13] 图6 十分组所需的 2007-2012 选股字段")
    # price_post_back 当初只买了 close（grill Q8：它只用于股价分位数基线），
    # 分组回测却要真的持仓，于是这里单独补 open。两个数据集各自只带一个字段，
    # 拼接时按字段取用，不会产生重复的 (date, order_book_id)。
    data.harvest("price_open_back", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=["open"], adjust_type="post"), DECILE_START, BACK_END)
    data.harvest("market_cap_back", lambda a, b: rq.get_factor(
        block_ids(a, b), ["market_cap"], a, b), DECILE_START, BACK_END)
    data.harvest("price_raw_back", lambda a, b: rq.get_price(
        block_ids(a, b), a, b, fields=["close", "limit_up", "volume"],
        adjust_type="none"), DECILE_START, BACK_END)
    # ST / 停牌 沿用同一个数据集：新块的日期区间与既有块不重叠。
    data.harvest("st", lambda a, b: melt_flags(
        rq.is_st_stock(block_ids(a, b), a, b), "is_st"),
        DECILE_START, BACK_END, label="is_st_stock (back)")
    data.harvest("suspended", lambda a, b: melt_flags(
        rq.is_suspended(block_ids(a, b), a, b), "is_suspended"),
        DECILE_START, BACK_END, label="is_suspended (back)")

    print("[13/13] industry snapshots")
    # 中信一级用 citics_2019（研报口径）。旧的 zx_instrument_industry 是异常源：把宁德时代
    # 等新能源股错分到汽车，与标准中信 citics/citics_2019 都不一致（图16 因此对不齐）。见 grill.md #2。
    if not data.exists("industry", "annual"):
        frames = []
        for year in range(2012, 2027):
            date = f"{year}-12-31" if year < 2026 else EXT_END
            snap = rq.get_instrument_industry(live_ids(date), source="citics_2019",
                                              level=1, date=date)
            if snap is not None and len(snap):
                frames.append(snap.reset_index()[["order_book_id", "first_industry_name"]]
                              .assign(date=pd.Timestamp(date)))
        data.write("industry", "annual", pd.concat(frames, ignore_index=True))

    print(f"\ndone. quota used: {data.quota_used() / 1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="校验接口并推算体量，只花几 MB")
    args = ap.parse_args()
    smoke() if args.smoke else harvest()
