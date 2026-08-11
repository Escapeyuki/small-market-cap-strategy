"""rqdatac 抓取 + 本地 Parquet 缓存 + 硬性额度守卫。

本授权下实测出来的额度经济学（见 grill.md）：

  * 每个 field-cell 约 8.8 字节（88.2 万 field-cell → 7.73 MB，干净的一次测量）
  * **每次调用约 250 KB 固定开销**，与返回量无关
  * 额度计量在调用后约 5 秒才结算，之后稳定；量测时若 sleep 不足会读到 0

中间那条决定了整个架构：**必须少量大调用**。逐股循环会在下载到任何有用数据之前
就把 1 GB 额度全烧在调用开销上。这里所有东西都可续传，中断后从第一个缺失的块
接着跑，而不是为已经落盘的数据再付一次钱。
"""
import time
from pathlib import Path

import pandas as pd
import rqdatac as rq

from .config import CFG, DATA_ROOT

_initialised = False


class QuotaExhausted(RuntimeError):
    """宁可抛异常，也不静默烧掉剩下的额度。"""


def init():
    global _initialised
    if not _initialised:
        rq.init()
        _initialised = True


def quota_used():
    init()
    return rq.user.get_quota()["bytes_used"]


def quota_remaining():
    return CFG["data"]["quota_limit_bytes"] - quota_used()


def check_quota():
    remaining = quota_remaining()
    if remaining < CFG["data"]["quota_floor_bytes"]:
        raise QuotaExhausted(
            f"only {remaining / 1e6:.0f} MB left, below the "
            f"{CFG['data']['quota_floor_bytes'] / 1e6:.0f} MB floor"
        )
    return remaining


# --------------------------------------------------------------------------- 缓存

def path(dataset, tag):
    return DATA_ROOT / dataset / f"{tag}.parquet"


def exists(dataset, tag):
    return path(dataset, tag).exists()


def write(dataset, tag, df):
    """先写临时文件再改名，中途被打断也不会留下半个块。"""
    target = path(dataset, tag)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(target)
    return target


# 一条逻辑序列按时段拆在多个数据集里（grill Q8/Q9）：`_back` 是 2000-01~2012-12，
# 无后缀是研报主区间，`_ext` 是 2022-04 至今。调用方应该只报逻辑名，让 `series`
# 去拼时段，而不必知道某个日期落在哪一段。
# price_post_back 只带 `close`、price_open_back 只带 `open`，两者日期虽有重叠，
# 但按字段拼接，同一字段永远不会出现重复的 (date, order_book_id)。
ERAS = {
    "price_post": ["price_post_back", "price_open_back", "price_post", "price_post_ext"],
    "price_raw": ["price_raw_back", "price_raw", "price_raw_ext"],
    "market_cap": ["market_cap_back", "market_cap", "market_cap_ext"],
    "pb": ["pb", "pb_ext"],
    "turnover": ["turnover", "turnover_ext"],
}


def parts(dataset, start=None, end=None):
    """与 [start, end] 有交集的缓存块。块名本身就编码了日期范围，
    所以范围外的块直接跳过，连读都不读。"""
    found = []
    for p in sorted((DATA_ROOT / dataset).glob("*.parquet")):
        if "_" in p.stem:
            first, last = p.stem.split("_")
            if start is not None and last < pd.Timestamp(start).strftime("%Y%m%d"):
                continue
            if end is not None and first > pd.Timestamp(end).strftime("%Y%m%d"):
                continue
        found.append(p)
    return found


def load(dataset, start=None, end=None):
    found = parts(dataset, start, end)
    if not found:
        raise FileNotFoundError(f"no cached data for {dataset!r}; run scripts/01_fetch.py")
    df = pd.concat((pd.read_parquet(p) for p in found), ignore_index=True)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df


def _pivot(frames, field):
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise FileNotFoundError(f"no rows carrying {field!r} in the requested window")
    return df.pivot(index="date", columns="order_book_id", values=field).sort_index()


def wide(dataset, field, start=None, end=None):
    """长表缓存 → 日期 × order_book_id 矩阵，回测要的就是这个形状。"""
    return _pivot([load(dataset, start, end)], field)


def series_many(name, fields, start=None, end=None, dates=None):
    """同 `wide`，但会把一条逻辑序列的各个时段拼起来；
    一个块无论要取几个字段都只读一遍。

    从来没带过某字段的时段直接跳过而不是报错：回补历史当初是**故意**只买了
    close（grill Q8），所以 series("price_post", "open") 从 2012-12 才开始
    是正常的，不是数据缺失。
    """
    collected = {f: [] for f in fields}
    for era in ERAS[name]:
        try:
            df = load(era, start, end)
        except FileNotFoundError:
            continue
        # 选股只发生在调仓日，在这里先筛一道，月频回测的 pivot 就是 112 行
        # 而不是 2267 行。
        if dates is not None:
            df = df[df["date"].isin(pd.DatetimeIndex(dates))]
        for f in fields:
            if f in df.columns:
                collected[f].append(df[["date", "order_book_id", f]])
    missing = [f for f in fields if not collected[f]]
    if missing:
        raise FileNotFoundError(f"no cached {name!r} carrying {missing}")
    return {f: _pivot(frames, f) for f, frames in collected.items()}


def series(name, field, start=None, end=None, dates=None):
    return series_many(name, [field], start, end, dates)[field]


def flatten(df):
    """rqdatac 返回的是 (order_book_id, date) 双重索引，这里摊平成列。"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name:
        df = df.reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- 抓取

def blocks(start, end, years):
    """按 `years` 个自然年切块，逐个吐出 (起, 止) 日期对。"""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cur = start
    while cur <= end:
        stop = min(pd.Timestamp(year=cur.year + years - 1, month=12, day=31), end)
        yield cur, stop
        cur = stop + pd.Timedelta(days=1)


def harvest(dataset, fetch, start, end, years=None, label=None):
    """可续传、带额度守卫的分块抓取。

    `fetch(block_start, block_end)` 必须返回一个 DataFrame。已经落盘的块直接
    跳过，连网都不碰。
    """
    years = years or CFG["data"]["chunk_years"]
    label = label or dataset
    fetched = skipped = 0

    for s, e in blocks(start, end, years):
        tag = f"{s:%Y%m%d}_{e:%Y%m%d}"
        if exists(dataset, tag):
            skipped += 1
            continue
        check_quota()
        t0 = time.time()
        df = flatten(fetch(s, e))
        target = write(dataset, tag, df)
        fetched += 1
        print(
            f"  {label:28s} {tag}  rows={len(df):>9,}  "
            f"{time.time() - t0:5.1f}s  cum_quota={quota_used() / 1e6:7.1f}MB  "
            f"-> {target.relative_to(DATA_ROOT.parent)}"
        )
    if skipped:
        print(f"  {label:28s} {skipped} block(s) already cached, skipped")
    return fetched


SETTLE_SECONDS = 6      # 计量在调用后约 5 秒落定，实测值，见模块说明


def measure(name, fn):
    """跑一次调用并报告它花掉的额度，等计量结算之后再读。"""
    before = quota_used()
    t0 = time.time()
    result = fn()
    time.sleep(SETTLE_SECONDS)
    after = quota_used()
    rows = 0 if result is None else len(result)
    print(
        f"  {name:34s} rows={rows:>8,}  delta={(after - before) / 1e6:6.2f}MB  "
        f"{time.time() - t0:5.1f}s"
    )
    return result, after - before, rows
