"""绘图。计算全在 analytics.py，这里只负责把算好的东西画出来。

**中文字体必须显式设置。** 不设的话 matplotlib 回落到 DejaVu Sans，所有中文
标签变成方框——而且它不报错，只在 stderr 上刷一行 warning。图看上去是画出来
了，只是没法读。`use_font()` 按可得性挑一个，一个都没有就明说，不装作没事。

这里只放**图形**，不放**图号**：图号与数据的对应关系在 scripts/03_analytics.py，
挨着算它的那几行。研报的 31 张图归根到底只有 6 种形状，按形状写函数不必写
31 遍几乎一样的代码。
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")                       # 无显示环境下也能跑
import matplotlib.pyplot as plt             # noqa: E402

# 按优先级排列。PingFang SC 是 macOS 现代中文默认字体，Heiti SC 是它的前任，
# Arial Unicode MS 覆盖面广但字形难看，放最后兜底。
FONTS = ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti",
         "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "SimHei"]

# 组合用暖红突出，其余退到背景里。这一串**不含** PORTFOLIO，且要够长：图6
# 一次画 10 个分组加一条基准，配色一旦回绕，第 1 组就会和基准撞成同一个颜色，
# 图上看起来像少了一条线。
PORTFOLIO = "#C0392B"
INDEX_COLORS = ["#2E86C1", "#28B463", "#8E44AD", "#F39C12", "#16A085",
                "#D35400", "#7F8C8D", "#2C3E50", "#1ABC9C", "#B7950B",
                "#5D6D7E", "#A569BD"]
BENCHMARK = "#111111"           # 基准单独一个颜色，且画成虚线


def use_font():
    """挑一个装得上的中文字体并全局设定，返回它的名字。"""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + FONTS
            plt.rcParams["axes.unicode_minus"] = False       # 负号也会变方框
            return name
    warnings.warn(
        "找不到任何中文字体，图上的中文标签会变成方框。"
        f"候选：{'、'.join(FONTS)}",
        stacklevel=2,
    )
    return None


def style():
    plt.rcParams.update({
        "figure.figsize": (10, 5.2),
        "figure.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 10,
    })


def _finish(ax, title, note=None, percent=False, legend=True):
    ax.set_title(title, fontsize=12, pad=12)
    if percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=9)
    if note:
        ax.figure.text(0.01, 0.01, note, fontsize=8, color="#666666")
    ax.figure.tight_layout(rect=(0, 0.03, 1, 1) if note else None)
    return ax.figure


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 形状

def lines(frame, title, note=None, log=False, percent=False, highlight=None,
          benchmark=None, ylabel=None):
    """多条时间序列。`highlight` 那一列用暖色加粗，`benchmark` 那一列画成黑虚线。

    缺失值留成缺口而不是连过去——中证1000 成分股在 2014-10-17 之前根本不存在，
    画一条直线跨过去等于伪造数据。
    """
    fig, ax = plt.subplots()
    for i, column in enumerate(frame.columns):
        series = frame[column].dropna()
        is_main = column == highlight
        is_bench = column == benchmark
        ax.plot(series.index, series.to_numpy(), label=str(column),
                color=BENCHMARK if is_bench else
                      PORTFOLIO if is_main else INDEX_COLORS[i % len(INDEX_COLORS)],
                linestyle="--" if is_bench else "-",
                linewidth=1.9 if is_main else 1.3 if is_bench else 1.1,
                alpha=1.0 if is_main or is_bench else 0.85)
    if log:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    if ylabel:
        ax.set_ylabel(ylabel)
    return _finish(ax, title, note, percent)


def density(samples, title, note=None, clip=None, xlabel=None):
    """概率密度图（高斯核）。`samples` 是 {标签: 一维样本}。

    研报图10、图12–14 画的都是这个。样本量少于 2 或方差为 0 时核密度估计没有
    定义，那一条直接跳过并在图例里说明，不画一条假的曲线。
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots()
    for i, (label, values) in enumerate(samples.items()):
        values = np.asarray(pd.Series(values).dropna(), dtype=float)
        if clip is not None:
            values = values[(values >= clip[0]) & (values <= clip[1])]
        if values.size < 2 or np.isclose(values.std(), 0):
            continue
        grid = np.linspace(values.min(), values.max(), 400)
        ax.plot(grid, gaussian_kde(values)(grid), label=f"{label}（n={values.size}）",
                color=INDEX_COLORS[i % len(INDEX_COLORS)], linewidth=1.6)
    ax.set_ylabel("概率密度")
    if xlabel:
        ax.set_xlabel(xlabel)
    return _finish(ax, title, note)


def barh(series, title, note=None, percent=True, color=PORTFOLIO):
    """横向条形图，研报图15–18 的行业分布用的就是这个版式（升序，最大的在上）。"""
    series = series.sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.28 * len(series))))
    ax.barh(range(len(series)), series.to_numpy(), color=color, alpha=0.85)
    ax.set_yticks(range(len(series)), list(series.index), fontsize=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%" if percent else f"{v:g}")
    ax.grid(axis="y", visible=False)
    for i, value in enumerate(series.to_numpy()):
        ax.text(value, i, f" {value * 100:.1f}%" if percent else f" {value:g}",
                va="center", fontsize=8)
    return _finish(ax, title, note, legend=False)


def grouped_bars(frame, title, note=None, percent=True, ylabel=None):
    """分组柱状图。研报图28 的日历效应：横轴周一到周五，每个阶段一组。"""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    n = len(frame.columns)
    width = 0.8 / n
    positions = np.arange(len(frame.index))
    for i, column in enumerate(frame.columns):
        ax.bar(positions + (i - (n - 1) / 2) * width, frame[column].to_numpy(),
               width=width, label=str(column), color=INDEX_COLORS[i % len(INDEX_COLORS)],
               alpha=0.9)
    ax.set_xticks(positions, list(frame.index))
    ax.axhline(0, color="#333333", linewidth=0.8)
    if percent:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.2f}%")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    return _finish(ax, title, note)


def stacked_bars_with_line(bars, line, title, note=None,
                           bar_label=None, line_label=None):
    """堆叠柱 + 右轴折线。研报图26（调出原因）就是这个版式：

    左轴是每期调出的总只数，右轴单独放退市/戴帽——它们只有个位数，跟左轴几十
    只的量级放一起会被压成一条贴地的线，看不出年报季那几根尖峰。
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = np.zeros(len(bars))
    for i, column in enumerate(bars.columns):
        values = bars[column].to_numpy(dtype=float)
        ax.bar(bars.index, values, bottom=bottom, width=20,
               label=str(column), color=INDEX_COLORS[i % len(INDEX_COLORS)], alpha=0.85)
        bottom += values
    if bar_label:
        ax.set_ylabel(bar_label)

    right = ax.twinx()
    right.grid(False)
    right.spines["top"].set_visible(False)
    for i, column in enumerate(line.columns):
        right.plot(line.index, line[column].to_numpy(dtype=float), label=str(column),
                   color=[PORTFOLIO, "#E67E22"][i % 2], linewidth=1.4, marker="o",
                   markersize=3)
    if line_label:
        right.set_ylabel(line_label)

    handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", fontsize=9)
    return _finish(ax, title, note, legend=False)


def dual_axis(left, right, title, note=None, left_label=None, right_label=None,
              left_log=False):
    """左右双轴。研报图31（止盈止损）左轴净值、右轴相对基准的净值比。"""
    fig, ax = plt.subplots()
    for i, column in enumerate(left.columns):
        ax.plot(left.index, left[column].to_numpy(dtype=float), label=str(column),
                color=INDEX_COLORS[i % len(INDEX_COLORS)], linewidth=1.4)
    if left_log:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
    if left_label:
        ax.set_ylabel(left_label)

    other = ax.twinx()
    other.grid(False)
    other.spines["top"].set_visible(False)
    for i, column in enumerate(right.columns):
        other.plot(right.index, right[column].to_numpy(dtype=float), label=str(column),
                   color=["#7F8C8D", "#34495E"][i % 2], linewidth=1.2, linestyle="--")
    if right_label:
        other.set_ylabel(right_label)

    handles = ax.get_legend_handles_labels()[0] + other.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + other.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", fontsize=9)
    return _finish(ax, title, note, legend=False)


def price_with_percentile(price, percentile, title, note=None, name=""):
    """左轴后复权股价、右轴时序分位数。研报图11 用平安银行做的示例。"""
    fig, ax = plt.subplots()
    ax.plot(price.index, price.to_numpy(), color=PORTFOLIO, linewidth=1.3,
            label=f"{name}股价（后复权）")
    ax.set_ylabel("股价（后复权）")

    right = ax.twinx()
    right.grid(False)
    right.spines["top"].set_visible(False)
    right.plot(percentile.index, percentile.to_numpy(), color="#2E86C1",
               linewidth=1.0, alpha=0.8, label="股价分位数")
    right.set_ylabel("股价分位数")
    right.set_ylim(0, 1)

    handles = ax.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + right.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left", fontsize=9)
    return _finish(ax, title, note, legend=False)
