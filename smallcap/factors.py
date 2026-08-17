"""因子计算 —— 纯计算，从本地价格派生，不碰 rqdatac。

专题之二第 3 部分的增强只用到一个因子：**波动率**（低波 50 的第二级筛选）。研报
全文从未定义它的窗口（grill_enhance.md E4）：附录表27 把「过去 1/3/6 个月日收益率
标准差」列为三个独立因子。本项目先验取**过去 60 个交易日日收益率标准差**，并对
{20, 60, 120} 作单变量敏感性——那张敏感性表本身就是交付物（grill.md Q14）。

多因子遍历（研报第 2 部分策略1-8、附录表27）不在本次范围内：它需要 PIT 财务因子，
违背 grill.md Q11。故本模块只实现波动率，不铺一个用不到的因子库。
"""
import pandas as pd


def volatility(post_close, dates, window, ids=None):
    """过去 `window` 个交易日的日收益率标准差，采样在选股日 `dates` 上。

    `post_close` 是全市场后复权收盘价（日期 × order_book_id），必须向前多带至少
    `window` 个交易日的历史，否则最早的选股日会因回看不足而得 NaN。返回宽表
    （行 = `dates`，列 = 股票），未合格/历史不足处为 NaN——上层 cascade 用 rank
    自然把 NaN 排除在筛选之外。

    上市满 1 年（243 交易日，E11）的选股范围保证被选股票必有 ≥243 日历史，
    足够 60 日窗口，所以低波 50 那一级不会因历史不足而选不满。
    """
    returns = post_close.pct_change()
    vol = returns.rolling(window, min_periods=window).std()
    vol = vol.reindex(pd.DatetimeIndex(dates))
    if ids is not None:
        vol = vol.reindex(columns=ids)
    return vol
