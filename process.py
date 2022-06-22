#!/usr/bin/env python3
"""
    :author: pk13055 :brief: Download the OHLC and metrics for a given list of tickers
"""
import argparse
from multiprocessing import Pool
import os
from random import shuffle

import numpy as np
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf

API_TOKEN = os.getenv("API_TOKEN", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe",
        type=str,
        default="data/universe.csv",
        help="universe of tickers to download",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("API_TOKEN"),
        help="Token for FMP API access",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        default="data/metrics.csv",
        help="Output metrics csv file",
    )
    args = parser.parse_args()
    return args


def download_candles(tickers: list) -> None:
    """Download OHLC from Yahoo Finance given list of tickers"""
    data = yf.download(tickers=" ".join(tickers), period="3y", group_by="ticker")
    for ticker in tickers:
        ticker_df = data[ticker].sort_index()
        ticker_df.to_csv(f"data/tickers/{ticker}.csv", float_format="%.2f")


def get_raw_data(ticker: str) -> pd.DataFrame:
    """Download and partially parse the raw data"""
    BASE_CASH_FLOW = "https://financialmodelingprep.com/api/v3/cash-flow-statement/"
    BASE_RATIOS = "https://financialmodelingprep.com/api/v3/ratios/"
    print(f"Processing {ticker} ...")
    cashflow = requests.get(
        f"{BASE_CASH_FLOW}{ticker}",
        params={"apikey": API_TOKEN, "period": "quarter", "limit": 13},
    ).json()
    ratios = requests.get(
        f"{BASE_RATIOS}{ticker}",
        params={"apikey": API_TOKEN, "period": "quarter", "limit": 13},
    ).json()

    cashflow = pd.DataFrame(cashflow)
    ratios = pd.DataFrame(ratios)
    try:
        cashflow = cashflow.loc[:, ["date", "operatingCashFlow"]]
        ratios = ratios.loc[
            :, ["date", "symbol", "enterpriseValueMultiple", "debtEquityRatio"]
        ]
        cashflow.date = pd.to_datetime(cashflow.date)
        ratios.date = pd.to_datetime(ratios.date)
        # filtering data only for the last 3 years
        cashflow = cashflow.loc[
            (cashflow.date >= "01-01-2019") & (cashflow.date <= "31-12-2022")
        ]
        ratios = ratios.loc[
            (ratios.date >= "01-01-2019") & (ratios.date <= "31-12-2022")
        ]
        cashflow = cashflow.sort_values("date", ascending=False).set_index("date")
        ratios = ratios.sort_values("date", ascending=False).set_index("date")
        metrics = pd.concat([cashflow, ratios], axis=1)
        metrics.reset_index(inplace=True)
    except Exception as e:
        print(f"[error] {ticker} NOT processed ({e})...")
        metrics = pd.DataFrame([])
    print(f"{ticker} processed ...")
    return metrics


def download_metrics(tickers: list) -> pd.DataFrame:
    """Download operating cash flow, EV/EBITDA, D/E for tickers"""
    pool = Pool(8)
    frames = pool.map_async(get_raw_data, tickers).get()
    df = pd.concat(frames)
    return df


def visualize(metrics: pd.DataFrame) -> None:
    """Visualize the given metrics"""
    # get last quarter data to visualize
    _metrics = metrics.groupby("symbol").last()

    # plot hist SP500 and NQ decile
    ax = sns.countplot(data=_metrics, x="SP500_decile", hue="sector")
    ax = sns.countplot(data=_metrics, x="NQ_decile", hue="sector")

    # plot grades
    _metrics.grades = pd.to_numeric(_metrics.grades, errors="coerce")
    ax = sns.boxenplot(data=_metrics, x="sector", y="grades")


def main():
    args = parse_args()

    univ = pd.read_csv(args.universe)
    tickers = univ.ticker.tolist()
    shuffle(tickers)

    # download and save ohlc candles for last 3y to `data/tickers/`
    download_candles(tickers)
    # download and save metrics
    metrics = download_metrics(tickers)
    metrics.to_csv(args.metrics, float_format="%.4f")

    # calculate operatingCashFlow growth rate
    ocf_growth = metrics.groupby("symbol").apply(
        lambda df: (df.operatingCashFlow - df.operatingCashFlow.shift(1))
        / df.operatingCashFlow.shift(1)
    )
    metrics.loc[:, "ocf_growth"] = ocf_growth.reset_index(drop=True)
    stats = metrics.groupby("symbol")[
        "operatingCashFlow", "enterpriseValueMultiple", "debtEquityRatio"
    ].describe()

    def calculate_percentiles(ticker: str, col: str, value: float) -> float:
        sub_stats = stats.loc[ticker, col]
        if value >= sub_stats["75%"]:
            percentile = 1
        elif sub_stats["75%"] > value >= sub_stats["50%"]:
            percentile = 2
        elif sub_stats["50%"] > value >= sub_stats["25%"]:
            percentile = 3
        else:
            percentile = 4
        return percentile

    metric_labels = ["operatingCashFlow", "enterpriseValueMultiple", "debtEquityRatio"]
    # 3y percentile score
    for stat in metric_labels:
        metrics.loc[:, f"{stat}_3y_percentile"] = metrics.apply(
            lambda row: calculate_percentiles(row.symbol, stat, row[stat]), axis=1
        )
    # ticker decile score
    for stat in metric_labels:
        metrics.loc[:, f"{stat}_ticker_decile"] = pd.qcut(
            metrics[stat], 10, labels=list(range(10, 0, -1))
        )
    univ = pd.read_csv("data/universe_goldmaster.csv")
    metrics = metrics.merge(
        univ.loc[:, ["ticker", "sector"]], left_on="symbol", right_on="ticker"
    )
    del metrics["ticker"]
    # sector decile score
    for metric in metric_labels:
        decile_scores = []
        for sector, df in metrics.groupby("sector"):
            ranks = pd.qcut(df[metric], 10, labels=list(range(10, 0, -1)))
            decile_scores.append(ranks)
        decile_scores = pd.concat(decile_scores)
        metrics.loc[decile_scores.index, f"{metric}_sector_decile"] = decile_scores

    # index based scoring
    indices = pd.read_csv("data/universe_corr.csv")
    sp500 = indices.loc[
        indices.loc[:, "S&P 500"].fillna(0.0).astype(bool)
    ].ticker.tolist()
    nq = indices.loc[indices.loc[:, "S&P 500"].fillna(0.0).astype(bool)].ticker.tolist()
    metrics.loc[metrics.symbol.apply(lambda ticker: ticker in sp500), "SP500"] = True
    metrics.loc[metrics.symbol.apply(lambda ticker: ticker in nq100), "NQ"] = True
    metrics["SP500_decile"] = pd.qcut(
        metrics.operatingCashFlow[metrics.SP500], 10, labels=list(range(10, 0, -1))
    )
    metrics["NQ_decile"] = pd.qcut(
        metrics.operatingCashFlow[metrics.NQ], 10, labels=list(range(10, 0, -1))
    )

    # return calculation
    all_returns = []
    for ticker, df in metrics.groupby("symbol"):
        ticker_df = pd.read_csv(f"data/tickers/{ticker}.csv", parse_dates=[0])
        dates = pd.to_datetime(df.date)
        ticker_df.index = pd.to_datetime(ticker_df.Date).dt.date
        indices = [ticker_df.index.get_loc(date, method="nearest") for date in dates]
        closes = ticker_df.iloc[indices]["Close"]
        returns = (closes - closes.shift(1)) / closes.shift(1)
        returns.index = df.index
        all_returns.append(returns)
    all_returns = pd.concat(all_returns)
    metrics.loc[all_returns.index, "Total_Returns"] = all_returns
    metric_labels.append("Total_Returns")

    # custom scoring

    for metric in metric_labels:
        metrics.loc[:, f"{metric}_custom"] = pd.qcut(
            metrics[metric], 5, labels=list(range(1, 6))
        )
    # weighted score
    weighted_scores = metrics.apply(
        lambda row: sum([0.25 * row[f"{metric}_custom"] for metric in metric_labels]),
        axis=1,
    )
    metrics.loc[:, "grades"] = pd.qcut(
        weighted_scores.rank(pct=True) * 100, 5, labels=[5, 4, 3, 2, 1][::-1]
    )

    # portfolio analysis
    def calculate_returns(ticker, details):
        values = [np.nan, np.nan]
        try:
            df = pd.read_csv(f"data/tickers/{ticker}.csv", parse_dates=[0], index_col=0)
            dates = list(map(pd.to_datetime, [details.sold, details.added]))
            indices = [df.index.get_loc(date, method="nearest") for date in dates]
            prices = df.iloc[indices]["Close"].tolist()
            values = [prices[0] * details.qty, prices[1] * details.qty]
        except Exception as e:
            values = [np.nan, np.nan]
        finally:
            return pd.Series(values, index=["SP", "CP"])

    past = pd.read_csv("data/portfolio_past.csv", index_col=0)
    past.loc[:, ["SP", "CP"]] = past.apply(
        lambda row: calculate_returns(row.name, row), axis=1
    )
    past.loc[:, "PnL"] = past.SP - past.CP
    past.dropna(subset=["PnL"], inplace=True)
    original_profit = past.PnL.sum()

    # calculate amount spent on bottom 10% and distribute it evenly amongst top 10%
    bottom_10 = past.loc[past.PnL <= past.PnL.quantile(0.1)]
    amount = bottom_10.CP.sum()
    # assume the investing is done in the same time frame as the original stocks
    # and equally distributed amongst the top 10%
    top_10 = past.loc[past.PnL >= past.PnL.quantile(0.9)]
    costs = top_10.CP / top_10.qty
    each_stock_amt = amount / top_10.shape[0]
    qts_added = np.floor(each_stock_amt / costs)
    past.loc[top_10.index, "qty"] += qts_added
    past.loc[bottom_10.index, "qty"] = 0

    past.loc[:, ["SP", "CP"]] = past.apply(
        lambda row: calculate_returns(row.name, row), axis=1
    )
    past.loc[:, "PnL"] = past.SP - past.CP
    new_profit = past.PnL.sum()
    # 49.15028% more
    print(
        f"New Profit: {new_profit} | Original Profit: {original_profit} | % Change: {(new_profit - original_profit) / original_profit * 100}"
    )

    visualize(metrics)


if __name__ == "__main__":
    main()
