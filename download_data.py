#!/usr/bin/env python3
"""
    :author: pk13055 :brief: Download the OHLC and metrics for a given list of tickers
"""
import argparse
import os

import pandas as pd
import requests
import yfinance as yf

API_TOKEN = os.getenv("API_TOKEN", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=str, default="data/universe.csv",
                        help="universe of tickers to download")
    parser.add_argument("--token", type=str, default=os.getenv("API_TOKEN"),
                        help="Token for FMP API access")
    args = parser.parse_args()
    return args


def download_candles(tickers: list) -> None:
    """Download OHLC from Yahoo Finance given list of tickers"""
    data = yf.download(
            tickers=" ".join(tickers),
            period="3y",
            group_by="ticker")
    for ticker in tickers:
        ticker_df = data[ticker].sort_index()
        ticker_df.to_csv(f"data/tickers/{ticker}.csv", float_format="%.2f")


def get_raw_data(ticker: str) -> pd.DataFrame:
    """Download and partially parse the raw data"""
    BASE_CASH_FLOW = "https://financialmodelingprep.com/api/v3/cash-flow-statement/"
    BASE_RATIOS = "https://financialmodelingprep.com/api/v3/ratios/"

    cashflow = requests.get(f"{BASE_CASH_FLOW}{ticker}", params={
        "apikey": API_TOKEN,
        "period": "quarter",
        "limit": 13
    }).json()
    ratios = requests.get(f"{BASE_RATIOS}{ticker}", params={
        "apikey": API_TOKEN,
        "period": "quarter",
        "limit": 13
    }).json()

    cashflow = pd.DataFrame(cashflow)
    ratios = pd.DataFrame(ratios)
    cashflow = cashflow.loc[:, ["date", "period", "operatingCashFlow"]]
    ratios = ratios.loc[:, ["date", "period", "enterpriseValueMultiple", "debtEquityRatio"]]
    cashflow.date = pd.to_datetime(cashflow.date)
    ratios.date = pd.to_datetime(ratios.date)
    # filtering data only for the last 3 years
    cashflow = cashflow.loc[(cashflow.date >= "01-01-2019")
                            & (cashflow.date <= "31-12-2022")]
    ratios = ratios.loc[(ratios.date >= "01-01-2019")
                            & (ratios.date <= "31-12-2022")]
    cashflow.index = cashflow.apply(lambda row: f"{row.period}_{row.date.year}", axis=1)
    ratios.index = ratios.apply(lambda row: f"{row.period}_{row.date.year}", axis=1)
    del cashflow['date'], cashflow['period'], ratios['date'], ratios['period']
    metrics = pd.concat([cashflow, ratios], axis=1)
    metrics.columns = [f"{ticker}_{metric}" for metric in list(metrics.columns)]
    return metrics



def download_metrics(tickers: list) -> pd.DataFrame:
    """Download operating cash flow, EV/EBITDA, D/E for tickers"""
    count = 0
    frames = []
    for ticker in tickers:
        data = get_raw_data(ticker)
        frames.append(data)
        count += 1
        if count == 3:
            break
    df = pd.concat(frames, axis=1)
    print(df)


def main():
    args = parse_args()

    univ = pd.read_csv(args.universe)
    tickers = univ.ticker.tolist()

    # download and save ohlc candles for last 3y to `data/tickers/`
    # download_candles(tickers)
    # download and save metrics
    download_metrics(tickers)

if __name__ == "__main__":
    main()
