# coding: utf-8
"""
    :brief: Foliowiz pilot data exploration
    :author: pk13055
"""
import os
import glob
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from process import (
    calc_custom_weights,
    calc_ocf_growth,
    calc_percentiles_deciles,
    download_candles,
    download_metrics,
    sector_index_decile,
)


@st.cache
def fetch_portfolio(filename: str) -> pd.DataFrame:
    """Fetch dataframe depending on file chosen"""
    df = pd.DataFrame([])
    try:
        df = pd.read_csv(filename).set_index("ticker")
        for _date_col in ["added", "sold"]:
            df[_date_col] = pd.to_datetime(df[_date_col])
    except Exception as e:
        st.info(f"[dataset read failure] {e}")
    return df


def prepare_universe():
    """Prepare the universe dataset for analysis"""

    """
    Following are some indices that are part of the universe:
    """
    indices = pd.Series(["IWM", "SPY", "ES", "VIX"])
    indices.name = "select_indices"
    st.table(indices)

    """The next step is to handle the ETF breakdown sheet. Here's a brief summary
    after cleaning of the dataset:"""

    breakdown_idxs = pd.read_csv("data/universe_index-breakdown.csv").set_index(
        "ticker"
    )
    st.dataframe(breakdown_idxs.groupby("source").describe())

    """The next dataset is a distribution of various tickers across sectors, across industries
    (_you can find a small snapshot of the dataset below_)"""
    goldmaster = pd.read_csv("data/universe_goldmaster.csv").set_index("ticker")
    st.table(goldmaster.head(15))

    with st.spinner("Loading Correlation Data ..."):
        """The final dataset as part of the `universe` is the `SUMMARY`.
        This contains a correlation matrix, of sorts."""
        (corr_df := pd.read_csv("data/universe_corr.csv").set_index("ticker"))

        corr_cols = list(corr_df.columns)
        st.info(f"""The above dataset contains the following columns: `{corr_cols}`""")

        """Given that this matrix has some informational (**as opposed to numeric**), 
        we can split the dataset into two, one the correlation matrix, and other the textual data."""
        corr_data = corr_df.loc[:, corr_cols[-5:]]
        corr_matrix = (
            corr_df.loc[:, corr_cols[:-5]].fillna(0.0).reset_index(drop=True).to_numpy()
        )
        """#### Correlation Dataset Snapshot"""
        st.table(corr_data.head())
        """#### Correlation Matrix"""
        st.dataframe(corr_matrix)

    return [indices, breakdown_idxs, goldmaster, corr_data, corr_matrix]


def prepare_portfolio() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the portfolio files for further use"""

    filename: str = "past"
    with st.spinner(text="Loading data ..."):
        (df := fetch_portfolio(f"data/portfolio_{filename}.csv"))
        st.info(f"Index Name: {df.index.name}")
        st.code(df.dtypes)
    other_filename = f"data/portfolio_curr.csv"
    st.success(
        f"Additionally, the [`{other_filename}`] dataset has been loaded as well, for further analysis."
    )
    if filename == "curr":
        past_df = fetch_portfolio(f"data/portfolio_past.csv")
        curr_df = df
    else:
        curr_df = fetch_portfolio(f"data/portfolio_curr.csv")
        past_df = df
    del df
    return curr_df, past_df


def calc_kpis(tickers: list):
    """Calculate KPI metrics"""
    kpi_df = pd.DataFrame([])
    if to_download := st.checkbox("Download metrics again?", False):
        if st.button("Start Download"):
            with st.spinner(text="Downloading metrics ..."):
                kpi_df = download_metrics(tickers)
    else:
        kpi_df = pd.read_csv("data/metrics_base.csv", parse_dates=[0])
    return kpi_df


@st.cache
def calc_total_returns(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate total returns for the specified dates"""
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
    return all_returns


def main():
    """Main function to run streamlit through"""
    st.title("FolioWiz Pilot - Data Exploration")

    """
    ## Data Cleaning
    
    ### `universe.xlsx`
    The first step is preparation of the universe dataset.
    By manually extracting the various pieces of information,
    it may save some time in automation.
    """
    idx_df, breakdown_df, desc_df, corr_data, corr_matrix = prepare_universe()
    tickers = desc_df.index.tolist()

    """
    ### `portfolio.xlsx`
    The next step is cleaning of the data. This involves reformatting of
    `portfolio.xlsx` as two separate datasets, namely, `current` and `past`.
    """
    curr, past = prepare_portfolio()

    """
    ## Data KPI Extraction

    Now that we have the data broken down into workable datasets, we can finally move onto the
    analysis of said data. The first step is to download the following KPI(s):
    - operating cash flow
    - total returns (calculated)
    - EV/EBITDA
    - D/E

    Except `total_returns`, all other metrics can be fetched from the FMP API
    """
    global API_TOKEN
    API_TOKEN = st.sidebar.text_input("API TOKEN", os.getenv("API_TOKEN", ""))
    metrics = calc_kpis(tickers)
    st.dataframe(metrics)

    """
    Now that we have the metrics, `operatingCashFlow`, `enterpriseValueMultiple` (EV/EBIDTA)
    and `debtEquityRatio` (D/E), we move on to calculating `Total_Returns`.
    For this, first we need to fetch the OHLC values for all the tickers
    for the past 3 years.
    """
    if st.checkbox("Download tickers again?", False):
        if st.button("Start Download"):
            with st.spinner(text="Downloading tickers ..."):
                download_candles(tickers)
    st.info(f"Tickers already available: {len(os.listdir('data/tickers/'))}")
    total_returns = calc_total_returns(metrics)
    metrics.loc[total_returns.index, "Total_Returns"] = total_returns
    st.dataframe(metrics)

    """
    ## Data Inference
    Next, we need to calculate the quarterly operating cashflow growth rate for the last 3 years.
    Here's a small sample of the calculated values:
    """
    metrics.loc[:, "ocf_growth"] = calc_ocf_growth(metrics)
    st.table(metrics.loc[:10, ["date", "symbol", "operatingCashFlow", "ocf_growth"]])

    metric_labels = [
        "operatingCashFlow",
        "enterpriseValueMultiple",
        "debtEquityRatio",
        "Total_Returns",
    ]
    f"""
    For the calculation of deciles, percentiles and quartiles, we have a
    fixed set of metrics, viz.
    `{metric_labels}`
    """
    metrics = calc_percentiles_deciles(metric_labels, metrics)
    st.dataframe(metrics)

    """
    Now, to calculate the decile ranks within a sector or within an index,
    we need to append the relevant information, ie, sector, index-presence to the metrics
    """
    metrics = sector_index_decile(metric_labels, metrics)
    st.dataframe(metrics)
    """
    Finally, we calculate custom scores for each metric and then a weighted
    score, aggregated across the custom scores.
    """
    metrics = calc_custom_weights(metric_labels, metrics)
    st.dataframe(
        metrics.loc[
            :,
            ["date", "symbol", "grades"]
            + [f"{metric}_custom" for metric in metric_labels],
        ]
    )


if __name__ == "__main__":
    main()
