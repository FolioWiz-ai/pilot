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

    portfolio_options = set(["past", "curr"])
    filename: str = st.sidebar.selectbox(
        "Choose Portfolio", list(portfolio_options), index=1
    )
    with st.spinner(text="Loading data ..."):
        (df := fetch_portfolio(f"data/portfolio_{filename}.csv"))
        st.info(f"Index Name: {df.index.name}")
        st.code(df.dtypes)
    other_filename = f"data/portfolio_{next(iter(portfolio_options - {filename}))}.csv"
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


def calc_kpis():
    """Calculate KPI metrics"""

    "> TODO: add calc of metrics here"
    kpi_df = pd.DataFrame([])
    return kpi_df


def calc_cfg_returns():
    """Calculate CFG Returns"""

    "> TODO: calculate qtrly CGR and daily monthly and qtrly TOTAL RETURNS"
    cfg_df = pd.DataFrame([])
    return cfg_df


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

    """
    ### `portfolio.xlsx`
    The next step is cleaning of the data. This involves reformatting of
    `portfolio.xlsx` as two separate datasets, namely, `current` and `past`.
    """
    curr, past = prepare_portfolio()

    """
    ## Data KPI Extraction

    Now that we have the data broken down into workable datasets, we can finally move onto the
    analysis of said data. The first step is to break down the data into the following KPI(s):
    - operating cash flow
    - total returns
    - EV/EBITDA
    - D/E
    """

    kpi_df = calc_kpis()
    st.dataframe(kpi_df)

    """
    Next, we need to calculate
    - the quarterly operating cashflow growth rate for the last 3 years
    - daily, monthly and quarterly total returns for the last 3 years
    """

    returns_df = calc_cfg_returns()
    st.dataframe(returns_df)

    """
    ## Data Inference
    > TODO: get example for clarity
    """


if __name__ == "__main__":
    main()
