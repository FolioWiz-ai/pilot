#!/usr/bin/env python3
# coding: utf-8
"""
    :brief: Foliowiz pilot data exploration
    :author: pk13055
"""
import os
import glob

import numpy as np
import pandas as pd
import streamlit as st


@st.cache
def fetch_data(filename: str) -> pd.DataFrame:
    """Fetch dataframe depending on file chosen"""
    df = pd.DataFrame([])
    try:
        df = pd.read_excel(filename)
    except Exception as e:
        st.info(f"[dataset read failure] {e}")
    return df


def main():
    """Main function to run streamlit through"""
    st.title("FolioWiz Pilot - Data Exploration")
    dataset_options = filter(
        lambda filename: filename.endswith("xlsx"), glob.glob("data/*")
    )
    filename: str = st.sidebar.selectbox("Choose dataset", list(dataset_options))

    df = fetch_data(filename)
    st.dataframe(df)


if __name__ == "__main__":
    main()
