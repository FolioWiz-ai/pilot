#!/bin/sh
# coding: utf-8
# brief: helper script to run the application
# author: pk13055

echo Using `python --version`
python -c 'import numpy as np, pandas as pd, streamlit as st;\
	print(f"Numpy: {np.__version__} | Pandas: {pd.__version__} | Streamlit: {st.__version__}")'
streamlit run app.py
