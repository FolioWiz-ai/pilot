# FolioWiz Pilot

> Exploratory US stock and equity data

## Installation

```bash
mkvirtualenv foliowiz --python 3.9.6
workon foliowiz
pip3 install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

## Dataset

1. FolioWiz stock universe (this represents about 1/8th of the total US stock market)
2. A sample portfolio (this is a virtual portfolio containing tickers and their buy/sell dates, quantity)

## Scope
1. Download and extract (for tickers in FolioWiz stock universe)
	a. 2 profile data (sector, industry)
	b. 4 metrics (operating cash flow, total returns, EV/EBITDA, D/E) - each individual metric falls into key buckets such as operating performance, returns, valuation, and risk

2. Compute
	a. quarterly operating cashflow growth rate for the last 3 years
	b. daily, monthly, quarterly total returns for the last 3 years

3. Process/Analyze
	a. Where does ticker’s metric sit with respect to its 3 year range? Assign quartile score (1-4 as in Top 25%ile ... Bottom 25%ile)
	b. What is the percentile rank of a ticker’s metric across all tickers in the superset (FolioWiz universe), sector, index (S&P500, NQ)? Assign decile score (1 - 10 as in Top 10%ile ... Bottom 10%ile) for each category.
	c. Develop a custom score of 1-5 for each metric using 4 range numbers (A, B, C, D; 5 if > A, 4 if between B and A, 3 if between C and B, 2 if between D and C, 1 if < D)
	d. Using weights for each metric, compute and rank all tickers using a weighted score based on 4 metrics (operating cash flow, total returns, EV/EBITDA, D/E)
	e. What if the sample portfolio’s bottom 10%ile stocks are replaced with top 10%ile of stocks (based on weighted score). In what ways can impact be measured? Illustrate with an example.

4. Visualize: Present the data using visualizations of your choice

5. _Optional_
	a. Develop / present correlation between metrics
	b. Use creativity to develop insights using additional / tangential data.
