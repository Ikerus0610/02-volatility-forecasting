### Market data

The app uses Yahoo Finance through yfinance. It requests the daily Adj Close field for SPY and QQQ and stores downloaded responses in the Streamlit cache for one hour. Yahoo can revise adjusted historical prices. Each evaluation records the hash of the exact input. The original synthetic series is retained for reproducible tests. Public display rights depend on the applicable data agreement. [yfinance documentation](https://ranaroussi.github.io/yfinance/)
