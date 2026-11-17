### Real-data source review — 17 November 2026

Candidate: **Alpha Vantage, TIME_SERIES_DAILY_ADJUSTED**. Its public [API documentation](https://www.alphavantage.co/documentation/#dailyadj) lists adjusted close, dividend and split fields, JSON/CSV output, an API key requirement and premium access. A full-history response is needed; 100 observations are insufficient for the default study.

The published [terms](https://www.alphavantage.co/terms_of_service/) grant personal, non-commercial use unless otherwise agreed in writing. A paid endpoint is not evidence of permission to redistribute a history in a public repository or app. Such rights have not been obtained for this project.

**Verification scope:** official documentation and terms were accessible and reviewed. An authenticated SPY/QQQ premium download was not tested: no user key or subscription was supplied. Consequently v0.1 includes no downloader and redistributes no vendor history. The application makes no external data requests.

To use authorized data, obtain completed dividend/split-adjusted daily closes from your provider, keep the source/retrieval date in your research notes, map adjusted-close columns to `date,SPY,QQQ`, and upload privately. Use one adjusted price convention consistently. Do not substitute raw closing prices. The app cannot certify adjustments or licensing from a CSV.

Bundled `synthetic_prices.csv` is locally generated artificial data (seed 20261117), not Alpha Vantage data. The small template also contains invented illustrative prices. On a hosted Streamlit instance, uploaded bytes are sent to the app server and retained temporarily in memory/cache; they are not committed or written to disk by the app. Use local execution if uploading provider data is not permitted.
