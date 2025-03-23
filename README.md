# Commodities Modelling
This repository contains a small useful class for modelling of commodity pricing, 
focusing on time-series forecasting - with and without covariates.

The main class is called [`Commodities`](CommoditiesClass.py) that facilitates data ingestion, preprocessing, 
and modeling for a wide range of financial and economic data and many different types of models.

The `Commodities` class serves as a data preparation and ingestion pipeline tailored for financial market and macroeconomic analysis:
- Downloads or loads data for a wide range of commodities, markets, and indices.
- Combines data systematically for analysis, such as backtesting, forecasting, or machine learning tasks.
- Provides flexibility to preprocess and process data for further modeling and evaluation.
- Performs modelling of various time-series models using [darts](https://unit8.com/darts-open-source/), including backtesting and evaluation.

# Instructions
Most parameters of interest are set in the `__init__` initialisation method. This method initializes the class with 
default values for attributes like dates, thresholds, flags, and paths. It also prepares external the libraries such as
[Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/) API for economic data, 
[`yfinance`](https://pypi.org/project/yfinance/) API for futures, market data, fx rates and US treasury yields 
and checks or creates a directory to save/export data.

## Dates
The dates used as bounds for collecting time-series data are specified in the `__init__` method:
- **Time-series' start (`start_date`)**: The starting date for data collection.
- **Time-series' end (`end_date`)**: The ending date for data collection.
- **Time-series' training/validation split date (`split_date`)**: Used to split data into training and validation or to separate historical and recent data.

## Data ingestion
Data ingestion is performed using the method `data_ingestion` that prepares a combined pandas DataFrame stored in `self.df_raw` 
from the data sources specified in the following dictionaries:
encoded as pandas DataFrames:
- **Commodities (`dict_commodities`)**: Contains commodity codes and their descriptions from FRED (e.g., crude oil prices, orange juice prices).
- **Futures (`dict_futures`)**: Futures prices data from Yahoo Finance (e.g., S&P, crude oil, gold).
- **Markets (`dict_markets`)**: Financial market indices (e.g., Nasdaq, Dow Jones).
- **Forex (`dict_fx`)**: Exchange rate data such as USD/JPY or EUR/JPY.
- **Treasury Yields (`dict_treasury_yields`)**: US treasury yield rates.
- **Macroeconomic Data (`dict_macroeconomic`)**: Key macroeconomic indicators (e.g., GDP, CPI) from FRED.

### Macroeconomic data ingestion
An instance of the `Fred` API (from the `fredapi` package) to collect data from FRED. 
The API key must be obtained from there and specified in the `__init__` method in the line `self.api_fred = Fred(api_key)`.
### yfinance
`yfinance` is used to fetch financial data (e.g., stock, futures, indices) from Yahoo Finance.
No API key must be provided

## Data processing
The data is preprocessed using the `data_preprocessing`, `data_mv_imputation` and `create_ts_datasets` methods, which ensures
- Normalised data, i.e. ensure dates are business days.
- Missing values are imputed in a reasonable way by filling them  with the most recent available past value.
- Time-series objects for the modelling step is prepared.

## Modelling
The modelling is performed through several methods, depending on the aim:
- `fit_evaluate`: Fit on ts_train and validate on ts_val with the given metric. If not provided, the time series in the object will be used. 
This method does *not* give the historical forecast; it gives the prediction of the model given the training data.
- `historical_forecasts_evaluate`: Performs historical forecasts training the model on `look_back_days` of data, 
subsequently moving forward `stride_days` at a time, predicting the time series `lookahead_days` in the future. 
- `backtest_evaluate`: Performs a backtest of the model using historical forecasts for the model and evaluating the model outputs against the validation data.
- `batch_backtesting`: Run backtests for many different models and aggregate the results.
