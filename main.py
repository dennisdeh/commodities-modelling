import pandas as pd
import darts
from CommoditiesClass import Commodities
import darts.models as darts_models
import darts.dataprocessing.transformers as dpp
import sklearn.preprocessing as pp
import datetime
import matplotlib.pyplot as plt

# %% 0: Initialisation
# instantiate Commodities class
com = Commodities()
# define parameters for single backtests
target_col = "DCOILBRENTEU"  # Brent oil price
target_col2 = "DCOILWTICO"  # WTI oil
look_back_days = 2000
lookahead_days = 70
stride_days = 100
# dictionary to store backtest results in
dict_bt = {}

# %% 1: Prepare for modelling
# 1.1: Data ingestion
com.data_ingestion(load=False, save=True)
# 1.2: Preprocessing the data
com.data_preprocessing(method_duplicates="first_non_nan", fill_method="ffill")
# 1.3: Investigating and imputing remaining missing values
com.data_mv_imputation(imputation_strategy="simple", load=False, save=True)
# 1.4: Create (transformed) time series data sets for training and validation
com.create_ts_datasets(
    transformer=dpp.Scaler(scaler=pp.StandardScaler()),
    features=None,
    plot=True,
    plot_max_nr_components=3,
    save_plot=True,
    show_plot=True,
)
"""
Overview of darts models:
https://github.com/unit8co/darts#forecasting-models
"""
# %% 2: Modelling: Uni-variate models
# 2.1: ARIMA
"""
Autoregressive integrated moving average models extensible with exogenous variables 
(future covariates) and seasonal components.
p: Order (number of time lags) of the autoregressive model (AR).
d: The order of differentiation
q: The size of the moving average window (MA).
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA
"""
# define parameters and the model object
model = darts_models.ARIMA(p=5, d=0, q=1)
# Backtest
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=None,
    future_covariates_cols=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# 2.2: AutoARIMA
"""
Automatically discover the optimal order for an ARIMA model.
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html#darts.models.forecasting.auto_arima.AutoARIMA
"""
# define parameters and the model object
model = darts_models.AutoARIMA()
# Backtest
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=None,
    future_covariates_cols=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# 2.3: Exponential Smoothing
"""
Holt-Winters’ exponential smoothing model.
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html#darts.models.forecasting.auto_arima.AutoARIMA
"""
# define parameters and the model object
model = darts_models.ExponentialSmoothing()
# Backtest
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=None,
    future_covariates_cols=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt


# %% 3: Multivariate models
# 3.1: NaiveMovingAverage
"""
This model forecasts using an autoregressive moving average (ARMA).
target: target + target2
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html#darts.models.forecasting.auto_arima.AutoARIMA
"""
# define parameters and the model object
model = darts_models.NaiveMovingAverage(input_chunk_length=10)
# Backtest
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=[target_col, target_col2],
    past_covariates_cols=None,
    future_covariates_cols=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt


# %% 4: Standard Machine-learning models
# 4.1: Linear Regression model
"""
Linear regression model with co-variates.
target: target
co-variates: Several
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html#darts.models.forecasting.linear_regression_model.LinearRegressionModel
"""
# define parameters and the model object
model = darts_models.LinearRegressionModel(lags=100, lags_past_covariates=100)
# Backtest
past_covariates_cols = list(set(com.ts_train.columns).difference(target_col))
future_covariates_cols = None
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=past_covariates_cols,
    future_covariates_cols=future_covariates_cols,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# 4.2: XGBModel
"""
This is a XGBModel implementation of Gradient Boosted Trees algorithm.
This implementation comes with the ability to produce probabilistic forecasts.
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel
"""
# define parameters and the model object
model = darts_models.XGBModel(lags=300, lags_past_covariates=300)
# Backtest
past_covariates_cols = list(set(com.ts_train.columns).difference(target_col))
future_covariates_cols = None
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=past_covariates_cols,
    future_covariates_cols=future_covariates_cols,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# 4.3: LigthGBM
"""
This is a LightGBM implementation of Gradient Boosted Trees algorithm.
This implementation comes with the ability to produce probabilistic forecasts.
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html#darts.models.forecasting.lgbm.LightGBMModel
"""
# define parameters and the model object
model = darts_models.XGBModel(lags=300, lags_past_covariates=300)
# Backtest
past_covariates_cols = list(set(com.ts_train.columns).difference(target_col))
future_covariates_cols = None
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=past_covariates_cols,
    future_covariates_cols=future_covariates_cols,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# %% 5: Deep machine-learning models
# 5.1: Recurrent Neural Networks
"""
This class provides three variants of RNNs: Vanilla RNN, LSTM, GRU
target: target
co-variates: None
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel
"""
# define parameters and the model object
model = darts.models.RNNModel(
    model="LSTM",
    input_chunk_length=200,
    hidden_dim=25,
    n_rnn_layers=2,
    training_length=240,
)
# Backtest
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=None,
    future_covariates_cols=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# 5.2: Transformer Model
"""
It is an encoder-decoder architecture whose core feature is the ‘multi-head attention’
 mechanism, which is able to draw intra-dependencies within the input vector and 
 within the output vector (‘self-attention’) as well as inter-dependencies between 
 input and output vectors (‘encoder-decoder attention’)
target: target
co-variates: Several
reference:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html#darts.models.forecasting.rnn_model.RNNModel
"""
# define parameters and the model object
model = darts.models.TransformerModel(
    input_chunk_length=200,
    output_chunk_length=1,
    output_chunk_shift=0,
    n_epochs=100,
    d_model=64,
    nhead=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    activation="relu",
    norm_type=None,
    custom_encoder=None,
    custom_decoder=None,
)
# Backtest
past_covariates_cols = list(set(com.ts_train.columns).difference(target_col))
future_covariates_cols = None
d_bt = com.backtest_evaluate(
    model=model,
    target_cols=target_col,
    past_covariates_cols=past_covariates_cols,
    future_covariates_cols=future_covariates_cols,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    save_plot=True,
    show_plot=True,
)
# add to evaluation dictionary
dict_bt[str(datetime.datetime.now())] = d_bt

# %% Y: Futures models
"""
Here the last available future price is taken as the prediction:
"""

# %% Run many backtests:
models = [darts_models.ARIMA(p=2, d=0, q=0), darts_models.ARIMA(p=5, d=0, q=0)]
look_back_days = [50, 200]
lookahead_days = [20, 50]
stride_days = [20, 50]
com.batch_backtesting(
    models=models,
    target_cols=target_col,
    past_covariates_cols=None,
    future_covariates_cols=None,
    ts_train=None,
    ts_val=None,
    look_back_days=look_back_days,
    lookahead_days=lookahead_days,
    stride_days=stride_days,
    transformer=None,
    ts_are_transformed=True,
    inverse_transform=True,
    save_plot=True,
    show_plot=True,
)

pd.DataFrame(dict_bt).T.to_excel("backtesting_results.xlsx")
