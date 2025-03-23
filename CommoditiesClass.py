import os
from typing import Any, Union
import yfinance as yf
from darts import TimeSeries
from fredapi import Fred
import numpy as np
import pandas as pd
import datetime
import darts
import darts.models as darts_models
import darts.metrics as darts_metrics
import darts.dataprocessing.transformers as dpp
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
import itertools

from utils.normalise_datetime_index import dt_normalise


class Commodities:
    """
    Commodities class
    """

    def __init__(self, path: Union[None, str] = None):
        """
        Initialise the Commodities class
        """
        # dates
        self.start_date = "2001-01-02"
        self.end_date = "2023-12-31"
        self.split_date = "2020-01-02"
        # data preprocessing
        self.threshold_nan = 0.05
        # data
        self.df_raw = pd.DataFrame()  # raw data
        self.df_pp = pd.DataFrame()  # preprocessed data
        self.df_mv_impute = pd.DataFrame()  # missing values imputed data
        self.ts_train = None
        self.ts_val = None
        # transformer
        self.transformer = None
        # evaluation
        self.dict_bt = {}
        # apis and API keys
        self.api_fred = Fred(api_key="YOUR FRED API KEY HERE")
        # paths
        if path is None:
            self.path = os.path.join(os.getcwd(), "export")
        else:
            self.path = os.path.abspath(path)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        else:
            pass
        # flags
        self.flag_raw_data = False
        self.flag_pp_data = False
        self.flag_mv_impute_data = False
        self.flag_ts_data = False

    # %% Data
    def data_ingestion(self, save: bool = False, load: bool = False):
        """
        Defines and downloads the data needed for modelling.
        """
        if load:
            print("Loading raw data from existing file... ", end="")
            self.df_raw = pd.read_parquet("commodities_raw.parquet")
            self.flag_raw_data = True
            print("Success!")
        else:
            print("Creating new raw data set:")
            # 1.1: define symbols
            # commodities (FRED)
            dict_commodities = {
                "DCOILBRENTEU": "Crude Oil Prices: Brent - Europe",
                "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI) CO",
                "POLVOILUSDM": "Global price of Olive Oil",
                "APU0000713111": "Average Price: Orange Juice, Frozen Concentrate, 12 Ounce Can (Cost per 16 Ounces/473.2 Milliliters) in U.S. City Average",
            }
            # futures (yfinance)
            dict_futures = {
                "ES=F": "S&P futures",
                "YM=F": "Dow futures",
                "NQ=F": "Nasdaq futures",
                "RTY=F": "Russell 2000 futures",
                "NIY=F": "NIKKEI futures",
                "ZB=F": "US treasury bond futures",
                "ZN=F": "10-year T-Note futures",
                "ZF=F": "5-year US treasury note",
                "ZT=F": "2-year US treasury note",
                "CL=F": "crude oil",
                "GC=F": "gold",
                "SI=F": "silver",
                "PL=F": "platinum",
                "HG=F": "copper",
                "PA=F": "palladium",
                "NG=F": "natural gas",
                "RB=F": "gasoline",
                "ZC=F": "corn",
                "ZO=F": "oat",
                "KE=F": "wheat",
                "ZR=F": "rice",
                "ZS=F": "soybean",
                "GF=F": "feeder cattle",
                "LE=F": "live cattle",
                "CC=F": "cocoa",
                "KC=F": "coffee",
                "CT=F": "cotton",
                "OJ=F": "orange juice",
                "SB=F": "suger",
            }
            # markets (yfinance)
            dict_markets = {
                "^SOX": "SOX index",
                "^VIX": "VIX index",
                "^XAX": "NY AMEX Composite Index",
                "^FCHI": "CAC 40",
                "^GSPC": "SP500 index",
                "^DJI": "Dow index",
                "^IXIC": "Nasdaq index",
                "^RUT": "Russell index",
                "^GSPE": "S&P 500 Energy Sector",
                "^SP500-20": "S&P 500 Industrials Sector",
                "^SP500-30": "S&P 500 Consumer Staples Sector",
                "^SP500-40": "S&P 500 Financials Sector",
                "^SP500-50": "S&P 500 Communication Services",
                "^SP500-15": "S&P 500 Materials Sector",
                "^SP500-25": "S&P 500 Consumer Discretionary",
                "^SP500-35": "S&P 500 Health Care Sector",
                "^SP500-45": "S&P 500 Information Technology Sector",
                "^SP500-55": "S&P 500 Utilities Sector",
                "^BANK": "NASDAQ Bank",
                "^IXCO": "NASDAQ Computer",
                "^NBI": "NASDAQ Biotechnology",
                "^NDXT": "NASDAQ 100 Technology",
                "^INDS": "NASDAQ Industrial",
                "^INSR": "NASDAQ Insurance",
                "^OFIN": "NASDAQ Other Finance",
                "^IXTC": "NASDAQ Telecommunications",
                "^TRAN": "NASDAQ Transportation",
            }
            # FX (yfinance)
            dict_fx = {
                "USDJPY=X": "USD/JPY",
                "EURJPY=X": "EUR/JPY",
                "AUDJPY=X": "AUD/JPY",
            }
            # treasury yields (yfinance)
            dict_treasury_yields = {
                "^IRX": "Treasury Yield 13 Weeks",
                "^FVX": "Treasury Yield 4 Years",
                "^TNX": "Treasury Yield 10 Years",
                "^TYX": "Treasury Yield 30 Years",
            }
            # macroeconomic data (FRED)
            dict_macroeconomic = {
                "GDP": "US Gross Domestic Product (quarterly)",
                "CPMNACSCAB1GQEU272020": "Gross Domestic Product for European Union (27 Countries from 2020)",
                "CORESTICKM159SFRBATL": "Sticky Price Consumer Price Index less Food and Energy (monthly)",
                "PPIACO": "Producer Price Index by Commodity: All Commodities (monthly)",
            }

            # 1.2: download data (where relevant: close prices are used)
            dict_data = {}
            # commodities (FRED)
            dict_commodities_data = {
                key: self.api_fred.get_series(key) for key in dict_commodities.keys()
            }
            dict_data["df_commodities"] = pd.DataFrame(dict_commodities_data)
            # futures (yfinance)
            df_futures = yf.download(
                list(dict_futures.keys()),
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                threads=True,
            )
            dict_data["df_futures"] = df_futures["Close"]
            # markets (yfinance)
            df_markets = yf.download(
                list(dict_markets.keys()),
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                threads=True,
            )
            dict_data["df_markets"] = df_markets["Close"]
            # FX (yfinance)
            df_fx = yf.download(
                list(dict_fx.keys()),
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                threads=True,
            )
            dict_data["df_fx"] = df_fx["Close"]
            # treasury yields (yfinance)
            df_treasury_yields = yf.download(
                list(dict_treasury_yields.keys()),
                start=self.start_date,
                end=self.end_date,
                interval="1d",
                threads=True,
            )
            dict_data["df_treasury_yields"] = df_treasury_yields["Close"]
            # macroeconomic data (FRED)
            dict_macroeconomic_data = {
                key: self.api_fred.get_series(key) for key in dict_macroeconomic.keys()
            }
            dict_data["df_macroeconomic"] = pd.DataFrame(dict_macroeconomic_data)

            # combine data and save
            df = pd.concat(dict_data.values(), axis="columns", join="outer")
            self.df_raw = df.sort_index(ascending=True)
            if save:
                print("Saving raw data to commodities_raw.parquet")
                df.to_parquet("commodities_raw.parquet", index=True)
            self.flag_raw_data = True
            print("Success!")

    def data_preprocessing(
        self, method_duplicates: str = "first_non_nan", fill_method: str = "ffill"
    ):
        """
        Normalise data, i.e. ensure dates are business days.
        Missing values are imputed in a reasonable way by filling them
        with the most recent available past value.
        Data is filtered to the period between start and end dates.
        """
        print("Preprocessing data:")
        assert self.flag_raw_data, "raw data must be loaded first"
        # normalise
        df = dt_normalise(
            self.df_raw,
            method_duplicates=method_duplicates,
            fill_method=fill_method,
            sort_ascending=True,
        )
        # filter dates to the interval of interest
        mask = (pd.Timestamp(self.start_date) <= df.index) & (
            pd.Timestamp(self.end_date) >= df.index
        )
        self.df_pp = df[mask]
        self.flag_pp_data = True
        print("Success!")

    def data_mv_imputation(
        self,
        imputation_strategy: str = "simple",
        save: bool = False,
        load: bool = False,
    ):
        """
        There are in general still missing values even after a forward fill
        because some data series did not start at the time where the modelling
        starts as defined by date_start.

        Hence we need to pay special attention to these.

        In the current development we just drop any column that has more
        than threshold_nan missing values.
        At a 5% threshold there are might be a few columns left with missing values: We check
        whether all these missing values are before the split date and if yes, then we do a backfill.

        We check afterwards if there are additional missing values in the validation data: in that
        case we need to do something else to the missing values in the validation data: For now
        the columns are just dropped.

        TODO:
            - sklearn impute should be implemented to deal with date where there are
                still missing values after a forward fill, but below the threshold
        :return:
        """
        print("Investigating and imputing remaining missing values:")
        if load:
            print("Loading missing value imputed data from existing file... ", end="")
            self.df_mv_impute = pd.read_parquet("commodities_processed.parquet")
            assert (
                self.df_mv_impute.isna().sum().sum() == 0
            ), "there are missing values!"
            self.flag_mv_impute_data = True
            print("Success!")
        else:
            print(f"   Threshold for dropping columns: {self.threshold_nan}")
            assert self.flag_pp_data, "pp data must be loaded first"
            # counting the missing values and see what columns are above the threshold
            df_count_nan = self.df_pp.isnull().sum() / len(self.df_pp)
            cols_to_drop = df_count_nan[
                df_count_nan > self.threshold_nan
            ].index.to_list()
            df = self.df_pp.drop(columns=cols_to_drop)
            print(
                f"Dropping {len(cols_to_drop)} columns that have too many missing values."
            )

            # impute missing values for remaining columns
            df_count_nan = df.isnull().sum() / len(df)
            cols_to_impute = df_count_nan[df_count_nan > 0.0].index.to_list()
            print(f"There are {len(cols_to_impute)} columns left with missing values:")
            if imputation_strategy == "simple":
                for col in cols_to_impute:
                    # get last day
                    last_date_nan = max(df[col][df[col].isna()].index)
                    if last_date_nan > pd.Timestamp(self.split_date):
                        print(
                            f"{col}: Missing values in the training data only, imputing by a "
                            f"backfill from most recent future values"
                        )
                        df[col] = df[col].copy().bfill()
                    else:
                        print(
                            f"{col}: Dropped because there are missing values in the validation data"
                        )
                        df = df.drop(columns=col)
            else:
                raise NotImplementedError("Imputation strategy is not yet implemented")

            # check that there are no more missing values
            assert df.isna().sum().sum() == 0, "there are still missing values!"
            self.df_mv_impute = df
            self.flag_mv_impute_data = True
            if save:
                print(
                    "Saving missing value imputed data to commodities_processed.parquet"
                )
                df.to_parquet("commodities_raw.parquet", index=True)
            print("Success!")

    def create_ts_datasets(
        self,
        transformer: Any,
        features: Union[list, None] = None,
        plot: bool = True,
        plot_max_nr_components: int = 3,
        save_plot: bool = True,
        show_plot: bool = True,
    ):
        """
        Create training and validation time series data sets.

        Transforming the data if a transformer is given: The
        transformer is only trained on the training data and the
        fitted object is stored.
        """
        print("Creating training and validation time series data sets:")
        if transformer is None:
            print("Data will not be transformed")
        else:
            print("Transformer: ")
        assert (
            self.flag_mv_impute_data
        ), "missing value imputed data must be loaded first"
        # create time series object
        ts = darts.TimeSeries.from_dataframe(
            df=self.df_mv_impute,
            time_col=None,  # the df index is time
            value_cols=features,  # use all columns if None
            fill_missing_dates=False,  # Ensure not to fill missing dates
            freq="B",  # Business day frequency
        )
        # split into training and validation data
        self.ts_train, self.ts_val = ts.split_after(pd.Timestamp(self.split_date))
        self.ts_train.transformed = False
        self.ts_val.transformed = False

        # plot some time series before the transformation
        if plot:
            fig = plt.figure(figsize=(18, 10))
            fig.suptitle("Untransformed time series", fontsize=16)
            self.ts_train.plot(
                label="training",
                max_nr_components=plot_max_nr_components,
                linewidth=1.0,
            )
            self.ts_val.plot(
                label="validation",
                max_nr_components=plot_max_nr_components,
                linewidth=1.0,
            )
            if save_plot:
                plt.savefig(os.path.join(self.path, "data_untransformed.png"))
            if show_plot:
                plt.show()
            plt.close()

        # transforming
        if not (transformer is None):
            self.transformer = transformer.fit(self.ts_train)
            self.ts_train = transformer.transform(self.ts_train)
            self.ts_val = transformer.transform(self.ts_val)
            self.ts_train.transformed = True
            self.ts_val.transformed = True

            # plot some time series after the transformation
            if plot:
                fig = plt.figure(figsize=(18, 10))
                fig.suptitle(
                    f"Transformed time series (transformer: {transformer.__str__()})",
                    fontsize=16,
                )
                self.ts_train.plot(
                    label="training",
                    max_nr_components=plot_max_nr_components,
                    linewidth=1.0,
                )
                self.ts_val.plot(
                    label="validation",
                    max_nr_components=plot_max_nr_components,
                    linewidth=1.0,
                )
                if save_plot:
                    plt.savefig(os.path.join(self.path, "data_transformed.png"))
                if show_plot:
                    plt.show()
                plt.close()
        # set flag
        self.flag_ts_data = True

    # %% Modelling
    """
    Below is a short description of the methods used in time-series modelling:
    - Fit and evaluate:
    Train a model on a given data set and make predictions using a given validation data.

    - Historical forecasting and backtests:
    Historical forecasting simulates predictions that would have been obtained historically with a 
    given model. It can take a while to produce, since the model is re-trained every time the 
    simulated prediction time advances.
    """

    def fit_evaluate(
        self,
        model: Any,
        target_cols: [str, list],
        past_covariates_cols: Union[None, list],
        future_covariates_cols: Union[None, list],
        look_back_days: Union[None, int],
        ts_train: Union[darts.TimeSeries, None] = None,
        ts_val: Union[darts.TimeSeries, None] = None,
        metric: Any = darts_metrics.mape,
        save_plot: bool = True,
        show_plot: bool = True,
    ):
        """
        Fit on ts_train and validate on ts_val with the given metric.
        If not provided, the time series in the object will be used.

        This method does _not_ give the historical forecast; it gives the
        prediction of the model given the training data, for n=len(ts_val)
        future periods.

        TODO:
            Make sure that the future/past covariates are only passed
            when the models supports them.

        :param model: Model to fit and evaluate
        :param target_cols: List or str
            The target column (uni-variate) or columns (multivariate) to use in
            the model.
        :param past_covariates_cols: None or Darts TimeSeries
            Past covariates to feed to the model.
        :param future_covariates_cols: None or Darts TimeSeries
            Future covariates to feed to the model.
        :param look_back_days: If None, the entire time series ts_train
            will be used for training.
            If an integer, only the last
            look_back_days periods will be used.
        :param ts_train: Data to train model on.
            If None, the data in the object will be used.
        :param ts_val: Validation data to compare against.
            If None, the data in the object will be used.
        :param metric: Metric to use for evaluating the model on the
            validation data.
        :param save_plot: Whether to save the plot or not.
        :param show_plot: Whether to show the plot or not.
        :return: Dictionary of metric values
        """
        print(f"Fitting and evaluating {model.__str__()}... ", end="")
        # prepare data to use
        if ts_train is None:
            ts_train = self.ts_train
        elif isinstance(ts_train, darts.TimeSeries):
            pass
        else:
            raise ValueError(
                "ts_train must be an instance of darts.TimeSeries if given"
            )
        if ts_val is None:
            ts_val = self.ts_val
        elif isinstance(ts_val, darts.TimeSeries):
            pass
        else:
            raise ValueError(
                "ts_train must be an instance of darts.TimeSeries if given"
            )
        if look_back_days is None:
            pass
        elif isinstance(look_back_days, int) and look_back_days > 0:
            ts_train = ts_train[-look_back_days:]
        else:
            raise ValueError("look_back_days must be an integer or None")

        # prepare columns
        if past_covariates_cols is None:
            past_covariates = None
            str_past_covariates = "None"
        elif isinstance(past_covariates_cols, list):
            past_covariates = ts_train[past_covariates_cols]
            str_past_covariates = len(past_covariates_cols)
        else:
            raise ValueError("past_covariates_cols must be a list or None")
        if future_covariates_cols is None:
            future_covariates = None
            str_future_covariates = "None"
        elif isinstance(future_covariates_cols, list):
            future_covariates = ts_train[future_covariates_cols]
            str_future_covariates = len(future_covariates_cols)
        else:
            raise ValueError("past_covariates_cols must be a list or None")
        if isinstance(target_cols, str) or isinstance(target_cols, list):
            ts_train_target = ts_train[target_cols]
        else:
            raise ValueError("target_cols must be a str or list")

        # prepare model name string
        str_name = model.__str__().split("\n")[0]
        if str_name[-1] == ",":
            str_name = f"{str_name} ...)"

        # fit the model on the appropriate data (taking the last look_back_days days)
        model.fit(
            ts_train_target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        # predict and compare
        ts_predictions = model.predict(n=len(ts_val))
        # calculate metric
        d = {model.__str__(): metric(ts_val, ts_predictions)}
        # plot
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"{str_name}: Fit and evaluation", fontsize=16)
        ts_train_target.plot(label="training", linewidth=1.0)
        ts_val.plot(label="validation")
        ts_predictions.plot(label="prediction")

        if save_plot:
            plt.savefig(os.path.join(self.path, f"evaluation_{model.__str__()}.png"))
        if show_plot:
            plt.show()
        plt.close()
        print("Success!")
        return d

    def historical_forecasts_evaluate(
        self,
        model: Any,
        target_cols: [str, list],
        past_covariates_cols: Union[None, list],
        future_covariates_cols: Union[None, list],
        look_back_days: Union[None, int],
        lookahead_days: int,
        stride_days: int,
        ts_train: Union[darts.TimeSeries, None] = None,
        ts_val: Union[darts.TimeSeries, None] = None,
        transformer: Union[None, Any] = None,
        ts_are_transformed: bool = False,
        inverse_transform: bool = False,
        num_samples: int = 1,
        save_plot: bool = True,
        show_plot: bool = True,
    ):
        """
        Performs historical forecasts training the model on look_back_days of
        data, subsequently moving forward stride_days at a time, predicting the
        time series lookahead_days in the future.

        Initially, the last look_back_days periods of ts_train are used, but as
        the forecasting progresses, it will use more and more data from the
        validation time series. If no time series are provided, the ones stored in the object

        The forecasted time series using this setup is returned. The time series
        can be inversely transformed when the transformer is given.

        :param model: Darts model
            The model to use for historical forecasts.
        :param target_cols: List or str
            The target column (uni-variate) or columns (multivariate) to use in
            the model.
        :param past_covariates_cols: None or Darts TimeSeries
            Past covariates to feed to the model.
        :param future_covariates_cols: None or Darts TimeSeries
            Future covariates to feed to the model.
        :param look_back_days: None or int
            If None, the entire length of the time series ts_train will be used
            for training. If an integer, only the last look_back_days periods
            will be used.
        :param lookahead_days: Int
            Number of periods in the future to predict using the given model.
        :param stride_days: Int
            Number of periods to progress at each step in the historical forecast.
        :param ts_train: Darts TimeSeries
            Initial data to train model on, which will subsequently be extended
            by more and more validation data, while keeping the look back days
            fixed. Data is assumed to be transformed using the transformer if the
            parameter ts_are_transformed is True. All target and covariates are
            assumed to be in the time series.
            If None, the data in the object will be used.
        :param ts_val: Darts TimeSeries
            Validation data to compare against. Data is assumed to be transformed
            using the transformer if the parameter ts_are_transformed is True.
            All targets and covariates are assumed to be in the time series.
            If None, the data in the object will be used.
        :param transformer: Union[None, Any]
            The transformer applied to the data, if any. If it can be inferred that
            the data is transformed, the transformer stored in the object will be used.
        :param ts_are_transformed: Bool.
            Only applicable if data is explicitly given, otherwise it will be inferred.
            If a transformer is given, it will be applied to the data if
            ts_are_transformed is False. If ts_are_transformed, they are considered
            to be transformed.
        :param inverse_transform: Bool
            If a transformer is given, and inverse_transform is True, then the
            historical forecast plot will be produced using an inverse transform.
        :param num_samples: Int
            Number of samples to draw for the historical forecast distribution
            when the model is stochastic.
        :param save_plot: Whether to save the plot or not.
        :param show_plot: Whether to show the plot or not.
        :return: Dictionary with forecasted time series and transformer
        """
        print(f"*** Performing historical forecast of {model.__str__()} ***")
        # prepare data to use
        if ts_train is None:
            ts_train = self.ts_train
        elif isinstance(ts_train, darts.TimeSeries):
            pass
        else:
            raise ValueError(
                "ts_train must be an instance of darts.TimeSeries if given"
            )
        if ts_val is None:
            ts_val = self.ts_val
        elif isinstance(ts_val, darts.TimeSeries):
            pass
        else:
            raise ValueError(
                "ts_train must be an instance of darts.TimeSeries if given"
            )
        # determine split combining training and validation data
        split_date_hf = max(ts_train.time_index)

        # transform data, if needed
        try:
            ts_are_transformed = getattr(ts_train, "transformed")
            assert ts_are_transformed == getattr(
                ts_val, "transformed"
            ), "Inconsistent transformed data"
            transformer = self.transformer
            if ts_are_transformed:
                assert self.transformer._fit_called, "transformer is not fitted"
                print("Input data is transformed")
                str_tr_file = f"{transformer.__str__()}"
                str_tr_plot = f"transformer: {transformer.__str__()}"
            else:
                print("Input data is not performed")
                str_tr_file = "untransformed"
                str_tr_plot = "untransformed data"
        except AttributeError:
            if transformer is None:
                transformer = self.transformer
            if transformer is None:
                print(
                    "No transformation of input data is performed (no transformer given)"
                )
                str_tr_file = "untransformed"
                str_tr_plot = "untransformed data"
            else:
                if ts_are_transformed:
                    assert transformer._fit_called, "transformer is not fitted"
                    print("Input data is assumed to be transformed (transformer given)")
                else:
                    print("Input data will be transformed (transformer given)")
                    transformer = transformer.fit(ts_train)
                    ts_train = transformer.transform(ts_train)
                    ts_val = transformer.transform(ts_val)
                str_tr_file = f"{transformer.__str__()}"
                str_tr_plot = f"transformer: {transformer.__str__()}"

        # combine data
        ts_data = ts_train.append(ts_val)

        # prepare columns
        if past_covariates_cols is None:
            past_covariates = None
            str_past_covariates = "None"
        elif isinstance(past_covariates_cols, list):
            past_covariates = ts_data[past_covariates_cols]
            str_past_covariates = len(past_covariates_cols)
        else:
            raise ValueError("past_covariates_cols must be a list or None")
        if future_covariates_cols is None:
            future_covariates = None
            str_future_covariates = "None"
        elif isinstance(future_covariates_cols, list):
            future_covariates = ts_data[future_covariates_cols]
            str_future_covariates = len(future_covariates_cols)
        else:
            raise ValueError("past_covariates_cols must be a list or None")
        if isinstance(target_cols, str) or isinstance(target_cols, list):
            ts_data_target = ts_data[target_cols]
        else:
            raise ValueError("target_cols must be a str or list")

        # call historical forecast method and get forecasted time series
        ts_hf_target = model.historical_forecasts(
            series=ts_data_target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            train_length=look_back_days,
            start=split_date_hf,
            forecast_horizon=lookahead_days,
            stride=stride_days,
            retrain=True,
            verbose=True,
        )
        # create a new time series with all the same columns as the validation data
        ts_hf = ts_val.copy().slice_intersect(ts_hf_target)
        df_hf = ts_hf.pd_dataframe()
        df_hf[target_cols] = ts_hf_target.pd_dataframe()
        ts_hf = darts.TimeSeries.from_dataframe(df_hf)

        # get look_back_days
        if look_back_days is None:
            look_back_days = len(ts_train)

        # optional inverse transform for plotting
        if transformer is None:
            str_inv_tr_file = ""
            str_inv_tr_plot = ""
        elif inverse_transform:
            ts_train = transformer.inverse_transform(ts_train)
            ts_val = transformer.inverse_transform(ts_val)
            ts_hf = transformer.inverse_transform(ts_hf)
            str_inv_tr_file = "_with_inverse_transform"
            str_inv_tr_plot = " (with inverse transform)"
        else:
            str_inv_tr_file = "_without_inverse_transform"
            str_inv_tr_plot = " (no inverse transform)"

        # prepare model name string
        str_name = model.__str__().split("\n")[0]
        if str_name[-1] == ",":
            str_name = f"{str_name} ...)"

        # plot
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Historical forecasting: {str_name}", fontsize=16)
        fig.text(
            x=0.13,
            y=0.90,
            s=f"target column(s): {target_cols}, past covariates: {str_past_covariates}, "
            f"future covariates: {str_future_covariates}"
            f"\nsplit date: {str(split_date_hf.date())}, look back days: {look_back_days}, "
            f"lookahead days: {lookahead_days}, forecasting stride: {stride_days}, {str_tr_plot}{str_inv_tr_plot}",
            fontsize=14,
        )
        # plot target
        ts_train[target_cols].plot(
            label="training", color="cornflowerblue", linewidth=1.0
        )
        ts_val[target_cols].plot(label="validation", color="dodgerblue", linewidth=1.0)
        ts_hf[target_cols].plot(
            label="historical forecast", color="coral", linewidth=1.0
        )

        if save_plot:
            plt.savefig(
                os.path.join(
                    self.path,
                    f"historical forecast_{str_name}_la{lookahead_days}_lb{look_back_days}_"
                    f"{str_tr_file}{str_inv_tr_file}.png",
                )
            )
        if show_plot:
            plt.show()
        plt.close()
        print("Success!")
        return {"ts_hf": ts_hf, "transformer": transformer}

    def backtest_evaluate(
        self,
        model: Any,
        target_cols: [str, list],
        past_covariates_cols: Union[None, list],
        future_covariates_cols: Union[None, list],
        look_back_days: Union[None, int],
        lookahead_days: int,
        stride_days: int,
        ts_train: Union[darts.TimeSeries, None] = None,
        ts_val: Union[darts.TimeSeries, None] = None,
        transformer: Union[None, Any] = None,
        ts_are_transformed: bool = False,
        inverse_transform: bool = False,
        num_samples: int = 1,
        save_plot: bool = True,
        show_plot: bool = True,
    ):
        """
        Performs a backtest of the model using historical forecasts for the model
        and evaluating the model outputs against the validation data.

        :param model: Darts model
            The model to use for historical forecasts.
        :param target_cols: List or str
            The target column (uni-variate) or columns (multivariate) to use in
            the model.
        :param past_covariates_cols: None or Darts TimeSeries
            Past covariates to feed to the model.
        :param future_covariates_cols: None or Darts TimeSeries
            Future covariates to feed to the model.
        :param look_back_days: None or int
            If None, the entire length of the time series ts_train will be used
            for training. If an integer, only the last look_back_days periods
            will be used.
        :param lookahead_days: Int
            Number of periods in the future to predict using the given model.
        :param stride_days: Int
            Number of periods to progress at each step in the historical forecast.
        :param ts_train: Darts TimeSeries
            Initial data to train model on, which will subsequently be extended
            by more and more validation data, while keeping the look back days
            fixed. Data is assumed to be transformed using the transformer if the
            parameter ts_are_transformed is True. All target and covariates are
            assumed to be in the time series.
            If None, the data in the object will be used.
        :param ts_val: Darts TimeSeries
            Validation data to compare against. Data is assumed to be transformed
            using the transformer if the parameter ts_are_transformed is True.
            All targets and covariates are assumed to be in the time series.
            If None, the data in the object will be used.
        :param transformer: Union[None, Any]
            The transformer applied to the data, if any. If it can be inferred that
            the data is transformed, the transformer stored in the object will be used.
        :param ts_are_transformed: Bool.
            Only applicable if data is explicitly given, otherwise it will be inferred.
            If a transformer is given, it will be applied to the data if
            ts_are_transformed is False. If ts_are_transformed, they are considered
            to be transformed.
        :param inverse_transform: Bool
            If a transformer is given, and inverse_transform is True, then the
            historical forecast plot will be produced using an inverse transform.
        :param num_samples: Int
            Number of samples to draw for the historical forecast distribution
            when the model is stochastic.
        :param save_plot: Whether to save the plot or not.
        :param show_plot: Whether to show the plot or not.
        :return: Dictionary with metrics
        """
        # call historical forecasting helper
        d_hf = self.historical_forecasts_evaluate(
            model=model,
            target_cols=target_cols,
            past_covariates_cols=past_covariates_cols,
            future_covariates_cols=future_covariates_cols,
            look_back_days=look_back_days,
            lookahead_days=lookahead_days,
            stride_days=stride_days,
            ts_train=ts_train,
            ts_val=ts_val,
            transformer=transformer,
            ts_are_transformed=ts_are_transformed,
            inverse_transform=inverse_transform,
            num_samples=num_samples,
            save_plot=save_plot,
            show_plot=show_plot,
        )
        ts_hf = d_hf["ts_hf"]
        transformer = d_hf["transformer"]

        # prepare validation data to use
        if ts_val is None:
            ts_val = self.ts_val
        elif isinstance(ts_val, darts.TimeSeries):
            pass
        else:
            raise ValueError(
                "ts_train must be an instance of darts.TimeSeries if given"
            )
        # transform validation sample if needed
        if inverse_transform or (transformer is None):
            pass
        else:
            ts_val = transformer.transform(ts_val)

        # down-sample ts_val to ts_hf because of the different strides
        ts_val_target = ts_val.slice_intersect(ts_hf)[target_cols]
        ts_hf_target = ts_hf[target_cols]

        # evaluate metrics
        # consider first metrics that can be problematic
        # try:
        #     m_mase = darts_metrics.mase(ts_val_target, ts_hf_target)
        #     m_msse = darts_metrics.msse(ts_val_target, ts_hf_target)
        #     m_rmsse = darts_metrics.rmsse(ts_val_target, ts_hf_target)
        # except IndexError:
        #     m_mase = np.nan
        #     m_msse = np.nan
        #     m_rmsse = np.nan
        # try:
        #     # not for standardised data
        #     m_ope = darts_metrics.ope(ts_val_target, ts_hf_target)
        # except ValueError:
        #     m_ope = np.nan

        # return
        return {
            "model": model.__str__(),
            "transformer:": transformer.__str__(),
            "look_back_days": look_back_days,
            "lookahead_days": lookahead_days,
            "target_cols": target_cols,
            "past_covariates_cols": past_covariates_cols,
            "future_covariates_cols": future_covariates_cols,
            "Mean Error (agg_abs)": darts_metrics.merr(ts_val_target, ts_hf_target),
            "Mean Absolute Error (agg_abs)": darts_metrics.mae(
                ts_val_target, ts_hf_target
            ),
            "Mean Squared Error (agg_abs)": darts_metrics.mse(
                ts_val_target, ts_hf_target
            ),
            "Root Mean Squared Error (agg_abs)": darts_metrics.rmse(
                ts_val_target, ts_hf_target
            ),
            "Root Mean Squared Log Error (agg_abs)": darts_metrics.rmsle(
                ts_val_target, ts_hf_target
            ),
            # "Mean Absolute Scaled Error (agg_rel)": m_mase,
            # "Mean Squared Scaled Error (agg_rel)": m_msse,
            # "Root Mean Squared Scaled Error (agg_rel)": m_rmsse,
            "Mean Absolute Percentage Error (agg_rel)": darts_metrics.mape(
                ts_val_target, ts_hf_target
            ),
            "Symmetric Mean Absolute Percentage Error (agg_rel)": darts_metrics.smape(
                ts_val_target, ts_hf_target
            ),
            # "Overall Percentage Error (agg_rel)": m_ope,
            "Mean Absolute Ranged Relative Error (agg_rel)": darts_metrics.marre(
                ts_val_target, ts_hf_target
            ),
            "Coefficient of Determination R^2": darts_metrics.r2_score(
                ts_val_target, ts_hf_target
            ),
            "Coefficient of Variation in percent": darts_metrics.coefficient_of_variation(
                ts_val_target, ts_hf_target
            ),
        }

    # %% Backtesting
    def batch_backtesting(
        self,
        models: Any,
        lookahead_days: list,
        look_back_days: list,
        target_cols: [str, list],
        past_covariates_cols: Union[None, list],
        future_covariates_cols: Union[None, list],
        stride_days: Union[int, list],
        ts_train: Union[darts.TimeSeries, None] = None,
        ts_val: Union[darts.TimeSeries, None] = None,
        transformer: Union[None, Any] = None,
        ts_are_transformed: bool = False,
        inverse_transform: bool = False,
        num_samples: int = 1,
        save_plot: bool = True,
        show_plot: bool = True,
    ):
        """
        Run backtests for many different models and aggregate the results
        """
        # prepare iteration
        if isinstance(stride_days, int):
            stride_days = [stride_days]
        elif isinstance(stride_days, list):
            pass
        else:
            raise ValueError("invalid stride days")

        n = 0
        d_bt = {}
        for e in itertools.product(models, look_back_days, lookahead_days, stride_days):
            try:
                d_bt[n] = self.backtest_evaluate(
                    model=e[0],
                    target_cols=target_cols,
                    past_covariates_cols=past_covariates_cols,
                    future_covariates_cols=future_covariates_cols,
                    look_back_days=e[1],
                    lookahead_days=e[2],
                    stride_days=e[3],
                    ts_train=ts_train,
                    ts_val=ts_val,
                    transformer=transformer,
                    ts_are_transformed=ts_are_transformed,
                    inverse_transform=inverse_transform,
                    num_samples=num_samples,
                    save_plot=save_plot,
                    show_plot=show_plot,
                )
                n += 1
            except:
                print("Backtest failed, skipping!")
                d_bt[n] = None
                n += 1

        # save
        print("Saving summary of batch backtesting... ", end="")
        df_bt = pd.DataFrame(d_bt)
        df_bt.T.to_excel(
            os.path.join(self.path, "batch_backtesting.xlsx"), engine="openpyxl"
        )
        print("Success!")

