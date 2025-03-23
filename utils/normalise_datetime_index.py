import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Union


def is_business_day(date: datetime):
    """
    Determines if a day is a business day or not
    """
    return bool(len(pd.bdate_range(date, date)))


def dt_normalise(
    df: pd.DataFrame,
    date_col: Union[str, None] = None,
    fill_method: Union[str, None] = "bfill",
    fill_limit: Union[int, None] = None,
    method_duplicates: str = "first_non_nan",
    sort_ascending: bool = False,
    sorting_algorithm: str = "quicksort",
    silent: bool = False,
):
    """
    This function normalises the datetime index of a pd.DataFrame: Any
    non-business date is moved to the nearest (future) business day.

    If the resulting dates are overlapping, the correct method to deal
    with the duplicate dates can be chosen (default to combining data
    and take the first non-nan in each column).

    Parameters
    ----------
    df: pd.DataFrame
        Input data frame to be normalised.
    date_col: str or None
        If not None, this column will be used as index.
    fill_method: str or None
        bfill: fill a backward index direction (e.g. old to new
        when sorting descending order)
        ffill: fill forward an index direction (e.g. old to new
        when sorting ascending order)
        None: Do no filling
    fill_limit: int
        How many rows to fill. (default is None, i.e. no limit)
    method_duplicates: "keep_only_last_row", "keep_only_first_row",
        "first_non_nan", "last_non_nan", "average_value", "raise"
        The default is to combine the observations and take the first
        non-nan observation and use that.
    sort_ascending: bool
        Descending: new --> old (default, convention in stock objects)
        Ascending: old --> new
    sorting_algorithm: str
        quicksort, heapsort, mergesort, timsort
    silent: bool
        Show informative messages during the process?

    Returns
    -------
    pd.DataFrame
    """
    # check that date_col is present if it is given, set index
    if date_col is not None:
        assert date_col in df.columns, "date_col is not a valid column in df"
        # reset index
        df = df.reset_index()
        df = df.set_index(date_col)

    # Ensure sorting direction is from descending (new --> old) or ascending (old --> new)
    df = df.sort_index(kind=sorting_algorithm, ascending=sort_ascending)

    # get transformed dates (offset by 1 or 2 days; NaTs are kept)
    idx_business_days = pd.Index(
        [
            (
                x
                if pd.isnull(x) or is_business_day(x)
                else (
                    x + timedelta(days=1)
                    if is_business_day(x + timedelta(days=1))
                    else x + timedelta(days=2)
                )
            )
            for x in df.index
        ]
    )
    # replace index
    df.index = idx_business_days

    # check consistency, are there duplicates? NaTs are also thrown away
    if sum(df.index.duplicated(keep=False)) > 0:  # keep all duplicates in counting
        if not silent:
            print("There are duplicate indices:", end=" ")
        # method for how to deal with duplicates:
        if method_duplicates == "raise":
            raise IndexError("Indices are duplicated")
        elif method_duplicates == "keep_only_last_row":
            idx_duplicates = df.index.duplicated(keep="last")
            df = df[~idx_duplicates]
            if not silent:
                print("only first row kept")
        elif method_duplicates == "keep_only_first_row":
            idx_duplicates = df.index.duplicated(keep="first")
            df = df[~idx_duplicates]
            if not silent:
                print("only last row kept")
        elif method_duplicates == "first_non_nan":
            idx_duplicates = df.index.duplicated(keep=False)
            # split duplicates and non-duplicates dataframes
            df_dpl = df[idx_duplicates]
            df = df[~idx_duplicates]
            # group by the duplicates and take the first non-nan
            df_dpl = df_dpl.groupby(level=0).first(numeric_only=False)
            # merge observations
            df = pd.concat(objs=[df, df_dpl], verify_integrity=True)
            if not silent:
                print("first non-NaN observation in each column is kept")
        elif method_duplicates == "last_non_nan":
            idx_duplicates = df.index.duplicated(keep=False)
            # split duplicates and non-duplicates dataframes
            df_dpl = df[idx_duplicates]
            df = df[~idx_duplicates]
            # group by the duplicates and take the last non-nan
            df_dpl = df_dpl.groupby(level=0).last(numeric_only=False)
            # merge observations
            df = pd.concat(objs=[df, df_dpl], verify_integrity=True)
            if not silent:
                print("last non-NaN observation in each column is kept")
        elif method_duplicates == "average_value":  # only for numeric columns
            idx_duplicates = df.index.duplicated(keep=False)
            # split duplicates and non-duplicates dataframes
            df_dpl = df[idx_duplicates]
            df = df[~idx_duplicates]
            # group by the duplicates and take the average value of all columns
            df_dpl = df_dpl.groupby(level=0).mean(
                numeric_only=True
            )  # tries to use everything
            # check that columns were not dropped
            if df_dpl.shape[1] < df.shape[1]:
                raise TypeError(
                    "There were non-numeric columns present, for which the mean could not be calculated."
                )
            # merge observations
            df = pd.concat(objs=[df, df_dpl], verify_integrity=True)
            if not silent:
                print(
                    "observations combined and average value taken for numerical columns"
                )
        else:
            raise NameError("Duplicate method is not valid.")

    # Sort again: ensure sorting direction is descending (new --> old) or ascending (old --> new)
    df = df.sort_index(kind=sorting_algorithm, ascending=sort_ascending)

    # optional filling missing values: Only when there is 1 or more business day difference
    if fill_method is None:
        pass
    elif fill_method == "ffill":
        df = df.ffill(limit=fill_limit).infer_objects(copy=False)
    elif fill_method == "bfill":
        df = df.bfill(limit=fill_limit).infer_objects(copy=False)
    else:
        raise ValueError("Invalid input for fill_method")

    return df
