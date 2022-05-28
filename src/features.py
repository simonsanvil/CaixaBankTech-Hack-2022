from typing import Iterable
import pandas as pd
import numpy as np


def make_features(df, cols_to_make_features_of,**kwargs):
    """
    Build the features of a given timeseries to be used in a model
    """
    features_args = dict(num_lags=3, roll_nums=(3,5))
    features_args.update(kwargs)
    feature_dfs = [make_ts_features(df[col], add_time_based=(i==0),**features_args) for i,col in enumerate(cols_to_make_features_of)]
    features_df = pd.concat(feature_dfs,axis=1).join(df[df.columns.drop(cols_to_make_features_of)[1:]])
    return features_df

def make_ts_features(
    x:pd.Series,
    num_lags:int = 5,
    roll_nums:tuple = (3,5,7),
    add_time_based:bool = False,
    roll_agg: str = 'mean',
) -> pd.DataFrame:
    """
    Build the features of a given timeseries
    based on its historical values and time-related
    attributes
    """
    roll_agg = [roll_agg] if not isinstance(roll_agg, (list,tuple)) else roll_agg
    x = x.copy()
    df = pd.DataFrame(x)
    df = df.assign(
        **{ # add lag features
            f"{x.name}_lag_{i}": x.shift(i)
            for i in range(1,num_lags+1)
        },
        **{ # add rolling average features
            f"{x.name}_rolling_{agg}_{i}": x.rolling(i).agg(agg)
            for i in roll_nums
            for agg in roll_agg
        }, # add rolling sum features
        **{
            # features based on the trend
            f"{x.name}_mean_week_diff" : x.rolling(7).mean().diff(),
            f"{x.name}_mean_3_diff" : x.rolling(3).mean().diff(),
        }
    )
    if add_time_based:
        # add features based on time and date
        df = df.assign(
            month = lambda X: X.index.month,
            julian_day = lambda X: X.index.dayofyear,
            day_of_month = lambda X: X.index.day,
            day_of_week = lambda X: X.index.dayofweek,
            is_weekend = lambda X: X.index.weekday.isin([5,6]),
            is_spring = lambda X: X.index.month.isin([3,4,5]),
            is_summer = lambda X: X.index.month.isin([6,7,8]),
            is_winter = lambda X: X.index.month.isin([12,1,2])
        )
    return df.dropna()