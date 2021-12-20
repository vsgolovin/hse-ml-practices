"""
Functions for reading and concatenating / splitting data.
"""

import pandas as pd


def read_dataset(folder: str, filename: str) -> pd.DataFrame:
    """
    Read dataframe from `folder`/`filename` csv file.
    """
    path = '/'.join((folder, filename))
    return pd.read_csv(path)


def read_train(folder: str = 'data/raw',
               filename: str = 'train.csv') -> pd.DataFrame:
    """
    Read train dataset `filename` from directory `folder`
    """
    return read_dataset(folder, filename)


def read_test(folder: str = 'data/raw',
              filename: str = 'test.csv') -> pd.DataFrame:
    """
    Read test dataset `filename` from directory `folder`
    """
    return read_dataset(folder, filename)


def concat_df(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate dataframes with train and test sets.
    """
    return pd.concat([df_train, df_test], sort=True).reset_index(drop=True)


def split_df(df_all: pd.DataFrame, row_num=890, drop_test=('Survived')):
    """
    Split dataframe `df` in 2 dataframes -- train and test sets.
    Include all rows up to `row_num` in the train set.
    Both returned dataframes are copies of the original.
    """
    df_train = df_all.loc[:row_num].copy()
    if drop_test:
        df_test = df_all.loc[row_num + 1:].drop(drop_test, axis=1)
    else:
        df_test = df_all.loc[row_num + 1:].copy()
    return df_train, df_test.reset_index(drop=True)
