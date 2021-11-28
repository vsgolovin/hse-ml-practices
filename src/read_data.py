"""
Functions for reading and concatenating / splitting data.
"""


import pandas as pd

DATA_PATH = 'data'


def read_dataset(folder: str, filename: str) -> pd.DataFrame:
    """
    Read dataframe from `DATA_PATH`/`folder`/`filename` csv file.
    """
    path = '/'.join((DATA_PATH, folder, filename))
    return pd.read_csv(path)


def read_train(folder: str = 'raw', filename: str = 'train.csv') -> pd.DataFrame:
    """
    Read train dataset `filename` from directory `folder`
    (`folder` is one of `'raw'`, `'interim'`, `'processed'`).
    """
    return read_dataset(folder, filename)


def read_test(folder: str = 'raw', filename: str = 'test.csv') -> pd.DataFrame:
    """
    Read test dataset `filename` from directory `folder`
    (`folder` is one of `'raw'`, `'interim'`, `'processed'`).
    """
    return read_dataset(folder, filename)


def concat_df(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate dataframes with train and test sets.
    """
    return pd.concat([df_train, df_test], sort=True).reset_index(drop=True)


def split_df(df_all: pd.DataFrame, row_num=890):
    """
    Split dataframe `df` in 2 dataframes -- train and test sets.
    Include all rows up to `row_num` in the train set.
    Both returned dataframes are copies of the original.
    """
    df_train = df_all.loc[:row_num].copy()
    df_test = df_all.loc[row_num + 1:].drop(['Survived'], axis=1)
    return df_train, df_test