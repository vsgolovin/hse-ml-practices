import pandas as pd
import pytest
import titanic.read_data as rd


@pytest.mark.parametrize('folder', ['raw', 'interim', 'processed'])
def test_read_train(folder):
    folder = '/'.join(('data', folder))
    df = rd.read_train(folder)
    assert isinstance(df, pd.core.frame.DataFrame) and df.shape[0] == 891


@pytest.mark.parametrize('folder', ['raw', 'interim', 'processed'])
def test_read_test(folder):
    folder = '/'.join(('data', folder))
    df = rd.read_test(folder)
    assert isinstance(df, pd.core.frame.DataFrame) and df.shape[0] == 418


@pytest.fixture
def raw_train_dataset():
    return rd.read_train('data/raw')


@pytest.fixture
def interim_train_dataset():
    return rd.read_train('data/interim')


@pytest.mark.parametrize('row_num', [*range(891)])
@pytest.mark.parametrize(
    'col_label',
    ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
def test_interim_item(row_num, col_label, raw_train_dataset,
                      interim_train_dataset):
    x = raw_train_dataset.loc[row_num, col_label]
    y = interim_train_dataset.loc[row_num, col_label]
    assert not pd.isna(y) and (pd.isna(x) or abs(x) < 1e-6 or x == y)


@pytest.fixture
def df_top():
    return pd.DataFrame(data=[[1, 2], [5, 4]])


@pytest.fixture
def df_bot():
    return pd.DataFrame(data=[[3, 7]])


@pytest.fixture()
def df_full():
    return pd.DataFrame(data=[[1, 2], [5, 4], [3, 7]])


def test_concat(df_top, df_bot, df_full):
    assert (rd.concat_df(df_top, df_bot) == df_full).all().all()


def test_split(df_top, df_bot, df_full):
    df1, df2 = rd.split_df(df_full, 1, drop_test=())
    print(df1)
    print(df2)
    assert (df1 == df_top).all().all() and (df2 == df_bot).all().all()
