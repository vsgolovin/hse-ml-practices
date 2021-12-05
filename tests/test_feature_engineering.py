import pytest
import random
import pandas as pd
import src.feature_engineering as fteng


def test_extract_surname():
    surnames = pd.Series(data=(
        "Caldwell, Master. Alden Gates",
        "Doling, Mrs. John T (Ada Julia Bone)",
        "Navratil, Mr. Michel",
        "Hakkarainen, Mrs. Pekka Pietari (Elin Matilda Dolck)"))
    ans = ["Caldwell", "Doling", "Navratil", "Hakkarainen"]
    assert fteng.extract_surname(surnames) == ans


def nonunique_args(n, xmax):
    x, y, z = [], [], []
    samples = random.sample(range(xmax), n)  # all values are unique
    for i in range(n):
        if random.random() < 0.8:
            x.append(samples[i])
            y.append(samples[i])
            z.append(samples[i])
        else:
            if random.choice((True, False)):
                x.append(samples[i])
            else:
                y.append(samples[i])
    return x, y, z


@pytest.mark.parametrize('n', [*range(10, 100)])
@pytest.mark.parametrize('xmax', [100, 147, 999])
def test_nonuniques(n, xmax):
    seq_1, seq_2, ans = nonunique_args(n, xmax)
    assert set(fteng.nonuniques(seq_1, seq_2)) == set(ans)


def test_survival_rates():
    col_1 = 'Family'
    col_2 = 'FSize'
    df = pd.DataFrame(columns=[col_1, col_2, 'Survived'])
    df[col_1] = ['Smith', 'Smith', 'Wilson', 'Wilson', 'DiCaprio']
    df[col_2] = [3, 3, 2, 2, 1]
    df['Survived'] = [1, 0, 0, 0, 0]
    df2 = pd.DataFrame(columns=[col_1, col_2, 'Survived'])
    df2[col_1] = ['Smith']
    df2[col_2] = [3]
    df2['Survived'] = 0
    tpl = fteng.survival_rates(df, df2, col_1, col_2)
    assert tpl[0] == [0.5, 0.5, 0.2, 0.2, 0.2] \
        and tpl[1] == [1, 1, 0, 0, 0] \
        and tpl[2] == [0.5] and tpl[3] == [1]


def test_nonnumerical_features():
    data = [['a', 'b', 'c'], ['c', 'a', 'b'], ['b', 'c', 'a']]
    cols = ['A', 'B', 'C']
    df = pd.DataFrame(data=data, columns=cols)
    data_ans = [['a', 1, 'c'], ['c', 0, 'b'], ['b', 2, 'a']]
    df_ans = pd.DataFrame(data=data_ans, columns=cols)
    fteng.convert_nonnumerical_features(df, ('B'))
    assert (df == df_ans).all().all()


def test_categorical_features():
    data = [[1, 'x', 2], [2, 'y', 1], [0, 'x', 3]]
    cols = ['A', 'B', 'C']
    df = pd.DataFrame(data=data, columns=cols)
    df_encoded = df.copy()
    df_encoded['B_1'] = [1.0, 0.0, 1.0]
    df_encoded['B_2'] = [0.0, 1.0, 0.0]
    df = fteng.encode_categorical_features(df, ('B'))
    assert (df == df_encoded).all().all()
