import random
import pandas as pd
from src.model import leaderboard_model, run_classifier
import pytest


def toy_dataset(n=1000, train=0.8):
    data = []
    for _ in range(n):
        x = random.random()
        if x < 0.5:
            data.append((1, 0, 0))
        elif x < 0.8:
            data.append((0, 1, 0))
        else:
            data.append((1, 1, 1))
    df = pd.DataFrame(columns=list('abc'), data=data)
    ind = int(round(train * n))
    x_train = df.iloc[:ind, :-1].to_numpy()
    y_train = df.iloc[:ind, -1].to_numpy()
    x_test = df.iloc[ind:, :-1].to_numpy()
    y_test = df.iloc[ind:, -1].to_numpy()
    return x_train, y_train, x_test, y_test, df.drop(columns=['c'])


@pytest.mark.parametrize('n', [1000])
@pytest.mark.parametrize('train', [0.8])
def test_classifier(n, train):
    x_train, y_train, x_test, y_test, df = toy_dataset(n, train)
    print(y_train)
    model = leaderboard_model()
    probs = run_classifier(model, x_train, y_train, x_test, df)[2]
    cols = [col for col in probs if col.endswith('Prob_1')]
    y_pred = probs.loc[:, cols].mean(axis=1).to_numpy(dtype=int)
    assert (y_pred == y_test).all()
