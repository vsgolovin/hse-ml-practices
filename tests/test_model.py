import random
import matplotlib
import pandas as pd
from titanic.model import leaderboard_model, plot_roc_curve, run_classifier
import pytest


@pytest.fixture
def toy_dataset():
    n = 1000
    train = 0.8
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


@pytest.fixture
def classification_results(toy_dataset):
    x_train, y_train, x_test, y_test, df = toy_dataset
    model = leaderboard_model()
    return run_classifier(model, x_train, y_train, x_test, df)


def test_classifier(toy_dataset, classification_results):
    y_test = toy_dataset[3]
    probs = classification_results[2]
    cols = [col for col in probs if col.endswith('Prob_1')]
    y_pred = probs.loc[:, cols].mean(axis=1).to_numpy(dtype=int)
    assert (y_pred == y_test).all()


def test_roc_curve(classification_results):
    fprs, tprs, _, _ = classification_results
    fig = plot_roc_curve(fprs, tprs)
    assert isinstance(fig, matplotlib.figure.Figure) \
        and len(fig.get_axes()) == 1
