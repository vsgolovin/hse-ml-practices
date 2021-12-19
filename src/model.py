"""
Train and test random forest classifier using processed data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import titanic.read_data as rd
import click

N_FOLDS = 5


@click.command()
@click.option('--input_dir', default='processed',
              help='directory with input data')
@click.option('--output_dir', default='reports/data',
              help='directory for storing Kaggle submission csv file')
@click.option('--plot_dir', default='reports/figures',
              help='directory for plots')
@click.option('--seed', default=42, help='random seed')
def main(input_dir, output_dir, plot_dir, seed):
    # Read data
    df_train = rd.read_train(input_dir)
    df_test = rd.read_test(input_dir)
    df_all = rd.concat_df(df_train, df_test)
    df_all = df_all.drop(columns=['Survived'])

    # Create train and test datasets
    x_train = StandardScaler().fit_transform(
        df_train.drop(columns=['Survived']))
    y_train = df_train['Survived'].values
    x_test = StandardScaler().fit_transform(
        df_test.drop(columns=['Survived']))
    print('x_train shape: {}'.format(x_train.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('x_test shape: {}'.format(x_test.shape))

    # Random forest classifier
    model = leaderboard_model(seed)

    # train model
    fprs, tprs, probs, importances = run_classifier(
        model, x_train, y_train, x_test, df_all)

    # plotting
    fig = plot_roc_curve(fprs, tprs)
    fig.savefig('/'.join((plot_dir, 'roc.png')))
    fig = plot_importances(importances)
    fig.savefig('/'.join((plot_dir, 'feature_importance.png')))

    # csv file for Kaggle submission
    submission_df = kaggle_submission(probs)
    submission_df.to_csv('/'.join((output_dir, 'submissions.csv')),
                         header=True, index=False)


def leaderboard_model(seed):
    """
    Returns classifier with hyperparameters optimized for Kaggle leaderboard.
    Function created purely for testing convenience.
    """
    return RandomForestClassifier(criterion='gini',
                                  n_estimators=1750,
                                  max_depth=7,
                                  min_samples_split=6,
                                  min_samples_leaf=6,
                                  max_features='auto',
                                  oob_score=True,
                                  random_state=seed,
                                  n_jobs=-1,
                                  verbose=1)


def run_classifier(model, x_train, y_train, x_test, df_all):
    """
    Train and test the classifier `model`.
    """
    probs = pd.DataFrame(np.zeros((len(x_test), N_FOLDS * 2)),
                         columns=['Fold_{}_Prob_{}'.format(i, j)
                                  for i in range(1, N_FOLDS + 1)
                                  for j in range(2)])
    importances = pd.DataFrame(np.zeros((x_train.shape[1], N_FOLDS)),
                               columns=['Fold_{}'.format(i)
                                        for i in range(1, N_FOLDS + 1)],
                               index=df_all.columns)

    oob = 0
    fprs, tprs, scores = [], [], []
    skf = StratifiedKFold(n_splits=N_FOLDS, random_state=N_FOLDS, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train), 1):
        print('Fold {}\n'.format(fold))

        # Fitting the model
        model.fit(x_train[trn_idx], y_train[trn_idx])

        # Computing Train AUC score
        trn_fpr, trn_tpr, _ = roc_curve(
            y_train[trn_idx], model.predict_proba(x_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing Validation AUC score
        val_fpr, val_tpr, _ = roc_curve(
            y_train[val_idx], model.predict_proba(x_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)

        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)

        # x_test probabilities
        probs.loc[:, 'Fold_{}_Prob_0'.format(fold)] = \
            model.predict_proba(x_test)[:, 0]
        probs.loc[:, 'Fold_{}_Prob_1'.format(fold)] = \
            model.predict_proba(x_test)[:, 1]
        importances.iloc[:, fold - 1] = model.feature_importances_

        oob += model.oob_score_ / N_FOLDS
        print('Fold {} OOB Score: {}\n'.format(fold, model.oob_score_))

    print('Average OOB Score: {}'.format(oob))

    # Feature Importance
    importances['Mean_Importance'] = importances.mean(axis=1)
    importances.sort_values(by='Mean_Importance',
                            inplace=True, ascending=False)

    return fprs, tprs, probs, importances


def plot_roc_curve(fprs: list, tprs: list) -> plt.figure:
    """
    Plot reciever operating characteristic (ROC) curve.
    """
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(figsize=(15, 15))

    # Plotting ROC for each fold and computing AUC scores
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs), 1):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        axes.plot(fpr, tpr, lw=1, alpha=0.3,
                  label=f'ROC Fold {i} (AUC = {roc_auc:.3f})')

    # Plotting ROC for random guessing
    axes.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8,
              label='Random Guessing')

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plotting the mean ROC
    axes.plot(mean_fpr, mean_tpr, color='b',
              label=rf'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})',
              lw=2, alpha=0.8)

    # Plotting the standard deviation around the mean ROC Curve
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    axes.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                      alpha=.2, label=r'$\pm$ 1 std. dev.')

    axes.set_xlabel('False Positive Rate', size=15, labelpad=20)
    axes.set_ylabel('True Positive Rate', size=15, labelpad=20)
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    axes.set_xlim(-0.05, 1.05)
    axes.set_ylim(-0.05, 1.05)
    axes.set_title('ROC Curves of Folds', size=20, y=1.02)
    axes.legend(loc='lower right', prop={'size': 13})

    return fig


def plot_importances(importances: pd.DataFrame) -> plt.figure:
    """
    Plot a bar chart of feature importances.
    """
    fig = plt.figure(figsize=(15, 20))
    axes = fig.gca()
    sns.barplot(x='Mean_Importance', y=importances.index,
                data=importances, ax=axes)
    axes.set_xlabel('')
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=15)
    axes.set_title('Random Forest Classifier Mean Feature Importance '
                   + 'Between Folds', size=15)
    return fig


def kaggle_submission(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataframe for Kaggle submission.
    """
    class_survived = [col for col in probs.columns if col.endswith('Prob_1')]
    probs['1'] = probs[class_survived].sum(axis=1) / N_FOLDS
    probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / N_FOLDS
    probs['pred'] = 0
    pos = probs[probs['1'] >= 0.5].index
    probs.loc[pos, 'pred'] = 1

    y_pred = probs['pred'].astype(int)

    submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission_df['PassengerId'] = rd.read_test()['PassengerId']
    submission_df['Survived'] = y_pred.values
    return submission_df


if __name__ == '__main__':
    main()
