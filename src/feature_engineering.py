"""
Create new features and drop useless ones.
Input data is read from `data/interim` and should not have NaNs.
"""

import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import titanic.read_data as rd
import click


@click.command()
@click.option('--input_dir', default='data/interim', type=click.Path(),
              help='directory with input data')
@click.option('--output_dir', default='data/processed', type=click.Path(),
              help='directory for output data')
def main(input_dir, output_dir):
    """
    Change features in the titanic dataset. Read `train.csv` and `test.csv` and
    modify the exported versions to a different directory.
    """
    # Read data
    df_train = rd.read_train(input_dir)
    df_test = rd.read_test(input_dir)
    df_all = rd.concat_df(df_train, df_test)

    # Binning continuous features
    df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
    df_all['Age'] = pd.qcut(df_all['Age'], 10)

    # Frequency encoding
    df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium',
                  6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)
    df_all['Ticket_Frequency'] = \
        df_all.groupby('Ticket')['Ticket'].transform('count')

    # New features
    df_all['Title'] = df_all['Name'].str.split(
        ', ', expand=True)[1].str.split('.', expand=True)[0]
    df_all['Is_Married'] = 0
    df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1
    df_all['Title'].replace(
        ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
        'Miss/Mrs/Ms', inplace=True)
    df_all['Title'].replace(
        ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
        'Dr/Military/Noble/Clergy', inplace=True)

    # Target encoding
    df_all['Family'] = extract_surname(df_all['Name'])
    df_train, df_test = rd.split_df(df_all, drop_test=None)

    # Survival rates
    train_fsr, train_fsr_NA, test_fsr, test_fsr_NA = \
        survival_rates(df_train, df_test, 'Family', 'Family_Size')
    train_tsr, train_tsr_NA, test_tsr, test_tsr_NA = \
        survival_rates(df_train, df_test, 'Ticket', 'Ticket_Frequency')

    # Add corresponding features to datasets
    df_train['Family_Survival_Rate'] = train_fsr
    df_train['Family_Survival_Rate_NA'] = train_fsr_NA
    df_test['Family_Survival_Rate'] = test_fsr
    df_test['Family_Survival_Rate_NA'] = test_fsr_NA
    df_train['Ticket_Survival_Rate'] = train_tsr
    df_train['Ticket_Survival_Rate_NA'] = train_tsr_NA
    df_test['Ticket_Survival_Rate'] = test_tsr
    df_test['Ticket_Survival_Rate_NA'] = test_tsr_NA
    for df in [df_train, df_test]:
        df['Survival_Rate'] = \
            (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
        df['Survival_Rate_NA'] = \
            (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2

    # Feature transformation
    non_numeric_features = ['Embarked', 'Sex', 'Deck',
                            'Title', 'Family_Size_Grouped', 'Age', 'Fare']
    for df in [df_train, df_test]:
        convert_nonnumerical_features(df, non_numeric_features)

    cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title',
                    'Family_Size_Grouped']
    df_train = encode_categorical_features(df_train, cat_features)
    df_test = encode_categorical_features(df_test, cat_features)

    # Drop useless features
    drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size',
                 'Family_Size_Grouped', 'Name', 'Parch', 'PassengerId',
                 'Pclass', 'Sex', 'SibSp', 'Ticket',
                 'Title', 'Ticket_Survival_Rate',
                 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA',
                 'Family_Survival_Rate_NA']
    # unlike notebook, do not drop 'Survived'
    df_train.drop(columns=drop_cols, inplace=True)
    df_test.drop(columns=drop_cols, inplace=True)

    # port processed data
    df_train.to_csv('/'.join((output_dir, 'train.csv')), index=False)
    df_test.to_csv('/'.join((output_dir, 'test.csv')), index=False)


def extract_surname(data: pd.Series) -> list:
    """
    Extract surnames from `data`. Removes maiden names and punctuation.
    """
    families = []
    for j in range(len(data)):
        name = data.iloc[j]

        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name

        family = name_no_bracket.split(',')[0]
        for char in string.punctuation:
            family = family.replace(char, '').strip()
        families.append(family)

    return families


def nonuniques(x_1, x_2):
    """
    Returns a list of values that occur both in `x_1` and `x_2`.
    """
    s_1 = set(x_1)
    s_2 = set(x_2)
    return list(s_1.intersection(s_2))


def survival_rates(df_trn: pd.DataFrame,
                   df_tst: pd.DataFrame,
                   index_column: str,
                   value_column: str,
                   value_min: int = 1) -> tuple:
    """
    Calculate survival rates for unique values of `index_column`.
    Only uses entries with `value_column' feature `> value_min`.
    """
    # values occur in both train and test sets
    non_unique = nonuniques(df_trn[index_column], df_tst[index_column])
    # dataframe with `group_cols` columns and `column` index
    df_sr = df_trn.groupby(index_column)[[index_column, 'Survived',
                                          value_column]].median()
    rates = {}
    for i in range(len(df_sr)):
        if (df_sr.index[i] in non_unique
           and df_sr.iloc[i, 1] > value_min):
            rates[df_sr.index[i]] = df_sr.iloc[i, 0]

    # dict -> 2 lists for modifying df_tst and df_trn
    mean_surv_rate = np.mean(df_trn['Survived'])

    # train dataset
    rates_train = []
    rates_train_na = []
    for key in df_trn[index_column]:
        if key in rates:
            rates_train.append(rates[key])
            rates_train_na.append(1)
        else:
            rates_train.append(mean_surv_rate)
            rates_train_na.append(0)

    # test dataset
    rates_test = []
    rates_test_na = []
    for key in df_tst[index_column]:
        if key in rates:
            rates_test.append(rates[key])
            rates_test_na.append(1)
        else:
            rates_test.append(mean_surv_rate)
            rates_test_na.append(0)

    return rates_train, rates_train_na, rates_test, rates_test_na


def convert_nonnumerical_features(dframe, features):
    """
    Modify `dframe` to convert non-numerical `features` to numerical type.
    """
    for feature in features:
        dframe[feature] = LabelEncoder().fit_transform(dframe[feature])


def encode_categorical_features(dframe, features):
    """
    Encode categorical `features` in `dframe` with One-Hot encoder.
    """
    encoded_features = []
    for feature in features:
        encoded_feat = OneHotEncoder().fit_transform(
            dframe[feature].values.reshape(-1, 1)).toarray()
        num = dframe[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, num + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = dframe.index
        encoded_features.append(encoded_df)
    dframe = pd.concat([dframe, *encoded_features], axis=1)
    return dframe


if __name__ == '__main__':
    main()
