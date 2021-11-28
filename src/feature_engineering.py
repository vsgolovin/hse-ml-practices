"""
Create new features and drop useless ones.
Data is read from `data/interim` and should not have NaNs.
New data is saved to `data/processed` and it intended to be used for training
and testing the model.
"""

import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import read_data as rd


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


def survival_rates(df_trn, df_tst, index_column,
                   value_column, value_min=1):
    """
    Calculate survival rates for unique values of `index_column`.
    Only uses entries with `value_column' feature `> value_min`.
    """
    # values occur in both train and test sets
    non_unique = nonuniques(df_trn[index_column], df_tst[index_column])
    # dataframe with `group_cols` columns and `column` index
    group_cols = [index_column, 'Survived', value_column]
    value_index = 1
    df_sr = df_trn.groupby(index_column)[group_cols].median()
    rates = {}
    for i in range(len(df_sr)):
        if (df_sr.index[i] in non_unique \
           and df_sr.iloc[i, value_index] > value_min):
            rates[df_sr.index[i]] = df_sr.iloc[i, 0]
    return rates


# Read data
df_train = rd.read_train('interim')
df_test = rd.read_test('interim')
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
dfs = [df_train, df_test]

# Survival rates
family_rates = survival_rates(df_train, df_test, 'Family', 'Family_Size')
ticket_rates = survival_rates(df_train, df_test, 'Ticket', 'Ticket_Frequency')
mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)

for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(
            family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)

df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)

for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(
            ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)

df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

for df in [df_train, df_test]:
    df['Survival_Rate'] = \
        (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = \
        (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2

# Feature transformation
non_numeric_features = ['Embarked', 'Sex', 'Deck',
                        'Title', 'Family_Size_Grouped', 'Age', 'Fare']

for df in dfs:
    for feature in non_numeric_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])

cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title',
                'Family_Size_Grouped']
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(
            df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

# Drop useless features
drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size',
             'Family_Size_Grouped', 'Name', 'Parch', 'PassengerId', 'Pclass',
             'Sex', 'SibSp', 'Ticket', 'Title', 'Ticket_Survival_Rate',
             'Family_Survival_Rate', 'Ticket_Survival_Rate_NA',
             'Family_Survival_Rate_NA']
# unlike notebook, do not drop 'Survive'
df_train.drop(columns=drop_cols, inplace=True)
df_test.drop(columns=drop_cols, inplace=True)

# port processed data
df_train.to_csv('data/processed/train.csv', index=False)
df_test.to_csv('data/processed/test.csv', index=False)
