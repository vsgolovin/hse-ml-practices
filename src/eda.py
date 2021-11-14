import pandas as pd


def concat_df(train_data, test_data):
    "Returns a concatenated df of training and test set"
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


df_train = pd.read_csv('data/raw/train.csv')
df_test = pd.read_csv('data/raw/test.csv')
df_all = concat_df(df_train, df_test)

# Filling the missing values in Age with the medians of Sex and Pclass groups
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(
    lambda x: x.fillna(x.median()))

# Filling the missing values in Embarked with S
df_all['Embarked'] = df_all['Embarked'].fillna('S')

# Filling the missing value in Fare with the median Fare
# of 3rd class alone passenger
med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df_all['Fare'] = df_all['Fare'].fillna(med_fare)

# Creating Deck column from the first letter of the Cabin column
# (M stands for Missing)
df_all['Deck'] = df_all['Cabin'].apply(
    lambda s: s[0] if pd.notnull(s) else 'M')

# Passenger in the T deck is changed to A
idx = df_all[df_all['Deck'] == 'T'].index
df_all.loc[idx, 'Deck'] = 'A'

df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

# Dropping the Cabin feature
df_all.drop(['Cabin'], inplace=True, axis=1)

# Save processed train and test datasets
df_train, df_test = divide_df(df_all)
df_train.to_csv('data/interim/train.csv', index=False)
df_test.to_csv('data/interim/test.csv', index=False)
