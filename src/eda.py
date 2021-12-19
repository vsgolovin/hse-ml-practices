"""
Modify raw data by using the results of exploratory data analysis.
See the corresponding section of the jupyter-notebook for more details.
"""

import pandas as pd
import titanic.read_data as rd
import click


@click.command()
@click.option('--input_dir', default='raw', help='directory with input data')
@click.option('--output_dir', default='interim',
              help='directory for output data')
def main(input_dir, output_dir):
    """
    Modify the raw `titanic` dataset -- fill in blanks and replace the `Cabin`
    column with `Deck`.
    """
    df_train = rd.read_train(input_dir)
    df_test = rd.read_test(input_dir)
    df_all = rd.concat_df(df_train, df_test)

    # Filling the missing values in Age
    # with the medians of Sex and Pclass groups
    df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(
        lambda x: x.fillna(x.median()))

    # Filling the missing values in Embarked with S
    df_all['Embarked'] = df_all['Embarked'].fillna('S')

    # Filling the missing value in Fare with the median Fare
    # of 3rd class alone passenger
    med_fare = df_all.groupby(['Pclass',
                               'Parch', 'SibSp']).Fare.median()[3][0][0]
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
    df_train, df_test = rd.split_df(df_all)
    df_train.to_csv('/'.join((rd.DATA_PATH, output_dir, 'train.csv')),
                    index=False)
    df_test.to_csv('/'.join((rd.DATA_PATH, output_dir, 'test.csv')),
                   index=False)


if __name__ == '__main__':
    main()
