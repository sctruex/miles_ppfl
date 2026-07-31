import numpy as np
import pandas as pd


def read_adult_data(file_path=''):
    df = pd.DataFrame()
    df = pd.read_csv(file_path+'adult.data')
    df.columns = [colname.strip() for colname in df.columns]

    df = df.replace(" ?", np.nan)
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    df.replace(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
               ['divorced', 'married', 'married', 'married', 'not married', 'not married', 'not married'], inplace=True)

    category_col = ['race', 'marital-status', 'sex', 'income']
    for col in category_col:
        df[col] = pd.factorize(df[col])[0]

    category_col_1 = ['workclass', 'education', 'occupation',
                      'relationship', 'native-country']
    df_2 = pd.get_dummies(df, columns=category_col_1,
                          dtype=int, drop_first=True)

    # unknown Attribute is removed and income class label is appended in the end
    dataframe = df_2.drop(['fnlwgt'], axis=1)
    dataframe = dataframe[[
        c for c in dataframe if c not in ['income']] + ['income']]

    dataframe.to_csv(file_path+"adult.csv", header=True, index=None)


if __name__ == "__main__":
    read_adult_data()
