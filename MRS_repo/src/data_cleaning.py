import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

def rating_cleaning(df):
    print('Number of null value before cleaning:')
    print(df.isnull().sum())
    df = df.dropna()

    df["timestamp"] = df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1e3))

    print('Number of null value after cleaning and dropping timestamp column:')
    print(df.isnull().sum())
    return df

def random_choice(df, pct_choice):
    rand_userIds = np.random.choice(df['userId'].unique(), 
                                size=int(len(df['userId'].unique())*pct_choice), 
                                replace=False)

    df = df.loc[df['userId'].isin(rand_userIds)]

    print(df)
    print('There are {} rows of data from {} users'.format(len(df), len(df['userId'].unique())))

    return df