from surprise import SVD, accuracy, SVDpp
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import time
from surprise.model_selection import train_test_split


def df_rating_cleaning(df):
    print('Number of null value before cleaning:')
    print(df.isnull().sum())
    df = df.dropna()
    #Drop the timestamp column as we don't need for SVD
    df = df.drop(columns='timestamp')

    print('Number of null value after cleaning and dropping timestamp column:')
    print(df.isnull().sum())
    return df

def train_test_splitting(df):
    reader = Reader (rating_scale=(0.5,5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.20)
    return data, trainset, testset

def SVD_calculation(data , trainset, testset, time, cv):
    start = time.time()
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    #svd_accuracy = accuracy.rmse(predictions)
    cross_validate_svd_dict = cross_validate(algo, data, measures = ['RMSE'],cv=cv,verbose=True)
    end = time.time()
    time = end-start
    
    return time, cross_validate_svd_dict


def SVDpp_calculation(data , trainset, testset, time, cv):
    start = time.time()
    algo = SVDpp()
    algo.fit(trainset)
    predictions = algo.test(testset)
    cross_validate_svdpp_dict = cross_validate(algo, data, measures = ['RMSE'],cv=cv,verbose=True)
    end = time.time()
    time = end-start
    
    return time, cross_validate_svdpp_dict


