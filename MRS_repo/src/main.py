
import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import time
from surprise.model_selection import train_test_split
from  SVD import *
from  data_cleaning import random_choice, rating_cleaning
from NCF import NCF_calculation
def main():
    path='C:\\Users\\kamho\\OneDrive\\Documents\\GitHub\\movieRecommendationSystem\\MRS_repo\\data\\raw\\Data3_movielens\\'
    ratings_filename = 'ratings.csv'
    df_rating = pd.read_csv(os.path.join(path, ratings_filename))

    df_rating = random_choice(df_rating, 0.5)

    df_rating_cleaned = rating_cleaning(df_rating)
    #print(df_rating_cleaned.head())
    
    data, train_set, test_set = train_test_splitting(df_rating_cleaned)

    svd_time, svd_cv = SVD_calculation(data, train_set,test_set, time, 5)
    #print('The accuracy of SVD model:',svd_accuracy)
    print('Time Elapsed in SVD model',svd_time)

    print("Starting cross validation in SVDpp. It may take a while.")
    svdpp_time, svdpp_cv = SVDpp_calculation(data, train_set,test_set, time, 5)
    print('Time Elapsed in SVDpp model',svdpp_time)
    

    NCF_calculation(df_rating_cleaned)

if __name__ == "__main__":
    main()