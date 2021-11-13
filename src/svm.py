import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from itertools import chain, combinations

# datatypes and do necessary data transformation after loading the raw data to the dataframe.
df_data = pd.read_csv("../mood_data.csv", sep=',',header=None, encoding='unicode_escape', thousands=',')

df_data.columns= df_data.iloc[0]
df_data = df_data.drop(0)

clf = svm.SVC()

print(df_data)