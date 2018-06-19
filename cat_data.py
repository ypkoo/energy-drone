import sys
from input_preprocess import *
import pandas as pd

FILE_NUM = 9
FILE_DIR = 'exp'

data_list = []

for i in range(FILE_NUM):
	df = pd.read_csv(FILE_DIR + str(i))
	df = df[df.isnull().any(1)].index # delete if a row contains NaN
	df_list.append(df)

df_concat = pd.concat(df_list)