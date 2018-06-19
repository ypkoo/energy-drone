import sys
from input_preprocess import *
import pandas as pd

# filename = sys.argv[1]
# data = pd.read_csv(filename)
df = concat('exp', 9)
df = get_current(df)

df.to_csv('exp'+"_power.csv", index=False, sep=",")