from input_preprocess import *
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time

# df = get_df('0424_merged_actv.csv')
df = get_df('exp_power_actv_history.csv')
df_corr = df[['power','power_shifted_by_1', 'power_shifted_by_2', 'act_v', 'vel', 'acc']]
# df_corr = df[['power', 'vel', 'acc', 'rc0', 'rc1', 'rc2', 'rc3']]

f, ax = plt.subplots(figsize=(10, 8))
corr = df_corr.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, annot=True, cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)

plt.show()