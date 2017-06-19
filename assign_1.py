import pandas as pd

rainfall_df = pd.read_csv('C:/__backup/kaggle/assign/rainfall.dat', sep='\s+', header=None)

print(rainfall_df.shape[0])
print(rainfall_df.shape[1])
print(rainfall_df.info())
print(rainfall_df.head(2))

print(rainfall_df.columns.values.tolist())