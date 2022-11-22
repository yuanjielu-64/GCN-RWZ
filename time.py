import pandas as pd
import numpy as np
df = pd.read_csv("data/tyson/speed.csv", index_col= None)

df['time'] = pd.to_datetime(df['time'])

time = np.zeros(shape=df.shape)
for i in range(len(df)):
    a = df['time'].iloc[i]
    week = a.day_of_week
    hour = a.hour
    min = a.minute
    time[i] = (hour * 12 + int(min / 5)) / (24 * 12)

print("")