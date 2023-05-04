import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def remove_out():
    

df = pd.read_csv('../CI4Iot/Lab3_DataSets/EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv')
df["DATE"] = pd.to_datetime(df["Time (UTC)"])
#plt.plot(df)
plt.plot(df["DATE"],df["High"])
plt.show()
