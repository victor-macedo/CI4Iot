import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def remove_out(df,coluna,k):
    u = 0
    v = 0
    for i in range (0, len(df.index)):      
        u += df[coluna][i]     #Calculo da média
    u = u/len(df.index)
    for i in range (0, len(df.index)):
        v += (df[coluna][i]-u)**2 #Calculo da somatoria da variancia
    v = np.sqrt(v/len(df.index))             #Calculo final variancia
    for i in range(0, len(df.index)):
        if abs(df[coluna][i]-u) > k*v:
            df = df.drop(df.index[i])
    return df


    

df = pd.read_csv('../CI4Iot/Lab3_DataSets/EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv')
df["DATE"] = pd.to_datetime(df["Time (UTC)"])
#plt.plot(df)
df = remove_out(df,"High",0.6)
plt.plot(df["DATE"],df["High"])
plt.show()
