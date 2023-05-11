import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_out(df,coluna,k):      #Funcao para detetar os outliners, retorna vetor com os index outliners
    u = 0                           
    v = 0
    out = []            #Vetor com os index outliners
    for i in range (0, len(df.index)):     
        u += df[coluna][i]     #Calculo da média
    u = u/len(df.index)
    for i in range (0, len(df.index)):
        v += (df[coluna][i]-u)**2 #Calculo da somatoria da variancia
    v = np.sqrt(v/len(df.index))             #Calculo final variancia
    for i in range(0, len(df.index)):
        if abs(df[coluna][i]-u) > k*v:
            out.append(i)
    return out

def remove_out(df,coluna,k):    #Funcao para remover os outliners
    out = find_out(df,coluna,k)
    for i in out:
        df = df.drop(i)
    return df

def previous_out(df,coluna,k):    #Funcao para substituir o outliner pelo valor anterior
    out = find_out(df,coluna,k)
    for i in out:
        i += 1
        df[coluna][i] = df[coluna][i-1]
    return df

def interpolation_out(df,coluna,k):    #Funcao para substituir o outliner pelo valor interpolado
    out = find_out(df,coluna,k)
    for i in out:
        if i < 1:
            i = 1
        elif i > 1271:
            i = 1271
        df[coluna][i] = (df[coluna][i+1] + df[coluna][i-1])/2
    return df

df = pd.read_csv('../CI4Iot/Lab3/Lab3_DataSets/DCOILBRENTEUv2.csv')
#df = pd.read_csv('../CI4Iot/Lab3_DataSets/EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv')
#plt.plot(df)
coluna = "DCOILBRENTEU"
#df = interpolation_out(df,coluna,0.6)
df["var_prev"] = df[coluna][1] - df[coluna][0] 
for i in range (1, len(df.index)):
    df["var_prev"][i] = df[coluna][i] - df[coluna][i-1]

print(df)
plt.hist(df["var_prev"])

plt.show()
