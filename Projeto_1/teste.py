##Algoritimo para teste da rede
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from joblib import load
from sklearn import metrics


def find_out(df, coluna, k):      # Funcao para detetar os outliners, retorna vetor com os index outliners
    u = 0
    v = 0
    out = []            # Vetor com os index outliners
    for i in range(0, len(df.index)):
        u += df[coluna][i]     # Calculo da media
    u = u/len(df.index)
    for i in range(0, len(df.index)):
        v += (df[coluna][i]-u)**2           # Calculo da somatoria da variancia
    v = np.sqrt(v/len(df.index))             # Calculo final variancia
    for i in range(0, len(df.index)):
        if abs(df[coluna][i]-u) > k*v:
            out.append(i)
    return out



def interpolation_out(df,coluna,k):    # Funcao para substituir o outliner pelo valor interpolado
    out = find_out(df,coluna,k)
    for i in out:
        if i < 1:
            i = 1
        elif i > len(df.index)-2:   
            i = len(df.index)-2
        df[coluna][i] = (df[coluna][i+1] + df[coluna][i-1])/2
    return df


df_original = pd.read_csv("../CI4Iot/Projeto_1/Dataset/Lab6-Proj1_TestSet.csv")

### Pre processamento do dataset
## Loop para remover os outliers do dataset
k = 0.5
for colunas in df_original.columns:
    df = interpolation_out(df_original, colunas, k)
##Loop para normalizar os dados
for colunas in df_original.columns:
    df[colunas] = (df[colunas] -  df[colunas].min()) / (df[colunas].max() - df[colunas].min()) 

X = df.copy()
X = X.drop([colunas], axis =1)

Y = df.loc[:,df.columns[5]]
rede = load('rede.joblib')

print("Accuracy test: ", rede.score(X,Y))

Y_pred = rede.predict(X)

print("RMSE: ", np.sqrt(metrics.mean_squared_error(Y,Y_pred)))