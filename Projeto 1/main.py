import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as sk
from sklearn import svm


def find_out(df, coluna, k):      # Funcao para detetar os outliners, retorna vetor com os index outliners
    u = 0
    v = 0
    out = []            # Vetor com os index outliners
    for i in range(0, len(df.index)):
        u += df[coluna][i]     # Calculo da média
    u = u/len(df.index)
    for i in range(0, len(df.index)):
        v += (df[coluna][i]-u)**2           # Calculo da somatoria da variancia
    v = np.sqrt(v/len(df.index))             # Calculo final variancia
    for i in range(0, len(df.index)):
        if abs(df[coluna][i]-u) > k*v:
            out.append(i)
    return out


def remove_out(df, coluna, k):     # Funcao para remover os outliners
    out = find_out(df, coluna, k)
    for i in out:
        df = df.drop(i)
    return df


df = pd.read_csv("Dataset/Lab6-Proj1_Dataset.csv")

colunas = ['Anchor_Ratio', 'Transmission_Range', 'Node_Density', 'Step_Size', 'Iterations', 'ESLE']
k = 1

out = remove_out(df, 'ESLE', k)

print(len(out))

