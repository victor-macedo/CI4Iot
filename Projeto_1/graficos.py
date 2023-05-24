import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
        elif i > len(df.index)-2:
            i = len(df.index)-2
        df[coluna][i] = (df[coluna][i+1] + df[coluna][i-1])/2
    return df


df_original = pd.read_csv("../CI4Iot/Projeto_1/Dataset/Lab6-Proj1_Dataset.csv")
k = 0.5
for colunas in df_original.columns:
    df = previous_out(df_original, colunas, k)
#colunas = ['Anchor_Ratio', 'Transmission_Range', 'Node_Density', 'Step_Size', 'Iterations', 'ESLE']
#sns.relplot(df, x = 'Anchor_Ratio', y = 'Transmission_Range', hue = 'Node_Density',
#             style='Step_Size', size = "Iterations")
fig, axs = plt.subplots(6)
fig.suptitle('Dataset')
axs[0].plot(df.loc[:,"Anchor_Ratio"])
axs[1].plot(df.loc[:,"Transmission_Range"])
axs[2].plot(df.loc[:,"Node_Density"])
axs[3].plot(df.loc[:,"Step_Size"])
axs[4].plot(df.loc[:,"Iterations"])
axs[5].plot(df.loc[:,"ESLE"])
print(len(df.index))
plt.show()

