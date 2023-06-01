import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from simpful import *
from joblib import dump
from sklearn import metrics
from sklearn.neural_network import MLPRegressor



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


def remove_out(df, coluna, k):     # Funcao para remover os outliners
    out = find_out(df, coluna, k)
    for i in out:
        df = df.drop(i)
    return df


def previous_out(df,coluna,k):    # Funcao para substituir o outliner pelo valor anterior
    out = find_out(df,coluna,k)
    for i in out:
        if i != 0:          
            df[coluna][i] = df[coluna][i-1]
        #else:
         #   df[coluna][i] = df[coluna][i+1]
    return df


def interpolation_out(df,coluna,k):    # Funcao para substituir o outliner pelo valor interpolado
    out = find_out(df,coluna,k)
    for i in out:
        if i < 1:
            i = 1
        elif i > len(df.index)-2:   
            i = len(df.index)-2
        df[coluna][i] = (df[coluna][i+1] + df[coluna][i-1])/2
    return df


def plot_dados(df,col):
    n=0
    l=0
    coluna = df_original.columns
    fig, axs = plt.subplots(int(math.ceil(len(coluna)/col)),col)
    fig.suptitle('Dataset')
    for i in coluna:
        axs[n,l].plot(df.loc[:,i])
        axs[n,l].set_title(i)
        if n == (len(coluna)/col - 1):
            n = 0
            l += 1
        else:
            n += 1

    print(len(df.index))
    plt.show()


df_original = pd.read_csv("../CI4Iot/Projeto_1/Dataset/Lab6-Proj1_Dataset.csv")

### Pre processamento do dataset
## Loop para remover os outliers do dataset
k = 1.5
for colunas in df_original.columns:
    df = interpolation_out(df_original, colunas, k)


##Loop para padronizar os dados
for colunas in df_original.columns:
    df[colunas] = (df[colunas] -  df[colunas].min()) / (df[colunas].max() - df[colunas].min()) #Normalizacao
    #df[colunas] = df[colunas] - df[colunas].mean() / df[colunas].std()  #Z-score




# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS_N = FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
S_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
S_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Memory", LinguisticVariable([S_1, S_2, S_3], concept="Memory usage", universe_of_discourse=[0,1]))

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
F_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
F_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Processor", LinguisticVariable([F_1, F_2,F_3], concept="Processor load", universe_of_discourse=[0,1]))

# Define output fuzzy sets and linguistic variable
T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
T_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
T_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Load", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,1]))

# Define fuzzy rules
R1 = "IF (Memory IS low) AND (Processor IS low) THEN (Load IS low)"
R2 = "IF (Memory IS low) AND (Processor IS normal) THEN (Load IS low)"
R3 = "IF (Memory IS low) AND (Processor IS high) THEN (Load IS normal)"
R4 = "IF (Memory IS normal) AND (Processor IS low) THEN (Load IS low)"
R5 = "IF (Memory IS normal) AND (Processor IS normal) THEN (Load IS normal)"
R6 = "IF (Memory IS normal) AND (Processor IS high) THEN (Load IS fast)"
R7 = "IF (Memory IS high) AND (Processor IS low) THEN (Load IS normal)"
R8 = "IF (Memory IS high) AND (Processor IS normal) THEN (Load IS fast)"
R9 = "IF (Memory IS high) AND (Processor IS high) THEN (Load IS fast)"
FS_N.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

# Set antecedents values
FS_N.set_variable("Memory", 0.4)
FS_N.set_variable("Processor", 0.8)

# Perform Mamdani inference and print output
print(FS_N.Mamdani_inference(["Load"]))