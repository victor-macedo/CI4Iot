import math
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




# Rede Fuzzy, input memory usage and processor load, output Load
# Create a fuzzy system object
FS_L = FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
S_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
S_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_L.add_linguistic_variable("Memory", LinguisticVariable([S_1, S_2, S_3], concept="Memory usage", universe_of_discourse=[0,1]))

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
F_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
F_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_L.add_linguistic_variable("Processor", LinguisticVariable([F_1, F_2,F_3], concept="Processor load", universe_of_discourse=[0,1]))

# Define output fuzzy sets and linguistic variable
T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
T_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
T_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_L.add_linguistic_variable("Load", LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0,1]))

# Define fuzzy rules
R_L1 = "IF (Memory IS low) AND (Processor IS low) THEN (Load IS low)"
R_L2 = "IF (Memory IS low) AND (Processor IS normal) THEN (Load IS low)"
R_L3 = "IF (Memory IS low) AND (Processor IS high) THEN (Load IS normal)"
R_L4 = "IF (Memory IS normal) AND (Processor IS low) THEN (Load IS low)"
R_L5 = "IF (Memory IS normal) AND (Processor IS normal) THEN (Load IS normal)"
R_L6 = "IF (Memory IS normal) AND (Processor IS high) THEN (Load IS fast)"
R_L7 = "IF (Memory IS high) AND (Processor IS low) THEN (Load IS normal)"
R_L8 = "IF (Memory IS high) AND (Processor IS normal) THEN (Load IS fast)"
R_L9 = "IF (Memory IS high) AND (Processor IS high) THEN (Load IS fast)"
FS_L.add_rules([R_L1, R_L2, R_L3, R_L4, R_L5, R_L6, R_L7, R_L8, R_L9])

############ FIM Da REDE FS_L    ####################################


FS_N = FuzzySystem()
## REDE NETWORK
L_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
L_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
L_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Latency", LinguisticVariable([L_1, L_2, L_3], concept="Latency", universe_of_discourse=[0, 4]))

O_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
O_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
O_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Output", LinguisticVariable([O_1, O_2, O_3], concept="Output Network Throughput", universe_of_discourse=[0, 100]))

# Define output fuzzy sets and linguistic variable
N_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
N_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
N_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Network", LinguisticVariable([N_1, N_2, N_3], universe_of_discourse=[0, 1]))

# Define fuzzy rules

R_N1 = "IF (Latency IS low) AND (Output IS low) THEN (Network IS low)"
R_N2 = "IF (Latency IS low) AND (Output IS normal) THEN (Network IS low)"
R_N3 = "IF (Latency IS low) AND (Output IS high) THEN (Network IS normal)"
R_N4 = "IF (Latency IS normal) AND (Output IS low) THEN (Network IS low)"
R_N5 = "IF (Latency IS normal) AND (Output IS normal) THEN (Network IS normal)"
R_N6 = "IF (Latency IS normal) AND (Output IS high) THEN (Network IS fast)"
R_N7 = "IF (Latency IS high) AND (Output IS low) THEN (Network IS normal)"
R_N8 = "IF (Latency IS high) AND (Output IS normal) THEN (Network IS fast)"
R_N9 = "IF (Latency IS high) AND (Output IS high) THEN (Network IS fast)"

FS_N.add_rules([R_N1, R_N2, R_N3, R_N4, R_N5, R_N6, R_N7, R_N8, R_N9])

########FIM REDE Network ###########################################

#Rede Final
FS_F = FuzzySystem()

G_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
G_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
G_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_F.add_linguistic_variable("Network", LinguisticVariable([G_1, G_2, G_3], concept="Network", universe_of_discourse=[0, 4]))

H_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
H_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
H_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_F.add_linguistic_variable("Load", LinguisticVariable([H_1, H_2, H_3], concept="Load", universe_of_discourse=[0, 100]))

# Define output fuzzy sets and linguistic variable
J_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
J_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
J_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_F.add_linguistic_variable("Result", LinguisticVariable([J_1, J_2, J_3], universe_of_discourse=[0, 1]))

# Define fuzzy rules

R_F1 = "IF (Network IS low) AND (Load IS low) THEN (Result IS low)"
R_F2 = "IF (Network IS low) AND (Load IS normal) THEN (Result IS low)"
R_F3 = "IF (Network IS low) AND (Load IS high) THEN (Result IS normal)"
R_F4 = "IF (Network IS normal) AND (Load IS low) THEN (Result IS low)"
R_F5 = "IF (Network IS normal) AND (Load IS normal) THEN (Result IS normal)"
R_F6 = "IF (Network IS normal) AND (Load IS high) THEN (Result IS fast)"
R_F7 = "IF (Network IS high) AND (Load IS low) THEN (Result IS normal)"
R_F8 = "IF (Network IS high) AND (Load IS normal) THEN (Result IS fast)"
R_F9 = "IF (Network IS high) AND (Load IS high) THEN (Result IS fast)"

FS_F.add_rules([R_F1, R_F2, R_F3, R_F4, R_F5, R_F6, R_F7, R_F8, R_F9])
# Set antecedents values
FS_N.set_variable("Latency", 0.2)
FS_N.set_variable("Output", 0.3)

FS_L.set_variable("Memory", 0.2)
FS_L.set_variable("Processor", 0.3)

FS_F.set_variable("Load", float(FS_L.Mamdani_inference(["Load"])["Load"]))
FS_F.set_variable("Network", float(FS_N.Mamdani_inference(["Network"])["Network"]))

# Perform Mamdani inference and print output
print(FS_L.Mamdani_inference(["Load"]))

print(FS_N.Mamdani_inference(["Network"]))

print(FS_F.Mamdani_inference(["Result"]))



