import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as sk
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing

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

def plot_dados(df):
    n=0
    coluna = df_original.columns
    fig, axs = plt.subplots(len(coluna))
    fig.suptitle('Dataset')
    for i in coluna:
        axs[n].plot(df.loc[:,i])
        axs[n].set_title(i)
        n += 1

    print(len(df.index))
    plt.show()

df_original = pd.read_csv("../CI4Iot/Projeto_1/Dataset/Lab6-Proj1_Dataset.csv")

#colunas = ['Anchor_Ratio', 'Transmission_Range', 'Node_Density', 'Step_Size', 'Iterations', 'ESLE']
###Pré processamento do dataset
## Loop para remover os outliers do dataset
k = 0.5
for colunas in df_original.columns:
    df = interpolation_out(df_original, colunas, k)
##Loop para normalizar os dados
for colunas in df_original.columns:
    df[colunas] = (df[colunas] -  df[colunas].min()) / (df[colunas].max() - df[colunas].min()) 
    #df[colunas] = df[colunas] - df[colunas].mean() / df[colunas].std()

#plot_dados(df)
##Divisão do Dataset 
#Train de 72%, test de 20% e validation de 8%
X_train ,X_test, Y_train, Y_test = sk.train_test_split(df.loc[:,[df.columns[0],df.columns[1],df.columns[2],df.columns[3],
                                                                     df.columns[4]]],df.loc[:,df.columns[5]],test_size= 0.28, random_state=42)
X_test ,X_val, Y_test, Y_val = sk.train_test_split(X_test,Y_test,test_size= 0.28, random_state=42)


rede = MLPRegressor(random_state=1,activation='identity',solver ='lbfgs', max_iter=10000).fit(X_train, Y_train)

print("Accuracy test: ", rede.score(X_test,Y_test))
print("Accuracy validation: ", rede.score(X_val,Y_val))

