import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
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



##Divisao do Dataset 
#Train de 80%, test de 10% e validation de 10%
X_train ,X_test, Y_train, Y_test = sk.train_test_split(df.loc[:,[df.columns[0],df.columns[1],df.columns[2],df.columns[3],
                                                                     df.columns[4]]],df.loc[:,df.columns[5]],test_size= 0.2, random_state=42)
X_test ,X_val, Y_test, Y_val = sk.train_test_split(X_test,Y_test,test_size= 0.5, random_state=42)


rede = MLPRegressor(random_state=1,hidden_layer_sizes=(10,),activation='relu',solver ='lbfgs',learning_rate = 'adaptive', max_iter=10000).fit(X_train, Y_train)

save = pickle.dumps(rede)
dump(rede, 'rede.joblib')
print("Accuracy test: ", rede.score(X_test,Y_test))
print("Accuracy validation: ", rede.score(X_val,Y_val))

Y_pred = rede.predict(X_val)

print("RMSE: ", np.sqrt(metrics.mean_squared_error(Y_val,Y_pred)))
print("RMSE Maps: ", np.sqrt(metrics.mean_squared_error(Y_pred+20,Y_pred)))