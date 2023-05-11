import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as sk
from sklearn import svm

df = pd.read_csv("Lab4/Dataset/iris/iris.data",names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species'])

sns.relplot(df, x = 'Sepal_Length', y = 'Sepal_Width', hue = 'Species',
             style='Species', size = "Petal_Width")
#plt.grid()
#plt.show()

for i in range (1, len(df.index)):
    if df["Species"][i] == "Iris-setosa":
        df["Species"][i] = 0
    elif df["Species"][i] == "Iris-versicolor":
        df["Species"][i] = 1
    elif df["Species"][i] == "Iris-virginica":
        df["Species"][i] = 2

Train ,Test = sk.train_test_split(df, train_size= 0.66, random_state=42)
print(Test)
#rede = svm().fit(Train, Test)
#rede.predict()



