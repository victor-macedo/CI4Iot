import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as sk
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

iris_df = pd.read_csv("Lab5/Dataset/iris/iris.data",names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species'])

sns.relplot(iris_df, x = 'Sepal_Length', y = 'Sepal_Width', hue = 'Species',
             style='Species', size = "Petal_Width")

X_train ,X_test, Y_train, Y_test = sk.train_test_split(iris_df.loc[:,['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']],
                                                       iris_df.loc[:,['Species']],test_size= 0.28, random_state=42)
X_test ,X_val, Y_test, Y_val = sk.train_test_split(X_test,Y_test,test_size= 0.08, random_state=42)
#Train de 72%, test de 20% e validation de 8%
#rede = LinearRegression().fit(X_train,Y_train)


rede = MLPClassifier(random_state=1,activation='tanh', max_iter=1000).fit(X_train, Y_train)
print(rede.score(X_train,Y_train))

