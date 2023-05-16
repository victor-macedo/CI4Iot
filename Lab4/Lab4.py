import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.model_selection as sk
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Lab4/Dataset/iris/iris.data",names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width','Species'])

sns.relplot(df, x = 'Sepal_Length', y = 'Sepal_Width', hue = 'Species',
             style='Species', size = "Petal_Width")
#plt.grid()
#plt.show()

for i in range (0, len(df.index)):
    if df["Species"][i] == "Iris-setosa":
        df["Species"][i] = 0
    elif df["Species"][i] == "Iris-versicolor":
        df["Species"][i] = 1
    elif df["Species"][i] == "Iris-virginica":
        df["Species"][i] = 2

X_train ,X_test, Y_train, Y_test = sk.train_test_split(df.loc[:,['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']],
                                                       df.loc[:,['Species']],train_size= 0.66, random_state=42)

rede = LinearRegression().fit(X_train, Y_train)

rede.predict(X_test)
print(rede.score(X_test,Y_test))

