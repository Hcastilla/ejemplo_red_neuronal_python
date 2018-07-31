import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import preprocessing  

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

c = ['code','Clump', 'Uniformity_', 'Uniformity', 'Marginal', 
  'Single_Epithelial', 'Bare', 'Bland', 'Normal_Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv(url, names=c)
data = data.replace('?', 0)

entradas = data.iloc[:, 1:10]
salidas = data.iloc[:, 10:11]


X_train, X_test, y_train, y_test = train_test_split(entradas, salidas, test_size = 0.30)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=100)


mlp.fit(X_train, y_train.values.ravel()) 

resultados = mlp.predict(X_test)  
print(confusion_matrix(y_test, resultados))  
print(classification_report(y_test, resultados))

