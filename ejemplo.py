import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import preprocessing  
from sklearn.metrics import accuracy_score


#link de los datos
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

#nombre de las columnas
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#leemos el dataset
irisdata = pd.read_csv(url, names=names)


X = irisdata.iloc[:, 0:4]

y = irisdata.select_dtypes(include=[object])  

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21, tol=0.000000001)


mlp.fit(X_train, y_train.values.ravel()) 

predictions = mlp.predict(X_test)  
print(len(predictions))
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

