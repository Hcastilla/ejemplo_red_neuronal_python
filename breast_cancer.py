import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn import preprocessing  
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import accuracy_score


#url de los datos
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

#nombre de las columnas
names = ['code','Clump', 'Uniformity_size', 'Uniformity_shape', 'Marginal_Adhesion', 
  'Single_Epithelial_size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

#leemos los datos, el dataset esta dañado por lo que debemos arreglar los datos
datos = pd.read_csv(url, names=names)
datos = datos.replace({'?':0, "'":""})
datos = datos.apply(lambda x: pd.to_numeric(x))


#obtenemos los datos de entrada
X = datos.iloc[:, 1:10].values


#obtenemos la fila de resultados
y = datos.Class.values

#obtenemos los datos para el entramiento y las pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


#escalamos nuestros datos, esto hace que los datos sean lo mas reales posibles
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


n_iteraciones = 9
learning_rate = 0.001
hidden_layers = 50
max_iter = 150

for i in range (0,9):
  #definimos la red neuronal
  mlp = MLPClassifier(hidden_layer_sizes=(hidden_layers,hidden_layers,hidden_layers), max_iter=max_iter, alpha=0.0001,
                      solver='sgd', random_state=21, tol=0.000000001, learning_rate_init= learning_rate)

  print("Taza de aprendizaje :", learning_rate)
  print("Numero de capas ocultas :", hidden_layers)
  print("Maximo de iteraciones :", max_iter )

  #entrenamos la red
  mlp.fit(X_train, y_train) 

  predicciones = mlp.predict(X_test)  

  #matrix de confucion
  c_matrix = confusion_matrix(y_test, predicciones)

  #reporte de calsificacion
  c_reporte = classification_report(y_test, predicciones)

  #nivel de prediccion
  n_prediccion = accuracy_score(y_test, predicciones)

  print(c_matrix)
  print(c_reporte)
  print(n_prediccion)


  skplt.metrics.plot_confusion_matrix(y_test, predicciones, normalize=True)
  plt.show()

  learning_rate += 0.05
  hidden_layers += 100
  max_iter += 100
  print("------------------------------------------------------------------------------------------------")


  plt.plot(mlp.loss_curve_)
  plt.show()