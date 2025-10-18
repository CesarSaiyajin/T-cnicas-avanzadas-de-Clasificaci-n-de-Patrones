import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, GridSearchCV

from src.medidas_desempeno import calcula_medidas
from src.matriz_biclase import imprime_matriz_biclase
from src.grafica_acc import grafica_acc

base_path = os.path.dirname(__file__)
csv_path_campus = os.path.join(base_path, 'data', 'clean_data', 'campus_selection_limpio.csv')
df = pd.read_csv(csv_path_campus)

dataset = 'campus_selection'
k_list = [1, 3, 5, 10, 15, 20]

df_resultados = pd.DataFrame()
df_resultados[f'Medidas_{dataset}'] =  ['Accuracy', 'BA','F1', 'Acc - BA']

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

for k in k_list:
    knn_model = KNeighborsClassifier(n_neighbors = k)
    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        knn_model.fit(X_train, y_train)

        y_true.append(y_test[0])
        y_pred.append(knn_model.predict(X_test)[0])

    medidas = calcula_medidas(y_true, y_pred)
    for i, medida in enumerate(medidas.keys()):
        df_resultados.loc[i, f'{k}'] = medidas[medida]
    
    imprime_matriz_biclase(y_true, y_pred, dataset, k)

print('_'*75)
print(df_resultados)

grafica_acc(df_resultados, dataset)

