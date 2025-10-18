import pandas as pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut

from src.medidas_desempeno import calcula_medidas_2
from src.matriz_biclase import imprime_matriz_biclase
from src.grafica_medidas import grafica_medidas

base_path = os.path.dirname(__file__)
csv_path_campus = os.path.join(base_path, 'data', 'clean_data', 'campus_selection_limpio.csv')
df = pd.read_csv(csv_path_campus)

dataset = 'Campus_selection'

df_resultados = pd.DataFrame()
df_resultados[f'Medidas_{dataset}'] =  ['Kappa', 'MCC', 'B_Acc', 'F1']

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

n = df['status'].value_counts().max()

for k in range(1, n+1):
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

    medidas = calcula_medidas_2(y_true, y_pred)
    for i, medida in enumerate(medidas.keys()):
        df_resultados.loc[i, f'{k}'] = medidas[medida]
    
    if k in [1, 5, 10, n]:
        imprime_matriz_biclase(y_true, y_pred, dataset, k)

print(df_resultados)

grafica_medidas(df_resultados, dataset)

