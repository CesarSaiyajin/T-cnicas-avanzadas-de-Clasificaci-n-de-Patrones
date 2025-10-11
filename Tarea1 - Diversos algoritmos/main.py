import pandas as pd
import os

from generar_particiones import generar_particiones_estratificadas
from modelos import entrenar_modelo, predecir_modelo
from medidas_desempeño import calcula_medidas
from generar_tabla import generar_tabla

base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'datasets', 'crx_limpio.csv')
df = pd.read_csv(csv_path)

particiones = generar_particiones_estratificadas(df, 100)

nombre_modelos = ['1NN', '3NN', '5NN', 'Naive Bayes']
df_promedios = []

for nombre in nombre_modelos:
    medidas = []

    for X_Train, Y_Train, X_Test, Y_Test in particiones:
        modelo = entrenar_modelo(nombre, X_Train, Y_Train)
        Y_pred = predecir_modelo(modelo, X_Test)
        df_medidas = calcula_medidas(Y_Test, Y_pred)
        medidas.append(df_medidas)

    df_promedios.append(generar_tabla(medidas, nombre))

print(pd.concat(df_promedios, axis = 1))

