import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent.parent

def evaluar_matriz_confusion(y_test, y_pred, etiquetas, algoritmo, clase_positiva):
    """
    Recibe una matriz de confusión 2x2 y calcula medidas de desempeño junto con la gráfica.

    conf_matrix: np.array con formato sklearn:
                 [[TN, FP],
                  [FN, TP]]
    """

    nombre_archivo = f'{algoritmo}_positiva_{clase_positiva}.png'
    ruta= os.path.join(BASE_DIR, 'imagenes', nombre_archivo)

    conf_matrix = confusion_matrix(y_test, y_pred, labels = etiquetas)
    
    TN, FP, FN, TP = conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]

    
    conf_custom = np.array([[TP, FN],
                            [FP, TN]])


    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0        # Sensibilidad (para la clase positiva)
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0   # Especificidad (para la clase negativa)
    balanced_accuracy = (recall + specificity) / 2
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    mcc_num = (TP * TN) - (FP * FN)
    mcc_den = np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
    mcc = mcc_num / mcc_den if mcc_den != 0 else 0

    plt.figure(figsize=(6,4))
    sns.heatmap(conf_custom, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred. Pos", "Pred. Neg"], yticklabels=["Real Pos", "Real Neg"], cbar = False)
    plt.title("Matriz de Confusión (formato [TP,FN]/[FP,TN])")
    plt.ylabel("Valor Real")
    plt.xlabel("Predicción")
    plt.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.show()

    # Diccionario para usar en una función que guarde los resultados en un dataframe
    medidas = {
        "Accuracy": np.round(accuracy, 4),
        "Error Rate": np.round(error_rate,4),
        "Recall": np.round(recall, 4),
        "Specificity": np.round(specificity,4),
        "Balanced Accuracy": np.round(balanced_accuracy, 4),
        "Precision": np.round(precision,4),
        "F1-Score": np.round(f1_score,4),
        "MCC": np.round(mcc,4)
    }
    return pd.DataFrame(list(medidas.items()), columns = ['Medida', 'Valor'])

def genera_tabla(df_resultados):
    '''
    Recibe una lista de DataFrames con los resultados de las medidas de los modelos seleccionados, y lo guarda en una carpeta.
    '''
    df_resultados = pd.concat(df_resultados).reset_index(drop = True)
    df_resultados['Iteración'] = df_resultados.groupby('Medida').cumcount()
    tabla_final = df_resultados.pivot(index = 'Iteración', columns = 'Medida', values = 'Valor').reset_index(drop = True)

    return tabla_final

def guarda_tabla(tabla_final, positiva):

    '''
    Guarda la tabla de resultados por tipo de clase positiva en una carpeta.
    '''
    nombre_tabla =  f'Clase_positiva_{positiva}'
    ruta= os.path.join(BASE_DIR, 'tablas', nombre_tabla)

    tabla_final.to_csv(ruta, index = False)