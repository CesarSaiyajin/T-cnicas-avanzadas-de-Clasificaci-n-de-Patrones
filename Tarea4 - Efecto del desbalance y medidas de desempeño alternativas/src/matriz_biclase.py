import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def imprime_matriz_biclase(y_test, y_pred, dataset, k):
    '''
    Calculas las matriz de confusión 
    TP FN
    FP TN
    '''
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    matriz = pd.DataFrame(
        [[tp, fn],[fp, tn]],
        index = ['Real Positivo', 'Real Negativo'],
        columns=['Predicción Positivo', 'Predicción Negativo'])
    
    plt.figure(figsize=(5, 5))
    sn.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{dataset}_{k} vecinos')
    plt.savefig(f'img/{dataset}_{k}_vecinos')
    plt.close()


