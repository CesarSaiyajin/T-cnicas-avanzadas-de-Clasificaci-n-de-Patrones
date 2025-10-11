import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluar_matriz_confusion(conf_matrix):
    """
    Recibe una matriz de confusión 2x2 y calcula medidas de desempeño junto con la gráfica.

    conf_matrix: np.array con formato sklearn:
                 [[TN, FP],
                  [FN, TP]]
    """
    
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


    print("MEDIDAS DE DESEMPEÑO:\n")
    print(f" Accuracy = (TP + TN) / (TP + TN + FP + FN) = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = {accuracy:.4f}")
    print(f" Error Rate = 1 - Accuracy = 1 - {accuracy:.4f} = {error_rate:.4f}")
    print(f" Recall (Sensitivity) = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.4f}")
    print(f" Specificity = TN / (TN + FP) = {TN} / ({TN} + {FP}) = {specificity:.4f}")
    print(f" Balanced Accuracy = (Recall + Specificity) / 2 = ({recall:.4f} + {specificity:.4f}) / 2 = {balanced_accuracy:.4f}")
    print(f" Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f}")
    print(f" F1-Score = 2 * (Precision * Recall) / (Precision + Recall) = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1_score:.4f}")
    print(f" MCC = (TP*TN – FP*FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))")
    print(f"     = ({TP}*{TN} – {FP}*{FN}) / √(({TP}+{FP})({TP}+{FN})({TN}+{FP})({TN}+{FN})) = {mcc:.4f}\n")


    plt.figure(figsize=(6,4))
    sns.heatmap(conf_custom, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred. Pos", "Pred. Neg"],   
                yticklabels=["Real Pos", "Real Neg"])
    plt.title("Matriz de Confusión (formato [TP,FN]/[FP,TN])")
    plt.ylabel("Valor Real")
    plt.xlabel("Predicción")
    plt.show()

    # Diccionario para usar después si es necesario
    metricas = {
        "Accuracy": accuracy,
        "Error Rate": error_rate,
        "Recall": recall,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "F1-Score": f1_score,
        "MCC": mcc
    }
    return metricas

# Configurar matriz de confusión
conf_matrix = np.array([[39, 3],   # TN, FP
                        [4, 11]])  # FN, TP

resultados = evaluar_matriz_confusion(conf_matrix)