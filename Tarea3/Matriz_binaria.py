from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, balanced_accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def binary_confusion_metrics_save(model, X_test, y_test, save_dir="matrices_confusion"):
    """
    Calcula matrices de confusión binarias (One-vs-Rest) para cada clase,
    guarda cada matriz como imagen y devuelve un DataFrame con métricas
    incluyendo promedios macro, micro y weighted.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    classes = sorted(y_test.unique())
    y_pred = model.predict(X_test)

    metrics_list = []

    # Inicializar sumas para micro
    sum_tp = sum_fp = sum_tn = sum_fn = 0

    for cls in classes:
        y_true_bin = (y_test == cls).astype(int)
        y_pred_bin = (y_pred == cls).astype(int)

        # Matriz de confusión
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        tn, fp, fn, tp = cm.ravel()

        sum_tp += tp
        sum_fp += fp
        sum_tn += tn
        sum_fn += fn

        # Matriz rotada (TP arriba izquierda)
        cm_rotated = np.array([[tp, fn],
                               [fp, tn]])

        # Guardar imagen
        plt.figure(figsize=(4,3))
        sns.heatmap(cm_rotated, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Positiva', 'Negativa'],
                    yticklabels=['Positiva', 'Negativa'])
        plt.title(f'Matriz de confusión - Clase {cls}')
        plt.xlabel("Predicción")
        plt.ylabel("Clase real")
        plt.tight_layout()
        filename = os.path.join(save_dir, f"matriz_clase_{cls}.png")
        plt.savefig(filename)
        plt.close()  # cerramos la figura para no mostrarla

        # Métricas
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average=None
        )
        accuracy = accuracy_score(y_true_bin, y_pred_bin)
        balanced_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        try:
            roc_auc = roc_auc_score(y_true_bin, y_pred_bin)
        except ValueError:
            roc_auc = np.nan

        metrics_list.append({
            'Clase': cls,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': accuracy,
            'Balanced_Acc': balanced_acc,
            'Precision': precision[1],
            'Recall': recall[1],
            'Specificity': specificity,
            'F1': f1[1],
            'MCC': mcc,
            'ROC_AUC': roc_auc,
            'Support': support[1]
        })

    df_metrics = pd.DataFrame(metrics_list)

    # ===== Promedios =====
    weights = df_metrics['Support'] / df_metrics['Support'].sum()

    # Macro
    macro_avg = df_metrics[['Accuracy','Balanced_Acc','Precision','Recall',
                            'Specificity','F1','MCC','ROC_AUC']].mean()
    macro_avg['Clase'] = 'Macro'

    # Micro
    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    micro_accuracy = (sum_tp + sum_tn) / (sum_tp + sum_tn + sum_fp + sum_fn)
    micro_specificity = sum_tn / (sum_tn + sum_fp) if (sum_tn + sum_fp) > 0 else 0

    micro_avg = pd.Series({
        'Clase': 'Micro',
        'Accuracy': micro_accuracy,
        'Balanced_Acc': (micro_recall + micro_specificity)/2,
        'Precision': micro_precision,
        'Recall': micro_recall,
        'Specificity': micro_specificity,
        'F1': micro_f1,
        'MCC': np.nan,
        'ROC_AUC': np.nan,
        'Support': df_metrics['Support'].sum()
    })

    # Weighted
    weighted_avg = df_metrics[['Accuracy','Balanced_Acc','Precision','Recall',
                               'Specificity','F1','MCC','ROC_AUC']].apply(lambda col: np.average(col, weights=weights))
    weighted_avg['Clase'] = 'Weighted'

    # Combinar todo
    df_final = pd.concat([df_metrics, macro_avg.to_frame().T, micro_avg.to_frame().T, weighted_avg.to_frame().T], ignore_index=True)

    # ===== Guardar micro matriz como imagen =====
    cm_micro = np.array([[sum_tp, sum_fn],
                         [sum_fp, sum_tn]])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm_micro, annot=True, fmt='d', cmap='Oranges', cbar=False,
                xticklabels=['Positiva', 'Negativa'],
                yticklabels=['Positiva', 'Negativa'])
    plt.title("Matriz micro combinada")
    plt.xlabel("Predicción")
    plt.ylabel("Clase real")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "matriz_micro.png"))
    plt.close()

    print(f"Se guardaron las matrices de confusión en la carpeta: {save_dir}")
    return df_final



