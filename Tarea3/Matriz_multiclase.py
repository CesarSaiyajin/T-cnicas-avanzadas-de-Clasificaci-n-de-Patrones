import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_multiclase(model, X_test, y_test, title="Matriz de confusión"):
    """
    Genera la matriz de confusión visual y el reporte de clasificación.

    Parámetros:
    - model: modelo entrenado (SVM, k-NN, etc.)
    - X_test: conjunto de características de prueba
    - y_test: etiquetas reales de prueba
    - title: título del gráfico
    """
    # Predicciones
    y_pred = model.predict(X_test)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    # Graficar matriz de confusión
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(title, fontsize=14)
    plt.xlabel("Predicción", fontsize=12)
    plt.ylabel("Clase real", fontsize=12)
    plt.show()

    # Imprimir reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

