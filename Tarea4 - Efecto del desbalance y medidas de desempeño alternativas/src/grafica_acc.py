import matplotlib.pyplot as plt

def grafica_acc(df, dataset):

    # Seleccionar la fila "Acc - BA"
    fila = df[df[df.columns[0]] == 'Acc - BA']

    x = fila.columns[1:].astype(int)  
    y = fila.iloc[0, 1:].astype(float) 

    plt.figure(figsize=(7,5))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
    plt.xticks(x)
    plt.title(f"Accuracy - Balanced Accuracy '{dataset}'")
    plt.xlabel('k (vecinos)')
    plt.ylabel('Acc - BA')
    plt.grid(True)
    plt.savefig(f'img/Deltaacc_{dataset}')
    plt.close()
