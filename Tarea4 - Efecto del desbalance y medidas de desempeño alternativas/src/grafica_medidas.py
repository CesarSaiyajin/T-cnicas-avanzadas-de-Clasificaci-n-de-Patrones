import matplotlib.pyplot as plt

def grafica_medidas(df, dataset):

    for medida in ['Kappa', 'MCC', 'B_Acc', 'F1']:

        # Seleccionar la fila "Acc - BA"
        fila = df[df[df.columns[0]] == medida]

        x = fila.columns[1:].astype(int)  
        y = fila.iloc[0, 1:].astype(float) 

        plt.figure(figsize=(7,5))
        plt.plot(x, y, marker='o', linestyle='-', linewidth=2)
        plt.title(f"{medida} '{dataset}'")
        plt.xlabel('k (vecinos)')
        plt.ylabel('Acc - BA')
        plt.grid(True)
        plt.savefig(f'img/{medida}_{dataset}')
        plt.close()
