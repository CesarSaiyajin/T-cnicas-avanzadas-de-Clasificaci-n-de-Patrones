import pandas as pd

def generar_tabla(lista_medidas, nombre_modelo):

    '''
    Toma una lista de dataframes con las medidas de desempeño y el nombre del modelo
    y devuelve un datafarame con los proedios de cada medida
    '''

    lista_medidas = pd.concat(lista_medidas).reset_index(drop = True)
    lista_medidas['Iteración'] = lista_medidas.groupby('Medida').cumcount() 
    resultado_final = lista_medidas.pivot(index = 'Iteración', columns = 'Medida', values = 'Valor').reset_index(drop = True)

    promedios = resultado_final.mean().to_frame(name = f'{nombre_modelo}')

    return promedios
