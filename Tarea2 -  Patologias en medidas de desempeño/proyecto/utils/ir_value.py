import pandas as pd

def ir_value(df: pd.DataFrame, columna: str) -> float:
    '''
    Calcula el valor de IR del conjunto de datos.
    '''

    clases = df[columna].value_counts()
    mayor = clases.max()
    menor = clases.min()
    ir_value = mayor / menor

    if ir_value > 1.5:
        tipo = f'No Balanceado'
    else:
        tipo = f'Balanceado'

    return f"{tipo} - IR: {ir_value} - card_max: {mayor} - card_min: {menor}."