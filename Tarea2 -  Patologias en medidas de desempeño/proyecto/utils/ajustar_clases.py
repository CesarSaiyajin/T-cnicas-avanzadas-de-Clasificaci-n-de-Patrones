def ajustar_clases(df):
    '''
    AJusta las clases para funcinoar con 0 y 1
    '''
    columnas = df.columns
    clases = columnas[-1]

    df[clases] = df[clases].map({1:0, 2:1})

    return df.sample(frac = 1, random_state = 42).reset_index(drop = True)