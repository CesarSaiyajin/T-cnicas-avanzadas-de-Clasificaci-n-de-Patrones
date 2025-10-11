import pandas as pd
from sklearn.preprocessing import StandardScaler

def estandarizar_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Estandariza las variables numericas del df
    devuelve un df estandarizado
    '''

    columnas = df.columns
    caracteristicas = columnas[:-1]
    clases = columnas[-1]

    X = df[caracteristicas]
    Y = df[clases]

    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)
    df_escalado = pd.DataFrame(X_escalado, columns = caracteristicas)
    
    df_final = pd.concat([
        df_escalado.reset_index(drop = True),
        Y.reset_index(drop=True)], axis = 1)
    
    return df_final
