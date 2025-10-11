import pandas as pd

def hold_out_estratificado(df, proporcion = 0.8, target_col = 'Class', seed = 42):

    '''FUnción para validar el conjunto de datos con Hold Out Estratificado'''

    clases = df[target_col].unique()
    train_list = []
    test_list = []

    for clase in clases:
        # Crear un datarame para cada clase del df
        df_clase = df[df[target_col] == clase].reset_index(drop = True)
        n_filas = int(len(df_clase) * proporcion)

        # Ya se separo la clase pero hay que mezclar todo
        df_clase =df_clase.sample(frac = 1, random_state = seed).reset_index(drop = True)

        # Del df de cada clase mandar las partes a entrenaniento y a prueba
        train_list.append(df_clase.iloc[:n_filas, :]) # primer 80%
        test_list.append(df_clase.iloc[n_filas:, :])  # ultimo 20%

    # Generar los dataframes finales
    train_df = pd.concat(train_list).reset_index(drop = True)
    test_df = pd.concat(test_list).reset_index(drop = True)

    # Ya quedó unido pero puede tener orden y volvemos a mezclar
    train_df = train_df.sample(frac=1, random_state = seed)
    test_df = test_df.sample(frac = 1, random_state = seed)

    # separar X_train, X_Test, Y_train, Y_test
    X_Train = train_df.drop(columns = target_col, axis = 1)
    Y_Train = train_df[target_col]
    X_Test = test_df.drop(columns = target_col, axis = 1) 
    Y_Test = test_df[target_col]

    return X_Train, Y_Train, X_Test, Y_Test    