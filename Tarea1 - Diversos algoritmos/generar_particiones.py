from Hold_out_estratificado import hold_out_estratificado

def generar_particiones_estratificadas(df, n_particiones = 100):

    '''Genera 100 particiones usando la old_out_estratificado'''

    particiones = []

    for i in range(n_particiones):
        X_Train, Y_Train, X_Test, Y_Test = hold_out_estratificado(df, proporcion = 0.8, target_col = 'Class', seed = i)
        particiones.append((X_Train, Y_Train, X_Test, Y_Test))
    
    return particiones                     