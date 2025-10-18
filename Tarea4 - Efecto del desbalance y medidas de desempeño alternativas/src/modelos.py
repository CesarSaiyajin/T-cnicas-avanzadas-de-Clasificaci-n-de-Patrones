'''
PARTE 4. Aplica los siguientes clasificadores: 
*1NN, 
*3NN, 
*5NN, 
*Naive Bayes, 
*J48,
*Regresión Logística, 
*Random Forest, 
*SVM, 
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def entrenar_modelo(nombre, X_train, Y_train):
    '''
    Elegir que modelo se usara
    '''
    if nombre == '1NN':
        modelo = KNeighborsClassifier(n_neighbors = 1)
    elif nombre == '3NN':
        modelo = KNeighborsClassifier(n_neighbors = 3)
    elif nombre == '5NN':       
        modelo = KNeighborsClassifier(n_neighbors = 5)
    elif nombre == 'Naive Bayes':
        modelo = GaussianNB()
    elif nombre == 'J48':   
        modelo = DecisionTreeClassifier(random_state = 42)
    elif nombre == 'Regresión Logística':
        modelo = LogisticRegression(max_iter = 1000, random_state = 42)
    elif nombre == 'Random Forest':
        modelo = RandomForestClassifier(n_estimators = 100, random_state = 42)
    elif nombre == 'SVM':
        modelo = SVC(random_state = 42)
    else:
        raise ValueError(f'{nombre} no válido.')
    

    
    return modelo.fit(X_train, Y_train)


def predecir_modelo(modelo, X_test):
    '''
    devuelve las predicciones del modelo
    '''
    return modelo.predict(X_test)
