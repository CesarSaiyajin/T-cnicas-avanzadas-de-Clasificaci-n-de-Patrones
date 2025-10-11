from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def entrenar_modelo(algoritmo, X_train, Y_train):
    '''
    Entrena el modelo con el nombre del algoritmo seleccionado
    '''
    # '1NN', '3NN', '5NN', 'SVM', 'Naive Bayes', 'RF'
    # hace falta el C5.0 y el XGBOOST ------------------------------------ OJO

    if algoritmo == '1NN':
        modelo = KNeighborsClassifier(n_neighbors=1)

    elif algoritmo == '3NN':
        modelo = KNeighborsClassifier(n_neighbors=3)

    elif algoritmo == '5NN':
        modelo = KNeighborsClassifier(n_neighbors=5)

    elif algoritmo == 'SVM':
        modelo = SVC(gamma='auto')

    elif algoritmo == 'Naive Bayes':
        modelo = GaussianNB()
    
    elif algoritmo == 'RF':
        modelo = RandomForestClassifier(max_depth=2, random_state=0)
        
    elif algoritmo  == 'C5.0':   
        modelo = DecisionTreeClassifier(criterion="entropy")
    
    elif algoritmo  == 'XGBOOST':
        modelo = xgb.XGBClassifier(
        objective="multi:softmax",  # para clasificación multiclase
        num_class=2,                # número de clases
        n_estimators=100,           # árboles (boosting rounds)
        learning_rate=0.1,          # tasa de aprendizaje
        max_depth=4,                # profundidad máxima de cada árbol
        random_state=42)
       
    else:
        raise ValueError(f'{algoritmo} no disponible')
    
    return modelo.fit(X_train, Y_train)

def predecir_modelo(modelo, X_test):
    '''
    Genera la prediccion del modelo que fue seleccionado.
    '''
    return modelo.predict(X_test)