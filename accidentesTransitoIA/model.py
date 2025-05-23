# Como “Casualty Severity” puede tener múltiples niveles (leve, grave, fatal), usaremos un clasificador multiclase como RandomForest:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter

def train_model(X_train, y_train):
    # Balanceo la data con SMOTE
    print(Counter(y_train))
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    # Crear un modelo de bosque aleatorio con 100 árboles y una semilla fija para reproducibilidad
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Entrenar el modelo usando los datos de entrenamiento (características y etiquetas)
    model.fit(X_train_bal, y_train_bal)
    # Devolver el modelo ya entrenado
    return model


def evaluate_model(model, X_test, y_test):
    # Usar el modelo entrenado para predecir las etiquetas del conjunto de prueba
    y_pred = model.predict(X_test)
    # Generar un reporte de métricas de clasificación (precisión, recall, f1-score) como un diccionario
    report = classification_report(y_test, y_pred, output_dict=True)
    # Calcular la matriz de confusión para evaluar las predicciones del modelo
    matrix = confusion_matrix(y_test, y_pred)
    # Devolver el reporte y la matriz de confusión
    return report, matrix