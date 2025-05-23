# Limpieza y transformación
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def clean_data(df):
     # Eliminar columnas irrelevantes para la predicción
    df = df.drop(columns=["Reference_Number", "Accident_Date", "Time_24hr"], errors='ignore')
     # Eliminar filas donde la columna 'Casualty_Severity' esté vacía
    df = df.dropna(subset=['Casualty_Severity'])
    df = df[df['Casualty_Severity'] != 'NA']
    # Codificar variables categóricas
    categorical_cols = [
        "1st_Road_Class", "Road_Surface", "Lighting_Conditions", "Weather_Conditions",
        "Casualty_Class", "Sex_of_Casualty", "Type_of_Vehicle", 
        "Did_Police_Officer_Attend_Scene_of_Accident"
    ]
    
    encoders = {}
    
   # Recorre todas las columnas categóricas especificadas
    for col in categorical_cols:
        # Verifica si la columna existe en el DataFrame
        if col in df.columns:
            # Convierte los valores de la columna a tipo string y reemplaza valores faltantes (NaN) con "Unknown"
            df[col] = df[col].astype(str).fillna("Unknown")
            # Crea un codificador de etiquetas (LabelEncoder)
            le = LabelEncoder()
            # Asegura que "Unknown" esté registrado
            le.fit(list(df[col].unique()) + ["Unknown"]) 
            # Aplica el codificador a la columna, convirtiendo cada categoría en un número entero
            df[col] = le.fit_transform(df[col])
            encoders[col] = le  # Guardar el encode
        else:
            le = encoders.get(col)
            if le:
                df[col] = df[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
  
    joblib.dump(encoders, "label_encoders.pkl")
   # Llenar valores numéricos faltantes con la media  
    if "Age_of_Casualty" in df.columns:
        df["Age_of_Casualty"] = pd.to_numeric(df["Age_of_Casualty"], errors='coerce')
        df["Age_of_Casualty"] = df["Age_of_Casualty"].fillna(df["Age_of_Casualty"].mean())
    
    # Eliminar filas con valores faltantes restantes
    df = df.dropna()
    return df
 
# Función para dividir los datos en conjuntos de entrenamiento y prueba
# df: DataFrame de entrada con los datos ya preprocesados
# target: nombre de la columna objetivo que se desea predecir
def split_data(df, target="Casualty_Severity"):
    # Separa las variables independientes (X) eliminando la columna objetivo del DataFrame
    X = df.drop(columns=[target])
    # Selecciona la columna objetivo (y), es decir, la variable que se quiere predecir
    y = df[target]
    # Divide los datos en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    # random_state asegura que los resultados sean reproducibles
    return train_test_split(X, y, test_size=0.2, random_state=42)