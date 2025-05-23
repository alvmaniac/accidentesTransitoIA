from flask import Flask, render_template , request
from data_loader import merge_data
from preprocess import clean_data, split_data
from model import train_model, evaluate_model
from visualizations import plot_severity_histogram, create_accident_map
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

@app.route('/')
def dashboard():
    df = merge_data('data/accidentes.csv')
    df = clean_data(df)
    
    # Visualizaciones
    plot_severity_histogram(df)
    create_accident_map(df)
    
    # Modelo
    X_train, X_test, y_train, y_test = split_data(df, target='Casualty_Severity')
    model = train_model(X_train, y_train)
    report, matrix = evaluate_model(model, X_test, y_test)
    
           
     # Guardar el modelo
    joblib.dump(model, "modelo_accidentes.pkl")

    return render_template('dashboard.html', report=report, matrix=matrix)

@app.route('/mapa')
def mapa():
    return render_template('mapa_accidentes.html')

@app.route("/formulario")
def formulario():
    return render_template("formulario.html")

    


# Columnas categóricas que fueron codificadas
categorical_cols = [
    "1st_Road_Class", "Road_Surface", "Lighting_Conditions", "Weather_Conditions",
    "Casualty_Class", "Sex_of_Casualty", "Type_of_Vehicle",
    "Did_Police_Officer_Attend_Scene_of_Accident"
]

@app.route("/predecir", methods=["POST"])
def predecir():
    modelo =  joblib.load("modelo_accidentes.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    if request.method == "POST":
        # 1. Recibir los datos del formulario
        datos = {
            "Grid_Ref_Easting": 0,
            "Grid_Ref_Northing": 0,
            "Number_of_Vehicles": int(request.form["Number_of_Vehicles"]),
            "1st_Road_Class": "Unknown",
            "Road_Surface": "Unknown",
            "Lighting_Conditions": "Unknown",
            "Weather_Conditions": request.form["Weather_Conditions"],
            "Casualty_Class": "Unknown",
            "Sex_of_Casualty": request.form["Sex_of_Casualty"],
            "Age_of_Casualty": float(request.form["Age_of_Casualty"]),
            "Type_of_Vehicle": "Unknown",
            "Did_Police_Officer_Attend_Scene_of_Accident": "Unknown"
        }

        # 2. Convertir a DataFrame
        df_input = pd.DataFrame([datos])

        # 3. Preprocesar igual que en entrenamiento
        for col in categorical_cols:
            df_input[col] = df_input[col].astype(str)
            if col in label_encoders:
                le = label_encoders[col]
                df_input[col] = df_input[col].apply(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

        # 4. Predecir
        prediccion = modelo.predict(df_input)[0]

        return render_template("resultado.html", resultado=prediccion)

    return render_template("formulario.html")



if __name__ == '__main__':
    app.run(debug=True)