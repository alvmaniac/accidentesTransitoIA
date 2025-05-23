from flask import Flask, render_template , request
from data_loader import merge_data
from preprocess import clean_data, split_data
from model import train_model, evaluate_model
from visualizations import plot_severity_histogram, create_accident_map
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

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



# Preprocesamiento de los datos ingresados por el usuario
def preprocess_form_data(df):
    for col in ["Weather_Conditions", "Sex_of_Casualty"]:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

@app.route("/formulario")
def formulario():
    return render_template("formulario.html")

@app.route("/predecir", methods=["POST"])
def predecir():
    
    # Carga del modelo entrenado
    modelo = joblib.load("modelo_accidentes.pkl")
    datos = {
        "Number_of_Vehicles": int(request.form["Number_of_Vehicles"]),
        "Weather_Conditions": request.form["Weather_Conditions"],
        "Age_of_Casualty": float(request.form["Age_of_Casualty"]),
        "Sex_of_Casualty": request.form["Sex_of_Casualty"]
    }

    df_form = pd.DataFrame([datos])
    df_form = preprocess_form_data(df_form)
    prediccion = modelo.predict(df_form)[0]

    return f"La severidad del accidente predicha es: <b>{prediccion}</b>"


if __name__ == '__main__':
    app.run(debug=True)