from flask import Flask, render_template
from data_loader import merge_data
from preprocess import clean_data, split_data
from model import train_model, evaluate_model
from visualizations import plot_severity_histogram, create_accident_map

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

    return render_template('dashboard.html', report=report, matrix=matrix)

@app.route('/mapa')
def mapa():
    return render_template('mapa_accidentes.html')

if __name__ == '__main__':
    app.run(debug=True)