import pandas as pd
import mysql.connector

# Función para cargar datos desde un archivo CSV
# file_path: ruta del archivo CSV
def load_csv_data(file_path):
    # Lee el archivo CSV usando punto y coma (;) como delimitador de columnas
    return pd.read_csv(file_path, delimiter=';')

# Función para cargar datos desde una base de datos MySQL
def load_mysql_data():
    # Establece la conexión a la base de datos MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Asegúrate de usar una contraseña segura en entornos reales
        database="proyectoia"
    )
    # Define la consulta SQL para obtener todos los datos de la tabla accidentestransito
    query = "SELECT * FROM accidentestransito"
    # Ejecuta la consulta y carga los resultados en un DataFrame
    df = pd.read_sql(query, conn)
    # Cierra la conexión a la base de datos
    conn.close()
    # Devuelve el DataFrame resultante
    return df

# Función para combinar los datos del CSV y la base de datos
def merge_data(csv_path):
    # Carga los datos del archivo CSV
    df_csv = load_csv_data(csv_path)
    # Carga los datos desde la base de datos
    df_db = load_mysql_data()
    # Une ambos DataFrames en uno solo (opcional: podría incluir lógica de limpieza o eliminación de duplicados)
    df_total = pd.concat([df_csv, df_db], ignore_index=True)
    # Devuelve el DataFrame combinado (aunque actualmente solo devuelve el CSV)
    return df_total