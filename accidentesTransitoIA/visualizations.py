import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Transformer
import folium
from folium.plugins import MarkerCluster

def plot_severity_histogram(df):

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Casualty_Severity')
    plt.title('Distribución de Severidad de Accidentes')
    plt.savefig('static/severity_hist.png')
    plt.close()

def create_accident_map(df):
    df = df.dropna(subset=['Grid_Ref_Easting', 'Grid_Ref_Northing'])

    transformer = Transformer.from_crs("epsg:27700", "epsg:4326")  # British National Grid → WGS84

    # Convertir coordenadas
    df['Latitude'], df['Longitude'] = transformer.transform(df['Grid_Ref_Easting'].values, df['Grid_Ref_Northing'].values)

    MAX_PUNTOS = 1500  # Ajustar este número según el rendimiento que desee para graficar los puntos de accidentes de Tránsito
    if len(df) > MAX_PUNTOS:
        df = df.sample(MAX_PUNTOS)
   
    map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    accident_map = folium.Map(location=map_center, zoom_start=6)
    marker_cluster = MarkerCluster().add_to(accident_map)

    for _, row in df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f'Severidad: {row["Casualty_Severity"]}'
        ).add_to(marker_cluster)

    accident_map.save('templates/mapa_accidentes.html')