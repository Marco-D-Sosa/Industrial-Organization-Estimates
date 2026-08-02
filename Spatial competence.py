import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import folium
import webbrowser
import os


"""
Analisis de competencia espacial y definicion de mercados geograficos relevantes"
Base de datos: Starbucks Locations Worldwide (sacada de Kaggle)
"""


# 1. Carga de datos espaciales
df_espacial = pd.read_csv(r"C:\Users\HP\Documents\Org industrial\startbucks.csv")
df_espacial = df_espacial.dropna(subset=['latitude','longitude']).copy()
df_la = df_espacial[df_espacial['city']=='Los Angeles'].copy().reset_index(drop=True)
df_la['lat_rad'] = np.radians(df_la['latitude'])
df_la['lon_rad'] = np.radians(df_la['longitude'])

# 2. Calculo del HHI
radio_tierra_km = 6371.0
radio_mercado_km = 5.0
radio_rad = radio_mercado_km / radio_tierra_km
arbol = BallTree(df_la[['lat_rad','lon_rad']], metric='haversine')
indices_vecinos = arbol.query_radius(df_la[['lat_rad','lon_rad']], r=radio_rad)
df_la['competidores_5km'] = [len(vecinos) for vecinos in indices_vecinos]
df_la['hhi_local'] = 10000 / df_la['competidores_5km'] #se asume cuotas simetricas de mercado

# 3. Diagnostico
print("DIAGNOSTICO ESPACIAL: AREA DE CAPTACION (LOS ANGELES)")
print("="*60)
locales_aislados = df_la.sort_values('competidores_5km').head(5)
print("Top 5 locales mas aislados:")
print(locales_aislados[['storeNumber','competidores_5km','hhi_local']])
print("="*60)
locales_densos = df_la.sort_values('competidores_5km', ascending=False).head(5)
print("Top 5 locales con mayor superposicion:")
print(locales_densos[['storeNumber','competidores_5km','hhi_local']])
print("="*60)

# 4. Visualizacion espacial en mapa
centro_lat = df_la['latitude'].mean()
centro_lon = df_la['longitude'].mean()
mapa_completo = folium.Map(location=[centro_lat, centro_lon], zoom_start=11)
for index, fila in df_la.iterrows():
    lat = fila['latitude']
    lon = fila['longitude']
    competidores = fila['competidores_5km']
    hhi = fila['hhi_local']
    #Lógica de calor: Rojo=Monopolio espacial (< 5 competidores cerca), Azul oscuro=Alta canibalización (> 40 competidores cerca)
    if competidores <= 5:
        color_punto = 'red'
    elif competidores >= 40:
        color_punto = 'darkblue'
    else:
        color_punto = 'lightblue'
    # Dibujamos un punto para CADA local
    folium.CircleMarker(
        location=[lat, lon],
        radius=5, # Tamaño del puntito
        color=color_punto,
        fill=True,
        fill_opacity=0.8,
        popup=f"Local: {fila['storeNumber']}<br>Competidores (5km): {competidores}<br>HHI: {hhi:.0f}"
    ).add_to(mapa_completo)
nombre_archivo = "mapa_completo_la.html"
ruta_completa = os.path.abspath(nombre_archivo)
mapa_completo.save(ruta_completa)
webbrowser.open('file://' + ruta_completa)
