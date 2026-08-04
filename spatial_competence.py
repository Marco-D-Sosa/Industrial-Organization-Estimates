import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import folium
import webbrowser
import os
import kagglehub



"""
Analysis of spatial competition and definition of relevant geographic markets.
Dataset: Starbucks Locations Worldwide (sourced from Kaggle).

Note: Ensure you have your Kaggle API Token configured in your environment
or local directory so kagglehub can authenticate automatically.
"""

# 1. Spatial data loading
path = kagglehub.dataset_download("kukuroo3/starbucks-locations-worldwide-2021-version")
df_spatial = pd.read_csv(f"{path}/startbucks.csv")
df_spatial = df_spatial.dropna(subset=['latitude','longitude']).copy()
df_la = df_spatial[df_spatial['city']=='Los Angeles'].copy().reset_index(drop=True)
df_la['lat_rad'] = np.radians(df_la['latitude'])
df_la['lon_rad'] = np.radians(df_la['longitude'])

# 2. Calculation of the HHI
earth_radius_km =  6371.0
market_radius_km = 5.0
radius_rad = market_radius_km / earth_radius_km
tree = BallTree(df_la[['lat_rad','lon_rad']], metric='haversine')
index_neighbors = tree.query_radius(df_la[['lat_rad','lon_rad']], r=radius_rad)
df_la['competitors_5km'] = [len(neighbors) for neighbors in index_neighbors]
df_la['hhi_local'] = 10000 / df_la['competitors_5km'] # Symmetric market shares are assumed.

# 3. Diagnosis
print("Spatial analysis: catchment area (Los Angeles)")
print("="*60)
isolated_stores = df_la.sort_values('competitors_5km').head(5)
print("Top 5 most isolated stores:")
print(isolated_stores[['storeNumber','competitors_5km','hhi_local']])
print("="*60)
dense_stores = df_la.sort_values('competitors_5km', ascending=False).head(5)
print("Top 5 stores with the highest overlap:")
print(dense_stores[['storeNumber','competitors_5km','hhi_local']])
print("="*60)

# 4. Spatial visualization on a map
lat_center = df_la['latitude'].mean()
lon_center = df_la['longitude'].mean()
complete_map = folium.Map(location=[lat_center, lon_center], zoom_start=11)
for index, row in df_la.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    competitors = row['competitors_5km']
    hhi = row['hhi_local']
    #Heat logic: Red = Space monopoly (< 5 competitors nearby), Dark Blue = High cannibalization (> 40 competitors nearby)
    if competitors <= 5:
        color_dot = 'red'
    elif competitors >= 40:
        color_dot = 'darkblue'
    else:
        color_dot = 'lightblue'
    # Plotting a point for EACH store
    folium.CircleMarker(
        location=[lat, lon],
        radius=5, # Dot size
        color=color_dot,
        fill=True,
        fill_opacity=0.8,
        popup=f"Store: {row['storeNumber']}<br>Competitors (5km): {competitors}<br>HHI: {hhi:.0f}"
    ).add_to(complete_map)
file_name = "complete_map_la.html"
full_path = os.path.abspath(file_name)
complete_map.save(full_path)
webbrowser.open('file://' + full_path)
