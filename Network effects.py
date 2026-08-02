import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

"""
La base de datos fue descargada de Kaggle, llamada Video games Sale.
Al no contar con informacion sobre ventas de consolas, se utiliza las ventas de juegos por plataformas como proxy 
(Mientras mas consolas se venden, mas juegos se compran con dicha consola)
Problema con esta proxy: 
las ventas de juegos pueden ser mayores por ese mayor repertorio, sin que se traduzca en ventas de consolas.
El efecto de red podria sobreestimarse
"""


"1. Preparo la base de datos"
df_raw = pd.read_csv('vgsales.csv')
df_raw['Year'] = pd.to_numeric(df_raw['Year'],errors='coerce')
df = df_raw.dropna(subset=['Year']).copy()
df['Year'] = df['Year'].astype(int)
panel = df.groupby(['Platform','Year']).agg(ventas_globales=('Global_Sales', 'sum'), nuevos_juegos=('Name', 'count')).reset_index()
panel = panel.sort_values(by=['Platform','Year'])
panel['catalogo_acumulado'] = panel.groupby('Platform')['nuevos_juegos'].cumsum()
panel = panel[ (panel['ventas_globales']>0) & (panel['catalogo_acumulado']>0) ].copy()
panel['ln_ventas'] = np.log(panel['ventas_globales'])
panel['ln_red'] = np.log(panel['catalogo_acumulado'])
print(panel[panel['Platform'] == 'PS2'] [['Year','nuevos_juegos','catalogo_acumulado','ventas_globales']].head())
print("="*60)


"2. Estimacion del efecto red"
#Modelo: log(ventas) = log(red) + FE por plataforma
modelo = smf.ols('ln_ventas ~ ln_red + C(Platform)', data=panel).fit(cov_type='HC1')
print("Resultados: elasticidad del efecto de red")
print("="*60)
print(modelo.summary())
print("="*60)
elasticidad_red = modelo.params['ln_red']
print(f"Elasticidad de red: {elasticidad_red:.3f}")
print("="*60)
