import pandas as pd
import numpy as np
import pyblp
from linearmodels.iv import IV2SLS



"""
Analisis sobre M&A (Elasticidades, GUPP, UPP y Diversion ratio)
Modelo Logit
Dataset: Cereales de Nevo (2000)
"""

# 1. Preparamos el dataset
ruta_nevo = pyblp.data.NEVO_PRODUCTS_LOCATION
df = pd.read_csv(ruta_nevo)
df['sum_shares'] = df.groupby('market_ids')['shares'].transform('sum')
df['s0'] = 1 - df['sum_shares']
df['y_logit'] = np.log(df['shares']) - np.log(df['s0'])

# 2. Estimacion de las elasticidades mediante 2SLS (IV)
formula = 'y_logit ~ 1 + sugar + mushy + [prices ~ demand_instruments0]'
modelo_iv = IV2SLS.from_formula(formula, data=df)
resultados = modelo_iv.fit(cov_type='robust')
print("Resultados de la estimacion de la demanda (Logit IV - Nevo)")
print("="*60)
print(resultados.summary)
print("="*60)
alfa_precio = resultados.params['prices']
precio_promedio = df['prices'].mean()
share_promedio = df['shares'].mean()
elasticidad_propia = alfa_precio * precio_promedio * (1 - share_promedio)
elasticidad_cruzada = - alfa_precio * precio_promedio * share_promedio
print(f"Coeficiente Alpha de Precio: {alfa_precio:.4f}")
print(f"Elasticidad-precio propia promedio: {elasticidad_propia:.2f}")
print(f"Elasticidad-precio cruzada promedio: {elasticidad_cruzada:.4f}")

# 3. Evaluacion de fusiones (Diversion ratio, UPP y GUPPI)
def evaluacion_fusion(firma1, p1, c1, q1, firma2, p2, c2, q2, elast_propia, elast_cruzada):
    margen_absoluto_2 = p2 - c2
    margen_porcentual_2 = margen_absoluto_2 / p2
    # 1. Diversion Ratio (D12)
    desvio_matematico = abs(elast_cruzada / elast_propia) * (q2 / q1)
    d12 = min(desvio_matematico, 0.40)    
    # 2. UPP (Presión al alza en dólares, asumiendo cero eficiencias de costos)
    upp = d12 * margen_absoluto_2
    # 3. GUPPI (Índice porcentual de presión de precios)
    guppi = d12 * margen_porcentual_2 * (p2 / p1)    
    return d12, upp, guppi

# Seleccionamos dos firmas del dataset
firma1 = 1
firma2 = 2
df_f1 = df[df['firm_ids'] == firma1]
precio1 = df_f1['prices'].mean()
# Se aproxima el costo marginal asumiendo un margen razonable del 30% (o estimando C = P - MC)
costo1 = precio1 * 0.70  
volumen1 = df_f1['shares'].sum()
df_f2 = df[df['firm_ids'] == firma2]
precio2 = df_f2['prices'].mean()
costo2 = precio2 * 0.70
volumen2 = df_f2['shares'].sum()
d12_A, upp_A, guppi_A = evaluacion_fusion(
    firma1, precio1, costo1, volumen1, 
    firma2, precio2, costo2, volumen2, 
    elasticidad_propia, elasticidad_cruzada
)

# 4. REPORTE DE LA FUSION
print("\n" + "="*50)
print(f"DICTAMEN DE FUSIÓN: FIRMA {firma1} + FIRMA {firma2}")
print("="*50)
print(f"Diversion Ratio (D12): {d12_A * 100:.1f}% de los clientes de 1 se irían a 2.")
print(f"UPP (Presión Absoluta): ${upp_A:.4f} por porción.")
print(f"GUPPI (Índice Bruto): {guppi_A * 100:.2f}%")
if guppi_A > 0.05:
    print("\nVEREDICTO: ALERTA ROJA. El GUPPI supera el umbral del 5%.")
else:
    print("\nVEREDICTO: Fusión viable. No hay presión anticompetitiva significativa.")
