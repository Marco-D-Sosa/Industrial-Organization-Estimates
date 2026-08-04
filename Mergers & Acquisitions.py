import pandas as pd
import numpy as np



# 1. Variables del modelo econometrico
panel_reg = df.groupby(['county','category_name','vendor_name']).agg(
    Q_liters=('sale_liters','sum'),
    P_retail=('state_bottle_retail','mean'),
    Cost_wholesale=('state_bottle_cost','mean')).reset_index()
panel_reg['ln_Q'] = np.log(panel_reg['Q_liters'])
panel_reg['ln_P'] = np.log(panel_reg['P_retail'])
panel_reg['ln_Cost'] = np.log(panel_reg['Cost_wholesale'])
sum_precios_mercado = panel_reg.groupby(['county','category_name'])['P_retail'].transform('sum')
n_firmas_mercado = panel_reg.groupby(['county','category_name'])['P_retail'].transform('count')
panel_reg['P_competidores'] = np.where(
    n_firmas_mercado > 1, (sum_precios_mercado - panel_reg['P_retail']) / (n_firmas_mercado -1), np.nan)
panel_modelo = panel_reg.dropna(subset=['P_competidores']).copy()
panel_modelo['ln_P_comp'] = np.log(panel_modelo['P_competidores'])

# 2. Estimacion de las elasticidades
formula = 'ln_Q ~ 1 + ln_P_comp + C(category_name) + [ln_P ~ ln_Cost]'
modelo_iv = IV2SLS.from_formula(formula, data=panel_modelo)
resultados = modelo_iv.fit(cov_type='robust')
print("\n" + "="*60)
print("Resultados de la estimacion de la demanda")
print("="*60)
print(resultados.summary)
print("="*60)
elasticidad_propia = resultados.params['ln_P']
elasticidad_cruzada = resultados.params['ln_P_comp']
print(f"Elasticidad-precio de la demanda:{elasticidad_propia:.2f}")
print(f"Elasticidad-precio cruzada:{elasticidad_cruzada:.2f}")

# 3. Evaluacion de fusiones (Diversion ratio, UPP y GUPPI) usando elasticidad cruzada promedio
def evaluacion_fusion(firma1, p1, c1, q1, firma2, p2, c2, q2, elast_propia, elast_cruzada):
    margen_absoluto_2 = p2 - c2
    margen_porcentual_2 = margen_absoluto_2 / p2
    # 1. Diversion Ratio (D12)
    desvio_matematico = abs(elast_cruzada / elast_propia) * (q2 / q1)
    d12 = min(desvio_matematico, 0.40) # Tope lógico para una fusión de 2 firmas en un mercado amplio
    # 2. UPP (Presión al alza en dólares, asumiendo cero eficiencias de costos)
    upp = d12 * margen_absoluto_2
    # 3. GUPPI (Índice porcentual de presión de precios)
    guppi = d12 * margen_porcentual_2 * (p2 / p1)    
    return d12, upp, guppi

firma1 = "JIM BEAM BRANDS"
firma2 = "LAIRD & COMPANY"
df_f1 = df[df['vendor_name'] == firma1]
precio1 = df_f1['state_bottle_retail'].mean()
costo1 = df_f1['state_bottle_cost'].mean()
volumen1 = df_f1['sale_liters'].sum()
df_f2 = df[df['vendor_name'] == firma2]
precio2 = df_f2['state_bottle_retail'].mean()
costo2 = df_f2['state_bottle_cost'].mean()
volumen2 = df_f2['sale_liters'].sum()
d12_A, upp_A, guppi_A = evaluacion_fusion(firma1, precio1, costo1, volumen1, firma2, precio2, costo2, volumen2, elasticidad_propia, elasticidad_cruzada)

# 4. REPORTE DE DEFENSA DE LA COMPETENCIA
print("\n" + "="*50)
print("DICTAMEN DE FUSIÓN: FIRMA A + FIRMA B")
print("="*50)
print(f"Diversion Ratio (D12): {d12_A * 100:.1f}% de los clientes de A se irían a B.")
print(f"UPP (Presión Absoluta): ${upp_A:.2f} de incentivo al alza por botella.")
print(f"GUPPI (Índice Bruto): {guppi_A * 100:.2f}%")
if guppi_A > 0.05:
    print("\n VEREDICTO: ALERTA ROJA. El GUPPI supera el umbral del 5%.")
else:
    print("\n VEREDICTO: Fusión viable. No hay presión anticompetitiva significativa.")
"""