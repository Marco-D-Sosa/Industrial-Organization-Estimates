import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS


"""
Analisis de concentracion (HHI), estimacion de elasticidades (demanda precio y cruzadas) 
y analisis de fusiones (Diversion ratio, UPP y GUPPI).
Base de datos: Iowa Retail Sales (sacada de la API del estado de IOWA)
"""



# 1. Cargo la base y limpio"
url = "https://data.iowa.gov/resource/m3tr-qhgy.csv?$limit=50000"
df_raw = pd.read_csv(url)
df = df_raw.copy()
print(df)
print("="*60)
columnas = ['county','category_name','vendor_name','sale_dollars','sale_liters','state_bottle_retail','state_bottle_cost']
df = df[columnas].dropna()
df = df[(df['sale_liters']>0) & (df['state_bottle_retail']>0) & (df['state_bottle_cost']>0) & df['sale_dollars']>0]
print(df)
print("="*60)

# 2. Calculo del HHI"
panel = df.groupby(['county','category_name','vendor_name'])['sale_dollars'].sum().reset_index()
panel['ingreso_mercado'] = panel.groupby(['county','category_name'])['sale_dollars'].transform('sum')
panel['market_share'] = panel['sale_dollars'] / panel['ingreso_mercado']
panel['share_sq'] = (panel['market_share']*100) ** 2
hhi_df = panel.groupby(['county','category_name']).agg(HHI=('share_sq','sum'),numero_firmas=('vendor_name','nunique'),
                                                       ingreso_total=('ingreso_mercado','first')).reset_index()
print("\nVista del Panel (Firmas y sus cuotas de mercado):")
print(panel[['county','category_name','vendor_name','market_share']].sample(5))
print("="*60)
print("\nVista del HHI:")
print(hhi_df[['county','category_name','HHI']])
hhi_df_filtrado = hhi_df[hhi_df['ingreso_total']>5000].copy() #Descarto mercados "fantasma"
mercados_concentrados = hhi_df_filtrado.sort_values(by='HHI',ascending=False)
mercados_competitivos = hhi_df_filtrado.sort_values(by='HHI',ascending=True)
print("\n Top 5 mercados mas concentrados (HHI>2500):")
print(mercados_concentrados[['county','category_name','HHI','numero_firmas']].head())
print("="*60)
print("\n Top 5 mercados mas competitivos (HHI<1500):")
print(mercados_competitivos[['county','category_name','HHI','numero_firmas']].head())

# 3. Variables del modelo econometrico"
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

# 4. Estimacion de las elasticidades"
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

# 5. Evaluacion de fusiones (Diversion ratio, UPP y GUPPI) usando elasticidad cruzada promedio"
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
# REPORTE DE DEFENSA DE LA COMPETENCIA
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
