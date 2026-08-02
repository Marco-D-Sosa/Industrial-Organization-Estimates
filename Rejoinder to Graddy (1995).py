import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
import statsmodels.api as sm

fish = pd.read_stata(r"C:\Users\HP\Downloads\Estimaciones de IO (NEIO)\Bases de datos\fish_stata.dta")
print(fish.describe())
print("------------------------------------")
print("------------------------------------")

"""
Dato: Esta base sale de Graddy(1995), 
la autora tomo estos datos de las transacciones de un puesto de pescados para testear la existencia de discriminacion de precios de 3°grado. 
Como resultado pudo demostrar dicha discriminacion.
"""



# Notamos que se pesca de lunes a viernes (cerrado sábados y domingos), hay variables dummies que definen los días.
fish['frid'] = 0
condicion = (fish['mon']==0) & (fish['tues']==0) & (fish['wed']==0) & (fish['thurs']==0) 
fish.loc[condicion, 'frid'] = 1 
fish['dia'] = fish['mon'] + 2*fish['tues'] + 3*fish['wed'] + 4*fish['thurs'] + 5*fish['frid']
fish.sort_values(by='dia', inplace=True)

# Calcula para cada dia estadisticas del precio promedio ponderado (avgprc) y de la cantidad total (totqty)
print(fish.groupby('dia')[['avgprc','totqty']].describe())
print("------------------------------------")
print("------------------------------------")


# Calcula para cada dia estadisticas del precio a los asiaticos (prca), a los blancos (prcw) y las cantidades totales a los asiaticos (qtya) y a los blancos (qtyw)
print(fish.groupby('dia')[['prca','prcw','qtya','qtyw']].describe())
print("------------------------------------")
print("------------------------------------")


# Definimos la variable "t" como la variable temporal
fish.set_index('t', inplace=True)

# Mostramos la base de datos (problemas de endogeneidad)
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
fish.plot.scatter(ax=ax1, x='ltotqty', y='lavgprc')
ax1.set_title('')
ax1.set_ylabel('Ln(Precio promedio ponderado)')
ax1.set_xlabel('Ln(Cantidad)')
plt.show

reg = smf.ols('ltotqty ~ lavgprc', data=fish).fit()
print(reg.summary())
print("------------------------------------")
print("------------------------------------")



"""
Instrumentar el precio con la variables "olas" (wave) y "viento" (speed):
El costo de obtener pescado aumenta con las olas y el viento, pero eso no debería afectar la demanda de pescado en restaurantes, hogares, etc.
¿Por qué afecta el oleaje y el viento? 
(1) costo de embarcar, seguridad de embarcar, consumo de combustible, 
(2) los peces se pueden dispersar e ir a las profundidades, reduciendo la cantidad de pesca.
vce(hac nwest opt) corrige por la posible autocorrelacion con el pasado y por heterocedasticidad
"""
fish = sm.add_constant(fish)
mod = IV2SLS(fish['ltotqty'], fish[['mon','tues','wed','thurs','const']], fish[['lavgprc']], fish[['wave2','wave3','speed2','speed3']])
reg_2 = mod.fit(cov_type='kernel', kernel='bartlett')
print(reg_2)
print("------------------------------------")
print("------------------------------------")

# Ahora estimamos las demandas por separado (asiaticos y blancos):
    
# Asiaticos
fish['lprca'] = np.log(fish['prca'])
fish['lqtya'] = np.log(fish['qtya'])
mod_a = IV2SLS(fish['lqtya'], fish[['mon','tues','wed','thurs','const']], fish[['lprca']], fish[['wave2','wave3','speed2','speed3']])
reg_a = mod_a.fit(cov_type='kernel', kernel='bartlett')
print(reg_a)
print("------------------------------------")
print("------------------------------------")

# Blancos
fish['lprcw'] = np.log(fish['prcw'])
fish['lqtyw'] = np.log(fish['qtyw'])
mod_w = IV2SLS(fish['lqtyw'], fish[['mon','tues','wed','thurs','const']], fish[['lprcw']], fish[['wave2','wave3','speed2','speed3']])
reg_w = mod_w.fit(cov_type='kernel', kernel='bartlett')
print(reg_w)
print("------------------------------------")
print("------------------------------------")

