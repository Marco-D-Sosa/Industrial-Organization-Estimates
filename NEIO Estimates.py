import pandas as pd
from linearmodels.iv import IV2SLS
import numpy as np

datos = pd.read_stata(r"C:\Users\HP\Downloads\Estimaciones de IO (NEIO)\Bases de datos\datos_demanda.dta")
datos.columns = datos.columns.str.upper()
print(datos.describe())
print("----------------------------------")

"""
Esta base de datos es solo ficticia, generada para tener un primer acercamiento
P = price
Q = quantiti
Y = Income
B = instrumental var for offer curve
"""



"""
1) estimar las funciones:
Q = a1 + a2P + a3Y + a4(P*Y) (demanda lineal)
P = b1 + b2Q + b3B + a4(Q*B) (cmg lineal)
"""
datos['PY'] = datos['P']*datos['Y']
datos['QB'] = datos['Q']*datos['B']
datos['BY'] = datos['B']*datos['Y']
mod_d1 = IV2SLS(datos['Q'], datos['Y'], datos[['P','PY']], datos[['B','BY']]) #Estimacion de la demanda
res_d1 = mod_d1.fit()
print(res_d1)
print("----------------------------------")
print(res_d1.first_stage) #Comprobacion de instrumentos
print("----------------------------------")

a2 = res_d1.params['P']
a4 = res_d1.params['PY']
datos['pend'] = a2 + (a4 * datos['Y'])
datos['z'] = datos['Q'] / datos['pend']
datos['z_ins'] = datos['Y'] / datos['pend']
mod_o1 = IV2SLS(datos['P'], datos['B'], datos[['Q','QB','z']], datos[['Y','BY','z_ins']]) #Estimacion del cmg y conducta
res_o1 = mod_o1.fit()
print(res_o1)
print("----------------------------------")
print(res_o1.first_stage) #Comprobacion de instrumentos
print("----------------------------------")

datos = datos.drop(columns=['pend','z','z_ins'])



"""
2) estimar las funciones:
ln(Q) = ln(a1) + a2ln(P) + a3ln(Y) + a4(ln(P)*ln(Y)) (demanda isoelastica)
P = b1 + b2Q + b3B + a4(Q*B) (cmg lineal)
"""
datos['lnP'] = np.log(datos['P'])
datos['lnQ'] = np.log(datos['Q'])
datos['lnY'] = np.log(datos['Y'])
datos['lnB'] = np.log(datos['B'])
datos['lnP_lnY'] = np.log(datos['P']) * np.log(datos['Y'])
datos['lnB_lnY'] = np.log(datos['B']) * np.log(datos['Y'])
mod_d2 = IV2SLS(datos['lnQ'], datos['lnY'], datos[['lnP','lnP_lnY']], datos[['lnB','lnB_lnY']])
res_d2 = mod_d2.fit()
print(res_d2)
print("----------------------------------")
print(res_d2.first_stage)
print("----------------------------------")

a2 = res_d2.params['lnP']
a4 = res_d2.params['lnP_lnY']
datos['elast'] = a2 + (a4 * datos['lnY'])
datos['z'] = datos['P'] / datos['elast']
datos['z_ins'] = datos['elast']
mod_o2 = IV2SLS(datos['P'], datos['B'], datos[['Q','QB','z']], datos[['Y','BY','z_ins']])
res_o2 = mod_o2.fit()
print(res_o2)
print("----------------------------------")
print(res_o2.first_stage)
print("----------------------------------")

datos = datos.drop(columns=['elast','z','z_ins'])



"""
3) estimar las funciones:
P = a1 + a2Q + a3Y + a4(Q*Y) (demanda inversa lineal)
P = b1 + b2Q + b3B + a4(Q*B) (cmg lineal)
"""
datos['QY'] = datos['Q'] * datos['Y']
mod_d3 = IV2SLS(datos['P'], datos['Y'], datos[['Q','QY']], datos[['B','BY']])
res_d3 = mod_d3.fit()
print(res_d3)
print("----------------------------------")
print(res_d3.first_stage)
print("----------------------------------")

a2 = res_d3.params['Q']
a4 = res_d3.params['QY']
datos['pend'] = a2 + (a4 * datos['Y'])
datos ['z'] = datos['Q'] * datos['pend']
datos['z_ins'] = datos['Y'] * datos['pend']
mod_o3 = IV2SLS(datos['P'], datos['B'], datos[['Q','QB','z']], datos[['Y','BY','z_ins']])
res_o3 = mod_o3.fit()
print(res_o3)
print("----------------------------------")
print(res_o3.first_stage)
print("----------------------------------")

datos = datos.drop(columns=['pend','z','z_ins'])



"""
4) estimar las funciones:
ln(P) = ln(a1) + a2ln(Q) + a3ln(Y) + a4(ln(Q)*ln(Y)) (demanda inversa isoelastica)
P = b1 + b2Q + b3B + a4(Q*B) (cmg lineal)
"""
datos['lnQ_lnY'] = np.log(datos['Q']) * np.log(datos['Y'])
mod_d4 = IV2SLS(datos['lnP'], datos['lnY'], datos[['lnQ','lnQ_lnY']], datos[['lnB','lnB_lnY']])
res_d4 = mod_d4.fit()
print(res_d4)
print("----------------------------------")
print(res_d4.first_stage)
print("----------------------------------")

a2 = res_d4.params['lnQ']
a4 = res_d4.params['lnQ_lnY']
datos['elast'] = a2 + (a4 * datos['lnY'])
datos['z'] = datos['P'] * datos['elast']
datos['z_ins'] = datos['elast']
mod_o4 = IV2SLS(datos['P'], datos['B'], datos[['Q','QB','z']], datos[['Y','BY','z_ins']])
res_o4 = mod_o4.fit()
print(res_o4)
print("----------------------------------")
print(res_o4.first_stage)
print("----------------------------------")

datos = datos.drop(columns=['elast','z','z_ins'])

