import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS

data = pd.read_stata(r".\Base de data\data_demanda.dta") # Insert the corresponding directory
data.columns = data.columns.str.upper()
print(data.describe())
print("-"*60)



"""
This database is purely fictional, generated to provide an initial overview.
P = price
Q = quantiti
Y = Income
B = instrumental var for offer curve
"""



"""
1) estimate the functions:
Q = a1 + a2P + a3Y + a4(P*Y) (linear demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['PY'] = data['P']*data['Y']
data['QB'] = data['Q']*data['B']
data['BY'] = data['B']*data['Y']
mod_d1 = IV2SLS(data['Q'], data['Y'], data[['P','PY']], data[['B','BY']]) #Demand estimation
res_d1 = mod_d1.fit()
print(res_d1)
print("----------------------------------")
print(res_d1.first_stage) #Instrument check
print("----------------------------------")
a2 = res_d1.params['P']
a4 = res_d1.params['PY']
data['slope'] = a2 + (a4 * data['Y'])
data['z'] = data['Q'] / data['slope']
data['z_ins'] = data['Y'] / data['slope']
mod_o1 = IV2SLS(data['P'], data['B'], data[['Q','QB','z']], data[['Y','BY','z_ins']]) #Estimation of marginal cost and behavior
res_o1 = mod_o1.fit()
print(res_o1)
print("----------------------------------")
print(res_o1.first_stage) #Instrument check
print("----------------------------------")
data = data.drop(columns=['slope','z','z_ins'])



"""
2) estimate the functions:
ln(Q) = ln(a1) + a2ln(P) + a3ln(Y) + a4(ln(P)*ln(Y)) (isoelastic demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['lnP'] = np.log(data['P'])
data['lnQ'] = np.log(data['Q'])
data['lnY'] = np.log(data['Y'])
data['lnB'] = np.log(data['B'])
data['lnP_lnY'] = np.log(data['P']) * np.log(data['Y'])
data['lnB_lnY'] = np.log(data['B']) * np.log(data['Y'])
mod_d2 = IV2SLS(data['lnQ'], data['lnY'], data[['lnP','lnP_lnY']], data[['lnB','lnB_lnY']])
res_d2 = mod_d2.fit()
print(res_d2)
print("----------------------------------")
print(res_d2.first_stage)
print("----------------------------------")
a2 = res_d2.params['lnP']
a4 = res_d2.params['lnP_lnY']
data['elast'] = a2 + (a4 * data['lnY'])
data['z'] = data['P'] / data['elast']
data['z_ins'] = data['elast']
mod_o2 = IV2SLS(data['P'], data['B'], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o2 = mod_o2.fit()
print(res_o2)
print("----------------------------------")
print(res_o2.first_stage)
print("----------------------------------")
data = data.drop(columns=['elast','z','z_ins'])



"""
3) estimate the functions:
P = a1 + a2Q + a3Y + a4(Q*Y) (linear inverse demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['QY'] = data['Q'] * data['Y']
mod_d3 = IV2SLS(data['P'], data['Y'], data[['Q','QY']], data[['B','BY']])
res_d3 = mod_d3.fit()
print(res_d3)
print("----------------------------------")
print(res_d3.first_stage)
print("----------------------------------")
a2 = res_d3.params['Q']
a4 = res_d3.params['QY']
data['slope'] = a2 + (a4 * data['Y'])
data ['z'] = data['Q'] * data['slope']
data['z_ins'] = data['Y'] * data['slope']
mod_o3 = IV2SLS(data['P'], data['B'], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o3 = mod_o3.fit()
print(res_o3)
print("----------------------------------")
print(res_o3.first_stage)
print("----------------------------------")
data = data.drop(columns=['slope','z','z_ins'])



"""
4) estimate the functions:
ln(P) = ln(a1) + a2ln(Q) + a3ln(Y) + a4(ln(Q)*ln(Y)) (isoelastic inverse demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['lnQ_lnY'] = np.log(data['Q']) * np.log(data['Y'])
mod_d4 = IV2SLS(data['lnP'], data['lnY'], data[['lnQ','lnQ_lnY']], data[['lnB','lnB_lnY']])
res_d4 = mod_d4.fit()
print(res_d4)
print("----------------------------------")
print(res_d4.first_stage)
print("----------------------------------")
a2 = res_d4.params['lnQ']
a4 = res_d4.params['lnQ_lnY']
data['elast'] = a2 + (a4 * data['lnY'])
data['z'] = data['P'] * data['elast']
data['z_ins'] = data['elast']
mod_o4 = IV2SLS(data['P'], data['B'], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o4 = mod_o4.fit()
print(res_o4)
print("----------------------------------")
print(res_o4.first_stage)
print("----------------------------------")
data = data.drop(columns=['elast','z','z_ins'])
