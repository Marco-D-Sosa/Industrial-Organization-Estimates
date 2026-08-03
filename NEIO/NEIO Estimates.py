import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS



"""
This database is purely fictional, generated to provide an initial overview.
P = price
Q = quantity
Y = Income
B = instrumental var for offer curve
"""
data = pd.read_stata(r"\NEIO\datos_demanda.dta") # Insert the corresponding directory
data.columns = data.columns.str.upper()
print(data.describe())
print("-"*60)
plt.figure(figsize=(10, 6))
plt.scatter(data['Q'], data['P'], alpha=0.35, color='gray', edgecolors='none', label='Data (Q,P)')
plt.title('NEIO estimation: linear-linear')
plt.xlabel('Quantity (Q)')
plt.ylabel('Price (P)')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



"""
1) estimate the functions:
Q = a1 + a2P + a3Y + a4(P*Y) (linear demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['PY'] = data['P']*data['Y']
data['QB'] = data['Q']*data['B']
data['BY'] = data['B']*data['Y']
data = sm.add_constant(data)
mod_d1 = IV2SLS(data['Q'], data[['const','Y']], data[['P','PY']], data[['B','BY']]) #Demand estimation
res_d1 = mod_d1.fit()
print(res_d1)
print("-"*60)
print(res_d1.first_stage) #Instrument check
print("-"*60)
a1 = res_d1.params['const']
a2 = res_d1.params['P']
a3 = res_d1.params['Y']
a4 = res_d1.params['PY']
data['slope'] = a2 + (a4 * data['Y'])
data['z'] = data['Q'] / data['slope']
data['z_ins'] = data['Y'] / data['slope']
mod_o1 = IV2SLS(data['P'], data[['const','B']], data[['Q','QB','z']], data[['Y','BY','z_ins']]) #Estimation of marginal cost and behavior
res_o1 = mod_o1.fit()
print(res_o1)
print("-"*60)
print(res_o1.first_stage) #Instrument check
print("-"*60)
b1 = res_o1.params['const']
b2 = res_o1.params['Q']
b3 = res_o1.params['B']
b4 = res_o1.params['QB']
data = data.drop(columns=['slope','z','z_ins'])

# Plotting the graph
Y_mean = data['Y'].mean()
B_mean = data['B'].mean()
q_grid = np.linspace(data['Q'].min(), data['Q'].max(), 300)
p_demand = (q_grid - a1 - (a3*Y_mean)) / (a2 + (a4*Y_mean))
p_mc = b1 + (b2*q_grid) + (b3*B_mean) + (b4*q_grid*B_mean)
plt.figure(figsize=(10, 6))
plt.scatter(data['Q'], data['P'], alpha=0.35, color='gray', edgecolors='none', label='Data (Q,P)')
plt.plot(q_grid, p_demand, color='navy', linewidth=2.5, label="Estimated demand")
plt.plot(q_grid, p_mc, color='firebrick', linewidth=2.5, linestyle='--', label='Estimated MC')
plt.title('NEIO estimation: linear-linear')
plt.xlabel('Quantity (Q)')
plt.ylabel('Price (P)')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



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
mod_d2 = IV2SLS(data['lnQ'], data[['const','lnY']], data[['lnP','lnP_lnY']], data[['lnB','lnB_lnY']])
res_d2 = mod_d2.fit()
print(res_d2)
print("----------------------------------")
print(res_d2.first_stage)
print("----------------------------------")
a1 = res_d2.params['const']
a2 = res_d2.params['lnP']
a3 = res_d2.params['lnY']
a4 = res_d2.params['lnP_lnY']
data['elast'] = a2 + (a4 * data['lnY'])
data['z'] = data['P'] / data['elast']
data['z_ins'] = data['elast']
mod_o2 = IV2SLS(data['P'], data[['const','B']], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o2 = mod_o2.fit()
print(res_o2)
print("----------------------------------")
print(res_o2.first_stage)
print("----------------------------------")
b1 = res_o2.params['const']
b2 = res_o2.params['Q']
b3 = res_o2.params['B']
b4 = res_o2.params['QB']
data = data.drop(columns=['elast','z','z_ins'])

# Plotting the graph
lnY_mean = data['lnY'].mean()
p_demand_log = np.exp((np.log(q_grid) - a1 - (a3*lnY_mean)) / (a2 + (a4*lnY_mean)))
p_mc = b1 + (b2*q_grid) + (b3*B_mean) + (b4*q_grid*B_mean)
plt.figure(figsize=(10, 6))
plt.scatter(data['Q'], data['P'], alpha=0.35, color='gray', edgecolors='none', label='Data (Q,P)')
plt.plot(q_grid, p_demand_log, color='navy', linewidth=2.5, label="Estimated demand")
plt.plot(q_grid, p_mc, color='firebrick', linewidth=2.5, linestyle='--', label='Estimated MC')
plt.title('NEIO estimation: isoelastic-linear')
plt.xlabel('Quantity (Q)')
plt.ylabel('Price (P)')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()



"""
3) estimate the functions:
P = a1 + a2Q + a3Y + a4(Q*Y) (linear inverse demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['QY'] = data['Q'] * data['Y']
mod_d3 = IV2SLS(data['P'], data[['const','Y']], data[['Q','QY']], data[['B','BY']])
res_d3 = mod_d3.fit()
print(res_d3)
print("----------------------------------")
print(res_d3.first_stage)
print("----------------------------------")
a1 = res_d3.params['const']
a2 = res_d3.params['Q']
a3 = res_d3.params['Y']
a4 = res_d3.params['QY']
data['slope'] = a2 + (a4 * data['Y'])
data ['z'] = data['Q'] * data['slope']
data['z_ins'] = data['Y'] * data['slope']
mod_o3 = IV2SLS(data['P'], data[['const','B']], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o3 = mod_o3.fit()
print(res_o3)
print("----------------------------------")
print(res_o3.first_stage)
print("----------------------------------")
b1 = res_o3.params['const']
b2 = res_o3.params['Q']
b3 = res_o3.params['B']
b4 = res_o3.params['QB']
data = data.drop(columns=['slope','z','z_ins'])

# Plotting the graph
p_inverse_demand = a1 + (a2*q_grid) + (a3*Y_mean) + (a4*q_grid*Y_mean)
p_mc = b1 + (b2*q_grid) + (b3*B_mean) + (b4*q_grid*B_mean)
plt.figure(figsize=(10, 6))
plt.scatter(data['Q'], data['P'], alpha=0.35, color='gray', edgecolors='none', label='Data (Q,P)')
plt.plot(q_grid, p_inverse_demand, color='navy', linewidth=2.5, label="Estimated demand")
plt.plot(q_grid, p_mc, color='firebrick', linewidth=2.5, linestyle='--', label='Estimated MC')
plt.title('NEIO estimation: inverse_linear-linear')
plt.xlabel('Quantity (Q)')
plt.ylabel('Price (P)')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


"""
4) estimate the functions:
ln(P) = ln(a1) + a2ln(Q) + a3ln(Y) + a4(ln(Q)*ln(Y)) (isoelastic inverse demand)
P = b1 + b2Q + b3B + a4(Q*B) (linear MC)
"""
data['lnQ_lnY'] = np.log(data['Q']) * np.log(data['Y'])
mod_d4 = IV2SLS(data['lnP'], data[['const','lnY']], data[['lnQ','lnQ_lnY']], data[['lnB','lnB_lnY']])
res_d4 = mod_d4.fit()
print(res_d4)
print("----------------------------------")
print(res_d4.first_stage)
print("----------------------------------")
a1 = res_d4.params['const']
a2 = res_d4.params['lnQ']
a3 = res_d4.params['lnY']
a4 = res_d4.params['lnQ_lnY']
data['elast'] = a2 + (a4 * data['lnY'])
data['z'] = data['P'] * data['elast']
data['z_ins'] = data['elast']
mod_o4 = IV2SLS(data['P'], data[['const','B']], data[['Q','QB','z']], data[['Y','BY','z_ins']])
res_o4 = mod_o4.fit()
print(res_o4)
print("----------------------------------")
print(res_o4.first_stage)
print("----------------------------------")
b1 = res_o4.params['const']
b2 = res_o4.params['Q']
b3 = res_o4.params['B']
b4 = res_o4.params['QB']
data = data.drop(columns=['elast','z','z_ins'])

# Plotting the graph
p_inverse_demand_log = np.exp(a1 + (a2*np.log(q_grid)) + (a3*lnY_mean) + (a4*np.log(q_grid)*lnY_mean))
p_mc = b1 + (b2*q_grid) + (b3*B_mean) + (b4*q_grid*B_mean)
plt.figure(figsize=(10, 6))
plt.scatter(data['Q'], data['P'], alpha=0.35, color='gray', edgecolors='none', label='Data (Q,P)')
plt.plot(q_grid, p_inverse_demand_log, color='navy', linewidth=2.5, label="Estimated demand")
plt.plot(q_grid, p_mc, color='firebrick', linewidth=2.5, linestyle='--', label='Estimated MC')
plt.title('NEIO estimation: isoelastic_inverse_linear-linear')
plt.xlabel('Quantity (Q)')
plt.ylabel('Price (P)')
plt.legend(frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
