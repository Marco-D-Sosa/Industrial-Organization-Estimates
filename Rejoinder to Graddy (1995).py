import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS



"""
This dataset is based on Graddy (1995). 
The author used transaction data from a fish stall to test for the existence of third-degree 
price discrimination. As a result, she was able to demonstrate such discrimination.
"""

# Open dataset
fish = pd.read_stata(r"C:\Users\HP\Downloads\Industrial Organization Estimates\Bases de datos\fish_stata.dta")
print(fish.describe())
print("------------------------------------")

# Observe that fishing takes place from Monday to Friday (closed Saturdays and Sundays); 
# dummy variables define the days.
fish['frid'] = 0
condition = (fish['mon']==0) & (fish['tues']==0) & (fish['wed']==0) & (fish['thurs']==0) 
fish.loc[condition, 'frid'] = 1 
fish['day'] = fish['mon'] + 2*fish['tues'] + 3*fish['wed'] + 4*fish['thurs'] + 5*fish['frid']
fish.sort_values(by='day', inplace=True)

# Calculate statistics for the weighted average price (avgprc) and total quantity (totqty) for each day
print(fish.groupby('day')[['avgprc','totqty']].describe())
print("------------------------------------")
print("------------------------------------")

# Calculate daily price statistics for Asians (prca) and Whites (prcw)
# and the total quantities for Asians (qtya) and Whites (qtyw)
print(fish.groupby('day')[['prca','prcw','qtya','qtyw']].describe())
print("------------------------------------")

# Define the variable "t" as the time variable
fish.set_index('t', inplace=True)

# Present the database (endogeneity issues)
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
fish.plot.scatter(ax=ax1, x='ltotqty', y='lavgprc')
ax1.set_title('')
ax1.set_ylabel('Ln(Weighted average price)')
ax1.set_xlabel('Ln(Quantity)')
plt.show

reg = smf.ols('ltotqty ~ lavgprc', data=fish).fit()
print(reg.summary())
print("------------------------------------")



"""
Factor the variables of "waves" and wind "speed" into the pricing:
The cost of obtaining fish increases with wave and wind conditions, but this should not 
affect the demand for fish in restaurants, households, etc.

Why do waves and wind have an effect?
(1) Shipping costs, shipping safety, fuel consumption.
(2) The fish can scatter and move to deeper waters, reducing the catch.

vce(hac nwest opt) corrects for possible autocorrelation with the past and for heteroskedasticity
"""

fish = sm.add_constant(fish)
mod = IV2SLS(fish['ltotqty'], fish[['mon','tues','wed','thurs','const']], fish[['lavgprc']], fish[['wave2','wave3','speed2','speed3']])
reg_2 = mod.fit(cov_type='kernel', kernel='bartlett')
print(reg_2)
print("------------------------------------")

# Now estimate the demands separately (Asians and Whites):

# Asians
fish['lprca'] = np.log(fish['prca'])
fish['lqtya'] = np.log(fish['qtya'])
mod_a = IV2SLS(fish['lqtya'], fish[['mon','tues','wed','thurs','const']], fish[['lprca']], fish[['wave2','wave3','speed2','speed3']])
reg_a = mod_a.fit(cov_type='kernel', kernel='bartlett')
print(reg_a)
print("------------------------------------")
# Whites
fish['lprcw'] = np.log(fish['prcw'])
fish['lqtyw'] = np.log(fish['qtyw'])
mod_w = IV2SLS(fish['lqtyw'], fish[['mon','tues','wed','thurs','const']], fish[['lprcw']], fish[['wave2','wave3','speed2','speed3']])
reg_w = mod_w.fit(cov_type='kernel', kernel='bartlett')
print(reg_w)
print("------------------------------------")
