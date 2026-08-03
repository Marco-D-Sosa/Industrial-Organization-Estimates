import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import kagglehub



"""
First estimate of network effects in the console market.
Dataset: Video Game Sales (sourced from Kaggle).

In the absence of data on console sales, game sales by platform are used as a proxy (since the more consoles sold, the more 
games are purchased for them). The problem with this proxy is that game sales may be higher due to a larger library 
without necessarily translating into higher console sales; consequently, the network effect could be overestimated.

Note: Ensure you have your Kaggle API Token configured in your environment
or local directory so kagglehub can authenticate automatically.
"""

# 1. Open dataset
path = kagglehub.dataset_download("gregorut/videogamesales")
df_raw = pd.read_csv(f"{path}/vgsales.csv")

# 2. Set up the panel
df_raw['Year'] = pd.to_numeric(df_raw['Year'],errors='coerce')
df = df_raw.dropna(subset=['Year']).copy()
df['Year'] = df['Year'].astype(int)
panel = df.groupby(['Platform','Year']).agg(Global_Sales=('Global_Sales', 'sum'), New_Games=('Name', 'count')).reset_index()
panel = panel.sort_values(by=['Platform','Year'])
panel['cumulative_catalog'] = panel.groupby('Platform')['New_Games'].cumsum()
panel = panel[ (panel['Global_Sales']>0) & (panel['cumulative_catalog']>0) ].copy()
panel['ln_sales'] = np.log(panel['Global_Sales'])
panel['ln_network'] = np.log(panel['cumulative_catalog'])
print(panel[panel['Platform'] == 'PS2'] [['Year','New_Games','cumulative_catalog','Global_Sales']].head())
print("="*60)

# 3. Estimation of the network effect
#Model: log(sales) = log(network) + FE per platform
model = smf.ols('ln_sales ~ ln_network + C(Platform)', data=panel).fit(cov_type='HC1')
print("Results: elasticity of the network effect")
print("="*60)
print(model.summary())
print("="*60)
elasticity_network = model.params['ln_network']
print(f"Elasticity of the network: {elasticity_network:.4f}")
print("="*60)
