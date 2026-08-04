import pandas as pd
import numpy as np
import pyblp
from linearmodels.iv import IV2SLS



"""
M&A analysis: Demand Estimation & Upward Pricing Preassure (UPP/GUPPI)
Dataset: Readt-to-Eat Cereal Market (Nevo, 2000)
"""

# 1. Prepare the dataset
path_nevo = pyblp.data.NEVO_PRODUCTS_LOCATION
df = pd.read_csv(path_nevo)
df['sum_shares'] = df.groupby('market_ids')['shares'].transform('sum')
df['s0'] = 1 - df['sum_shares']
df['y_logit'] = np.log(df['shares']) - np.log(df['s0'])

# 2. Estimation of elasticities using 2SLS (IV)
formula = 'y_logit ~ 1 + sugar + mushy + [prices ~ demand_instruments0]'
model_iv = IV2SLS.from_formula(formula, data=df)
results = model_iv.fit(cov_type='robust')
print("Demand estimation results (Logit IV - Nevo)")
print("="*60)
print(results.summary)
print("="*60)
alpha_price = results.params['prices']
price_mean = df['prices'].mean()
share_mean = df['shares'].mean()
inherent_elasticity = alpha_price * price_mean * (1 - share_mean)
cross_elasticity = - alpha_price * price_mean * share_mean
print(f"Price Alpha Coefficient: {alpha_price:.4f}")
print(f"Average own-price elasticity: {inherent_elasticity:.2f}")
print(f"Average cross-price elasticity: {cross_elasticity:.4f}")

# 3. Merger evaluation (Diversion ratio, UPP, and GUPPI)
def merger_evaluation(firm1, p1, c1, q1, firm2, p2, c2, q2, inherent_elast, cross_elast):
    absolute_margin_2 = p2 - c2
    percentage_margin_2 = absolute_margin_2 / p2
    # 1. Diversion Ratio (D12)
    mathematical_deviation = abs(cross_elast / inherent_elast) * (q2 / q1)
    d12 = min(mathematical_deviation, 0.40)    
    # 2. UPP (Upward pricing pressure in dollars, assuming zero cost efficiencies)
    upp = d12 * absolute_margin_2
    # 3. GUPPI (Gross Upward Pricing Pressure Index)
    guppi = d12 * percentage_margin_2 * (p2 / p1)    
    return d12, upp, guppi

# Select two signatures from the dataset
firm1 = 1
firm2 = 2
df_f1 = df[df['firm_ids'] == firm1]
price1 = df_f1['prices'].mean()
# The marginal cost is approximated by assuming a reasonable margin of 30% (or by estimating C = P - MC)
cost1 = price1 * 0.70  
volume1 = df_f1['shares'].sum()
df_f2 = df[df['firm_ids'] == firm2]
price2 = df_f2['prices'].mean()
cost2 = price2 * 0.70
volume2 = df_f2['shares'].sum()
d12_A, upp_A, guppi_A = merger_evaluation(
    firm1, price1, cost1, volume1, 
    firm2, price2, cost2, volume2, 
    inherent_elasticity, cross_elasticity
)

# 4. MERGER REPORT
print("\n" + "="*50)
print(f"MERGER OPINION: FIRM {firm1} + FIRM {firm2}")
print("="*50)
print(f"Diversion Ratio (D12): {d12_A * 100:.1f}% of the customers from 1 would move to 2.")
print(f"UPP (Presión Absoluta): ${upp_A:.4f} per serving.")
print(f"GUPPI (Gross Index): {guppi_A * 100:.2f}%")
if guppi_A > 0.05:
    print("\nVERDICT: RED ALERT. The GUPPI exceeds the 5% threshold.")
else:
    print("\nVERDICT: Merger viable. No significant anti-competitive pressure.")
