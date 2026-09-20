import numpy as np
import pandas as pd
import pyblp
from linearmodels.iv import IV2SLS
from scipy.optimize import minimize_scalar

pyblp.options.verbose = False

"""
Own-price elasticities by consumer income segment (BLP random-coefficients logit).
Dataset: Nevo (2000) ready-to-eat cereal data, as distributed with pyblp.
(Note: the pyblp documentation describes it as "fake cereal data": semi-fabricated data
motivated by real scanner data. It is the standard teaching dataset for demand estimation.)

Pipeline:
  1. Baseline logit with 2SLS (product fixed effects + all 20 instruments).
  2. Random-coefficients logit (BLP) with price x income heterogeneity.
  3. Own-price elasticity by consumer segment (income above / below the median),
     computed from the individual-level price coefficients.
  4. Parametric bootstrap of the BLP estimates to measure the uncertainty of the gap.
  5. Optional: the estimated parameters are printed in a format that can be loaded into a
     spreadsheet model of uniform vs. segmented pricing (Excel/VBA). The estimation
     above does not depend on this step.
"""

DRAWS = 200
SEED = 0
PRICE_UNIT = 100  # prices are expressed in cents per serving inside the model

# 0. Data
products = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
agents = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)

# 1. Baseline logit (2SLS): product FE + all instruments (strong first stage)
products['s0'] = 1 - products.groupby('market_ids')['shares'].transform('sum')
products['y_logit'] = np.log(products['shares']) - np.log(products['s0'])
dummies = pd.get_dummies(products['product_ids'], prefix='prod', drop_first=True).astype(float)
df_logit = pd.concat([products, dummies], axis=1)
instruments = ' + '.join([f'demand_instruments{i}' for i in range(20)])
formula = f"y_logit ~ 1 + {' + '.join(dummies.columns)} + [prices ~ {instruments}]"
res_logit = IV2SLS.from_formula(formula, data=df_logit).fit(cov_type='clustered', clusters=df_logit['market_ids'])
alpha_logit = res_logit.params['prices']
elasticity_logit = (alpha_logit * products['prices'] * (1 - products['shares'])).mean()
print("=" * 60)
print(f"Baseline logit (2SLS, product FE, 20 IVs): price coef = {alpha_logit:.2f}")
print(f"Average own-price elasticity: {elasticity_logit:.2f}")

# 2. Random-coefficients logit (BLP), starting values from the pyblp tutorial
product_formulations = (
    pyblp.Formulation('0 + prices', absorb='C(product_ids)'),
    pyblp.Formulation('1 + prices + sugar + mushy'),
)
agent_formulation = pyblp.Formulation('0 + income + income_squared + age + child')
problem = pyblp.Problem(product_formulations, products, agent_formulation, agents)
initial_sigma = np.diag([0.3302, 2.4526, 0.0163, 0.2441])
initial_pi = np.array([
    [5.4819, 0, 0.2037, 0],
    [15.8935, -1.2000, 0, 2.6342],
    [-0.2506, 0, 0.0511, 0],
    [1.2650, 0, -0.8091, 0],
])
results = problem.solve(initial_sigma, initial_pi,
                        optimization=pyblp.Optimization('bfgs', {'gtol': 1e-5}), method='1s')
print("=" * 60)
print(f"BLP price coefficient (beta): {results.beta[0, 0]:.2f}")
print(f"BLP average own-price elasticity: "
      f"{np.mean(np.concatenate([np.diag(results.compute_elasticities(market_id=t)) for t in products['market_ids'].unique()])):.2f}")

# 3. Elasticities by consumer segment
markets = list(products['market_ids'].unique())
prod_idx = {t: np.where(products['market_ids'].values == t)[0] for t in markets}
agent_idx = {t: np.where(agents['market_ids'].values == t)[0] for t in markets}
X2 = np.column_stack([np.ones(len(products)), products['prices'], products['sugar'], products['mushy']])
NODES = agents[[f'nodes{k}' for k in range(4)]].values
DEMOG = agents[['income', 'income_squared', 'age', 'child']].values
WEIGHTS = agents['weights'].values
INCOME = agents['income'].values
INCOME_MEDIAN = np.median(INCOME)


def segment_elasticities(delta, sigma, pi, beta_price):
    """
    Own-price elasticity of the demand of each income segment (high / low).
    For consumers i in segment g: S_jg = sum_i w_i * s_ij and
    dS_jg/dp_j = sum_i w_i * alpha_i * s_ij * (1 - s_ij), with alpha_i = beta + sigma*nu_i + pi*d_i.
    Segment elasticity = (p_j / S_jg) * dS_jg/dp_j, averaged over products/markets weighted by S_jg.
    """
    acc = {'high': ([], [], []), 'low': ([], [], [])}
    for t in markets:
        j, i = prod_idx[t], agent_idx[t]
        coef = NODES[i] * np.diag(sigma)[None, :] + DEMOG[i] @ pi.T
        mu = coef @ X2[j].T
        u = np.exp(delta[j][None, :] + mu)
        s = u / (1 + u.sum(axis=1, keepdims=True))
        alpha_i = beta_price + coef[:, 1]
        for seg, mask in (('high', INCOME[i] > INCOME_MEDIAN), ('low', INCOME[i] <= INCOME_MEDIAN)):
            w = WEIGHTS[i][mask] / WEIGHTS[i][mask].sum()
            share = (w[:, None] * s[mask]).sum(axis=0)
            d_share = (w[:, None] * alpha_i[mask, None] * s[mask] * (1 - s[mask])).sum(axis=0)
            p = products['prices'].values[j]
            acc[seg][0].append(p * d_share / share)
            acc[seg][1].append(share)
            acc[seg][2].append(p)
    out = {}
    for seg, (e, sh, p) in acc.items():
        e, sh, p = np.concatenate(e), np.concatenate(sh), np.concatenate(p)
        out[seg] = {'elasticity': -np.average(e, weights=sh), 'share': sh.mean(), 'price': p.mean()}
    return out


def calibrate(seg, w_high=0.5, w_low=0.5):
    """
    Excel inputs for constant iso-elastic demand Q_g = A_g * P^(-e_g) and constant marginal cost c.
    A_g is calibrated to reproduce the observed quantity at the observed price; c is the marginal
    cost that makes the observed uniform price optimal for the firm (c = P * (1 - 1/e_agg)).
    Quantities are in market-share points (x100); prices in cents per serving.
    """
    e_h, e_l = seg['high']['elasticity'], seg['low']['elasticity']
    p = seg['high']['price'] * PRICE_UNIT
    q_h, q_l = 100 * w_high * seg['high']['share'], 100 * w_low * seg['low']['share']
    a_h, a_l = q_h * p ** e_h, q_l * p ** e_l
    e_agg = (e_h * q_h + e_l * q_l) / (q_h + q_l)
    c = p * (1 - 1 / e_agg)

    def profit(p1, p2):
        return (p1 - c) * a_h * p1 ** (-e_h) + (p2 - c) * a_l * p2 ** (-e_l)

    uniform = minimize_scalar(lambda x: -profit(x, x), bounds=(c * 1.001, c * 20),
                              method='bounded', options={'xatol': 1e-12})
    p_h, p_l = c * e_h / (e_h - 1), c * e_l / (e_l - 1)  # inverse elasticity rule, constant MC
    gain = (profit(p_h, p_l) / -uniform.fun - 1) * 100
    return {'A1': a_h, 'e1': e_h, 'A2': a_l, 'e2': e_l, 'c': c,
            'P_uniform': uniform.x, 'profit_uniform': -uniform.fun,
            'P_high': p_h, 'P_low': p_l, 'profit_discrim': profit(p_h, p_l), 'gain_pct': gain}


point = segment_elasticities(results.delta.flatten(), results.sigma, results.pi, results.beta[0, 0])
cal = calibrate(point)
print("=" * 60)
print("Own-price elasticity by income segment (point estimate):")
print(f"  High income: {-point['high']['elasticity']:.2f}")
print(f"  Low income:  {-point['low']['elasticity']:.2f}")

# 4. Parametric bootstrap
boot = results.bootstrap(draws=DRAWS, seed=SEED)
e_high, e_low, gains = [], [], []
for b in range(DRAWS):
    seg_b = segment_elasticities(boot.bootstrapped_delta[b].flatten(), boot.bootstrapped_sigma[b],
                                 boot.bootstrapped_pi[b], boot.bootstrapped_beta[b][0, 0])
    e_high.append(seg_b['high']['elasticity'])
    e_low.append(seg_b['low']['elasticity'])
    gains.append(calibrate(seg_b)['gain_pct'])
e_high, e_low, gains = np.array(e_high), np.array(e_low), np.array(gains)
print("=" * 60)
print(f"Bootstrap ({DRAWS} draws), 5th - 95th percentile:")
print(f"  High income elasticity: {np.percentile(e_high, 5):.2f} - {np.percentile(e_high, 95):.2f}")
print(f"  Low income elasticity:  {np.percentile(e_low, 5):.2f} - {np.percentile(e_low, 95):.2f}")
print(f"  Share of draws where low income is more elastic: {(e_low > e_high).mean() * 100:.0f}%")
print(f"  Profit gain from segmentation: {np.percentile(gains, 5):.1f}% - {np.percentile(gains, 95):.1f}%")

# 5. Optional: parameters formatted as inputs for a spreadsheet pricing model (sheet "Discriminacion", left block)
print("=" * 60)
print("EXCEL INPUTS (Discriminacion, left block)")
print(f"  E2 (A1, high income)  = {cal['A1']:.4f}")
print(f"  E3 (e1)               = {cal['e1']:.6f}")
print(f"  E4 (A2, low income)   = {cal['A2']:.4f}")
print(f"  E5 (e2)               = {cal['e2']:.6f}")
print("  E6 (a) = 0 ; E7 (b) = 0 ; E9 (tax) = 0   (constant marginal cost, no tax)")
print(f"  E8 (c)                = {cal['c']:.4f}")
print("EXPECTED SOLVER RESULTS")
print(f"  Uniform price: {cal['P_uniform']:.2f} | profit: {cal['profit_uniform']:.2f}")
print(f"  Segmented prices: {cal['P_high']:.2f} (high) / {cal['P_low']:.2f} (low) | profit: {cal['profit_discrim']:.2f}")
print(f"  Profit gain: {cal['gain_pct']:.2f}%")
