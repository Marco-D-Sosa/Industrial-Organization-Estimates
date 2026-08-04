import pandas as pd
import kagglehub



"""
Concentration analysis (HHI).
Dataset: Warehouse and Retail Sales (Kaggle).
"""

# 1. Open dataset
path = kagglehub.dataset_download("sahirmaharajj/retail-sales-analysis")
df = pd.read_csv(f"{path}/Warehouse_and_Retail_Sales.csv")
df.columns = df.columns.str.lower()
print(df)
print("="*60)

# 2. Calculation of the total HHI
df['retail sales'] = pd.to_numeric(df['retail sales'], errors='coerce').fillna(0)
df['warehouse sales'] = pd.to_numeric(df['warehouse sales'], errors='coerce').fillna(0)
df['total_cases_sold'] = df['retail sales'] + df['warehouse sales'] #Retail and wholesale sales are combined
aggregate_sales = df['total_cases_sold'].sum()
supplier_sales = df.groupby('supplier')['total_cases_sold'].sum()
market_shares_total = (supplier_sales / aggregate_sales) * 100
Total_hhi = (market_shares_total ** 2).sum()
print(f"Total HHI (Aggregate Market – by case volume): {Total_hhi:.2f}")
print("="*60)

# 3. Calculation of the HHI by beverage type
Type_sales_supp = df.groupby(['item type', 'supplier'])['total_cases_sold'].sum() #Calculation of sales by category and supplier
Type_sales = df.groupby('item type')['total_cases_sold'].sum()
market_shares_type = (Type_sales_supp / Type_sales) * 100
hhi_type = (market_shares_type ** 2).groupby('item type').sum().reset_index(name='HHI')
print("HHI by Beverage Type (Top 10 most concentrated markets):")
print(hhi_type.sort_values(by='HHI', ascending=False).head(10))
print("="*60)
