import pandas as pd
import numpy as np

#loading and checking data 
df = pd.read_csv("wallacecommunications.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df['new_contract_this_campaign'].value_counts())   # checking imbalance
