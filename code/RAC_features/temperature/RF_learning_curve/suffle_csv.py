import pandas as pd

df=pd.read_csv('extended_big_additive_category.csv')
  
# shuffle the DataFrame rows
df = df.sample(frac = 1)

df.to_csv('suffled_extended_big_additive_category.csv',index=False)

