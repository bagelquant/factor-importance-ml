import pandas as pd
import numpy as np
from preprocessor import preprocess   # import your functions

file_path = r"C:\Users\tvenu\OneDrive\Desktop\ML\OpenAP_Macro.parquet (1).gzip"
data = pd.read_parquet(file_path, engine="pyarrow")
print(data.columns)
print(data.head())

df = pd.read_parquet(file_path, engine="pyarrow")
print("Loaded dataset: ", df.shape)
print(df.head())

# Run this once to see what industry column you have
print([c for c in df.columns if "sic" in c.lower() or "industry" in c.lower()])
[x for x in data.columns if "sic" in x.lower()]
