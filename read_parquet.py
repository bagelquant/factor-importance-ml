import pandas as pd
from pathlib import Path

path_train_embed = Path(f"data/embeddings/train_embeddings_2009.parquet.gzip")
train_embeddings = pd.read_parquet(path_train_embed)

print(train_embeddings)

path_train_val_embed = Path(f"data/embeddings/train_val_embeddings_2009.parquet.gzip")
train_plus_val_embeddings = pd.read_parquet(path_train_val_embed)

print(train_plus_val_embeddings)
