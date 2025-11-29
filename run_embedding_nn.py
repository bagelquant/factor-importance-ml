# %% [markdown]
"""
# Run the neural network with embeddings on the processed dataset.

This Notebook will:

1. Load the processed dataset.
2. Iterate over a range of prediction years.
3. For each year:
   - Split the data into training, validation, and testing sets.
   - Initialize and train a neural network model with embeddings.
   - Call `model.auto_tune()` to perform hyperparameter tuning.
   - Call `model.train_final()` to train the final model.
   - Call `model.predict()` to make predictions on the test set.
4. save results for each year.
   - save the predictions to a list, concat all iterations and save to a csv file.
   - save embeddings vectors for training to a parquet file.
   - save embeddings vectors for training + validation to a parquet file.

"""

# %%
import json
import pandas as pd
import time
from pathlib import Path
from src.ml_backend import load_processed
from src.ml_backend import split_data
from src.ml_backend import NeuralNetworkWithEmbeddings

# time tracking
start_time = time.perf_counter()

# Load processed data
data = load_processed(reprocesse=False)

# %% [markdown]
"""
## Interation over prediction years

All the setup is store in `configs.json`, only need to read the test_end year
"""

# %%
# Read configuration parameters
with open("configs.json", "r") as f:
    config = json.load(f)["train_iteration"]
    test_end = config["test_end"]

print(f"Test will end at year: {test_end}")

# %%
# Initialize lists to store results
all_predictions: list[pd.Series] = []

# Iterate over prediction years
for year in range(2009, test_end + 1):
    print(f"{'='*30}\nProcessing year: {year}")
   
    # Split data into training, validation, and testing sets
    train_df, val_df, test_df = split_data(data, predict_year=year)
   
    # Initialize the neural network model with embeddings
    model = NeuralNetworkWithEmbeddings(train_df=train_df,
                                        val_df=val_df,
                                        test_df=test_df)

    model.auto_tune()  # auto-tune hyperparameters
    model.train_final()   # train the train+val set with best hyperparameters
    predictions: pd.Series= model.predict()  # make predictions on the test set

    # Store predictions with the corresponding year
    all_predictions.append(predictions)

    # Store embeddings
    train_embeddings: pd.DataFrame = model.best_train_embedding_vectors  # type: ignore
    train_plus_val_embeddings: pd.DataFrame = model.combined_train_val_embedding_vectors  # type: ignore
   
    # save embeddings to parquet files, crete all necessary directories
    path_train_embed = Path(f"data/embeddings/train_embeddings_{year}.parquet.gzip")
    path_train_embed.parent.mkdir(parents=True, exist_ok=True)
    train_embeddings.to_parquet(path_train_embed, compression="gzip")

    path_train_val_embed = Path(f"data/embeddings/train_val_embeddings_{year}.parquet.gzip")
    path_train_val_embed.parent.mkdir(parents=True, exist_ok=True)
    train_plus_val_embeddings.to_parquet(path_train_val_embed, compression="gzip")

# %% [markdown]
"""
## Save all predictions to a CSV file
"""

# %%
# Concatenate all predictions into a single Series
final_predictions = pd.concat(all_predictions)

# concat the real target values for the test set
test_targets = data.loc[final_predictions.index, "retadj_next"]

# Create a DataFrame to store predictions and true values
results_df = pd.DataFrame({
    "predicted_retadj_next": final_predictions,
    "true_retadj_next": test_targets
})

# Save the results to a CSV file
results_path = Path("data/predictions/nn_with_embeddings_predictions.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_path, index=True)

# time tracking
end_time = time.perf_counter()
elapsed_time = end_time - start_time
elapsed_hours_minutes_seconds = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
# save to a text file
time_log_path = Path("data/predictions/nn_with_embeddings_time_log.txt")
time_log_path.parent.mkdir(parents=True, exist_ok=True)
with open(time_log_path, "w") as f:
    f.write(f"Total elapsed time: {elapsed_hours_minutes_seconds}\n")
