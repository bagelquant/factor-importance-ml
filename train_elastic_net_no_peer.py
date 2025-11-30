# %% [markdown]
"""
# Train ElasticNet with Embeddings (No Peer Features)

This script trains an ElasticNet model using features augmented with:
1. The embedding vectors themselves.
*Excluding* peer-based features.

It iterates over prediction years, using a "walk-forward" validation scheme:
- **Auto-tuning**: Uses `train_df` (historical) and `val_df` (recent past).
- **Final Training**: Uses `train_df` + `val_df`.
- **Prediction**: Uses `test_df` (future).

"""

# %% 
import json
import time
import pandas as pd
from pathlib import Path
from src.ml_backend import load_train_ready_data_no_peer, split_data, ElasticNet

# time tracking
start_time = time.perf_counter()

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
   
    # 1. Load Train/Val/Test data with embeddings (NO PEER FEATURES)
    data_v1, data_v2 = load_train_ready_data_no_peer(test_year=year)

    # Fill NaNs (introduced by OOV embeddings) with 0.0
    data_v1 = data_v1.fillna(0.0)
    data_v2 = data_v2.fillna(0.0)

    # 2. Split data_v1 for Auto-Tuning
    train_df_v1, val_df_v1, _ = split_data(data_v1, predict_year=year)

    # 3. Split data_v2 for Final Training & Prediction
    train_df_v2, val_df_v2, test_df_v2 = split_data(data_v2, predict_year=year)

    # 4. Initialize and Train Model
    model = ElasticNet(
        train_df=train_df_v1,
        val_df=val_df_v1,
        test_df=test_df_v2,      # Used for prediction if test_df_final not specified
        train_df_final=train_df_v2, # Used for final training
        val_df_final=val_df_v2,     # Used for final training
        test_df_final=test_df_v2    # Explicitly provided for prediction
    )

    print("Auto-tuning...")
    model.auto_tune() 
    
    print("Final training...")
    model.train_final() 
    
    print("Predicting...")
    predictions = model.predict()

    # Store predictions with the corresponding year
    all_predictions.append(predictions)


# %% [markdown]
"""
## Save all predictions to a CSV file
"""

# %% 
# Concatenate all predictions into a single Series
final_predictions = pd.concat(all_predictions)

# To get the true target values, we can extract them from the test splits we used.
from src.ml_backend import load_processed
raw_data = load_processed(reprocesse=False)
test_targets = raw_data.loc[final_predictions.index, "retadj_next"]

# Create a DataFrame to store predictions and true values
results_df = pd.DataFrame({
    "predicted_retadj_next": final_predictions,
    "true_retadj_next": test_targets
})

# Save the results to a CSV file
results_path = Path("data/predictions/elastic_net_no_peer_predictions.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(results_path, index=True)

# time tracking
end_time = time.perf_counter()
elapsed_time = end_time - start_time
elapsed_hours_minutes_seconds = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

# save to a text file
time_log_path = Path("data/predictions/elastic_net_no_peer_time_log.txt")
time_log_path.parent.mkdir(parents=True, exist_ok=True)
with open(time_log_path, "w") as f:
    f.write(f"Total elapsed time: {elapsed_hours_minutes_seconds}\n")

print("Process complete.")
