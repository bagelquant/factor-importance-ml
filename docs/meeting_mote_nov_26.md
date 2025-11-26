# Meeting note Nov 26

## Data pipeline updates

1. Neural network model with embeddings
  - train-ready dataset 
    - oringinal data (208)
    - embeddings dimensions (100)
    - peer-based features (208)
  - Each iteration will have its own dataset file (one file containing everything)

2. Model training
 - Same data split as before (train, val, test)
 - split_data(df) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]
 - models (We have already implemented, all we need to do is call the functinos, change the `configs.json if needed)
    - ElasticNet (run twice, once with peer features, once without)
    - GradientBoostingTree 
    - NeuralNetwork

  - output
    - predictions + real values (pd.DataFrame, in entire time range 2000-2020)

3. Evaluation
  - metrics calculation (MAE, RMSE, R2)
  - Diebold-Mariano test (pair comparison of models)
    - compare between 3 models -> 3 pairs (AB, AC, BC)
    - compare between with/without peer features for each model -> 3 pairs (AA, BB, CC)

4. Ensemble model
  - average predictions of all models (with peer features)
  - weights based on individual permno level

## TODO


### Due Thursday 11/28

- [ ] Finish peer-based features -> Eric 

### Due Sunday 11/30

- [ ] Finish training
  - [ ] ElasticNet -> Sravya
  - [ ] GradientBoostingTree -> Thara
  - [ ] NeuralNetwork -> Eric

### Due Monday 12/1

- [ ] Finish evaluation
  - [ ] Metrics calculation
  - [ ] Diebold-Mariano test -> Eric
- [ ] Ensemble model

