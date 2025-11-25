# Hyperparameters configuration

The hyperparameters for the model training and evaluation are defined in the following sections.

> Change `configs.json` hyperparameters section accordingly if needed.

## Random seed

- `random_seed` (fixed)
  - Description: Seed for random number generators to ensure reproducibility.
  - Type: int
  - Value: 666

## Elastic Net Hyperparameters

- `alpha` (tune)
  - Description: Constant that multiplies the penalty terms. Defaults to 1.0.
  - Type: float
  - Values: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- `l1_ratio` (tune)
  - Description: The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is an L1 penalty.
  - Type: float
  - Values: [0.0, 0.25, 0.5, 0.75, 1.0]
- `max_iter` (fixed)
  - Description: The maximum number of iterations for the solver to converge.
  - Type: int
  - Value: 1000
- `tol` (fixed)
  - Description: The tolerance for the optimization.
  - Type: float
  - Value: 1e-4

## Gradient Tree Boosting Hyperparameters

- `n_leaves` (tune)
  - Description: The maximum number of leaves in one tree.
  - Type: int
  - Values: [3, 7, 15, 31, 63]
- `depth` (recommended)
  - Description: The maximum depth of each tree. Use in conjunction with `n_leaves` to control tree complexity.
  - Type: int
  - Example values: [2, 3, 4, 5, 6]
- `learning_rate_lambda` (tune)
  - Description: The learning rate shrinks the contribution of each tree by `learning_rate_lambda`. There is a trade-off between learning_rate_lambda and n_estimators.
  - Type: float
  - Example values: [0.01]
- `n_estimators` (tune)
  - Description: The number of boosting stages to be run (trees).
  - Type: int
  - Example value: 5000 (use smaller values for quick experiments)
- `column_sample_rate` (tune)
  - Description: The fraction of columns to be randomly sampled for each tree.
  - Type: float
  - Example heuristic: sqrt(number of features) / number of features

## Neural Network Hyperparameters

- `hidden_layers` (tune)
  - Description: The number of hidden layers in the neural network.
  - Type: int
  - Values: [1, 2, 3, 4, 5]
- `neurons_per_layer` (tune)
  - Description: The number of neurons in each hidden layer.
  - Type: int
  - Values: [16, 32, 64, 128, 256]
- `activation_functions` (tune)
  - Description: The activation function to use in the hidden layers.
  - Type: str
  - Values: ['relu', 'tanh', 'sigmoid']
- `learning_rates` (tune)
  - Description: The learning rate for the optimizer.
  - Type: float
  - Values: [0.001, 0.01, 0.1]
`batch_sizes` (tune)
  - Description: The number of samples per gradient update.
  - Type: int
  - Values: [16, 32, 64, 128]

- `training` (fixed)
  - `final_epochs`: The default number of epochs to run for final training (when not using early stopping).
    - Type: int
    - Example value: 100
  - `use_early_stopping`: Whether to enable early stopping during final training.
    - Type: bool
    - Example value: true
  - `early_stopping_patience`: Number of epochs with no improvement on validation to wait before stopping.
    - Type: int
    - Example value: 10

## Neural Network with Embeddings Hyperparameters

- `embedding_dims` (tune)
  - Description: The dimensions of the embedding layers for categorical features.
  - Type: int
  - Values: [4, 8, 16, 32]
