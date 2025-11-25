# Hyperparameters configuration

The hyperparameters for the model training and evaluation are defined in the following sections.

> Change `config.json` hyperparameters section accordingly if needed.

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

## Gradient Tree Booosting Hyperparameters

- `n_leaves` (tune)
  - Description: The maximum number of leaves in one tree.
  - Type: int
  - Values: [3, 7, 15, 31, 63]
- `depth` (fixed)
  - Description: The maximum depth of each tree. Based on n_leaves.
  - Type: int
  - Values: [2, 3, 4, 5, 6] accordingly to n_leaves
- `learning_rate_lambda` (fixed)
  - Description: The learning rate shrinks the contribution of each tree by `learning_rate_lambda`. There is a trade-off between learning_rate_lambda and n_estimators.
  - Type: float
  - Value: 0.01
- `n_estimators` (fixed)
  - Description: The number of boosting stages to be run. (trees), will use early stopping.
  - Type: int
  - Value: 5000
- `column_sample_rate` (fixed)
  - Description: The fraction of columns to be randomly sampled for each tree.
  - Type: float
  - Value: sqrt(number of features) / number of features


## Neural Network Hyperparameters

- `hidden_layers` (tune)
  - Description: The number of hidden layers in the neural network.
  - Type: int
  - Values: [1, 2, 3, 4, 5]
- `neurons_per_layer` (tune)
  - Description: The number of neurons in each hidden layer.
  - Type: int
  - Values: [16, 32, 64, 128, 256]
- `activation_function` (tune)
  - Description: The activation function to use in the hidden layers.
  - Type: str
  - Values: ['relu', 'tanh', 'sigmoid']
- `learning_rate` (tune)
  - Description: The learning rate for the optimizer.
  - Type: float
  - Values: [0.001, 0.01, 0.1]
- `batch_size` (tune)
  - Description: The number of samples per gradient update.
  - Type: int
  - Values: [16, 32, 64, 128]
- `epochs` (fixed)
  - Description: The number of epochs to train the model.
  - Type: int
  - Value: 1000

## Neural Network with Embeddings Hyperparameters

- `embedding_dims` (tune)
  - Description: The dimensions of the embedding layers for categorical features.
  - Type: int
  - Values: [4, 8, 16, 32]
