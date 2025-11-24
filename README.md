# Prediction Stock Monthly Returns Using Basic Machine Learning Models

Update date: Nov 2025

This is project for Rutgers Business School course "Machine Learning in Finance", taught by [Prof. Serdar Dinc](https://www.business.rutgers.edu/faculty/serdar-dinc)

Project aims to predict monthly stock returns using basic machine learning models such as:

- Elastic Net Regression
- Gradient Boosting Trees
- Neural Networks

Then by adding peer effects and see if the prediction performance improves.

The data used in this project is provided by Prof. Dinc, using the [openap dataset](https://www.openassetpricing.com/data/), the stock universe includes all stocks CRSP.

## Team Members

By alphabetical order:

- [Sravya Madireddi]()
- [Thara Venu]()
- [Yanzhong(Eric) Huang](https://bagelquant.com/about-me/)

## The data pipeline

Here is an overview of the data pipeline, the whole logic of the project

1. load raw data -> `src/data_loader.py`
2. preprocess data -> `src/preprocessor.py`
  - standardization
  - missing value handling
  - categorical encoding
3. train the baseline NN model with embedding of permno
4. adding peer-based features
  - `dist_peer`: distance weighted features (a continuous version of peer features)
  - industry encoding
5. models, definitions of models -> `src/models/`
  - `elastic_net.py`
  - `xgboost_tree.py`
  - `neural_network.py`
6. training -> `notebooks/train_xxx.ipynb`
  - baseline model using only firm characteristics + permno embedding
  - model with `dist_peer` feature
  - model with industry encoding
  - model with both `dist_peer` and industry encoding
7. compare the performance of different models in `notebooks/evaluate_models.ipynb`

## Module Organization

```plaintext
project_root/
│├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── data_splitter.py
│   ├── models/
│   │   ├── elastic_net.py
│   │   ├── xgboost_tree.py
│   │   └── neural_network.py
│├── notebooks/
│   ├── train_elastic_net.ipynb
│   ├── train_xgboost_tree.ipynb
│   ├── train_neural_network.ipynb
│   └── evaluate_models.ipynb
│├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│├── docs/
│   ├── modules/
│   │   ├── module_1.md
│   │   ├── module_2.md
│   │   └── module_3.md
│   ├── module_structure.md
│   ├── progress.md
│├── .gitignore
│└── README.md

```


## Progress

- [Initial Proposal](docs/proposal/proposal_v0.md)
- [Progress Log](docs/progress_log.md)

## Report

The final report and presentation will be added here upon project completion.
