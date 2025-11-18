# Module Structure Documentation

This document outlines the structure of the modules used in the machine learning project. It details the organization of code, data handling, and the relationships between different components.

## The data pipeline

Here is an overview of the data pipeline, the whole logic of the project

1. load raw data -> `src/data_loader.py`
2. preprocess data -> `src/preprocessor.py`  -> Thara
  - standardization
  - missing value handling
  - categorical encoding
3. feature engineering -> `src/feature_engineer.py`  -> Yanzhong
  - adding new features (e.g., peer group statistics)
4. data splitting -> `src/data_splitter.py` -> Sravya
  - train-test split
  - cross-validation setup
5. models, definitions of models -> `src/models/`
  - `elastic_net.py`
  - `xgboost_tree.py`
  - `neural_network.py`
6. training -> `notebooks/train_xxx.ipynb`
  - model training scripts for each model
7. evaluation -> `notebooks/evaluate_models.ipynb`

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


