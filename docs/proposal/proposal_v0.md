---
title: "Feature Augmentation and Economic Group Importance in Cross-Sectional Return Prediction"
dataset: "Open Asset Pricing Dataset (https://www.openassetpricing.com/data/)"
---

## 1. Motivation

Machine learning methods have shown strong predictive power in asset pricing (Gu, Kelly, & Xiu, 2020), largely due to their ability to capture nonlinear and high-dimensional relationships among firm characteristics. However, while these models achieve high out-of-sample accuracy, the **economic interpretation of features** remains limited.  

This project aims to bridge the gap between predictive accuracy and interpretability by **augmenting the feature space** with peer-based and fixed-effect features and by **quantifying the economic importance of feature groups**. Specifically, we use the Open Asset Pricing dataset — a large-scale, cleaned collection of firm-level monthly characteristics — to systematically assess how different economic categories of features contribute to model performance.

## 2. Objectives

1. **Enhance the predictive feature set** by incorporating structural and relational information, including:
   - **Firm fixed effects** (via categorical encoding of `permno`)
   - **Peer group features**, such as peer-averaged characteristics and peer average returns  
2. **Train multiple predictive models** to forecast future stock returns using the augmented dataset:
   - Elastic Net (linear, regularized baseline)
   - Gradient Boosted Trees (nonlinear, ensemble model)
   - Feedforward Neural Network (nonlinear, flexible)
3. **Assess the economic relevance** of feature groups by **systematically removing one economic group at a time** and observing the drop in model performance.  
   This quantifies which economic dimensions — e.g., valuation, profitability, investment, momentum — drive cross-sectional predictability.

## 3. Data and Feature Engineering

### 3.1 Dataset  

- **Source:** Open Asset Pricing dataset (monthly frequency, CRSP–Compustat merged sample).  
- **Observation unit:** Firm-month (identified by `permno`, `date`).  
- **Target variable:** Next-month excess return.  
- **Feature space:** 400+ firm characteristics grouped into economic categories (e.g., valuation, profitability, investment, risk, momentum, liquidity).

### 3.2 Feature Augmentation  


| Feature Type | Description | Construction |
|---------------|-------------|---------------|
| **Fixed Effect** | Controls for persistent firm-level heterogeneity | Treat `permno` as a categorical variable (encoded via one-hot or embeddings) |
| **Peer Group Average (K-Clustering)** | Firms with similar characteristics are clustered into *K* peer groups in training data | Use K-means on standardized feature space |
| **Peer Feature Average** | Average of each feature group within the same peer cluster | Compute mean within cluster per feature group |
| **Peer Return Average** | Peer average of previous-period returns | Add lagged peer return as contextual signal |


### 3.3 Feature Grouping  

Features are grouped by economic meaning (from the OpenAP taxonomy):

1. Valuation Ratios  
2. Profitability & Operating Performance  
3. Investment, Growth & Financing  
4. Momentum & Reversal  
5. Risk & Volatility  
6. Liquidity & Microstructure  
7. Accruals & Earnings Quality  
8. Analyst & Forecast-based Signals  
9. Intangibles & Innovation  
10. Governance & Ownership  

These groups will be used for the **ablation (group removal) analysis** in Stage 2.

## 4. Modeling Approach

### 4.1 Algorithms
1. **Elastic Net Regression**
   - Baseline linear model with L1/L2 regularization  
   - Interpretable feature weights, efficient training  
2. **Gradient Boosted Trees (e.g., LightGBM / XGBoost)**
   - Captures nonlinearities and interactions  
   - Handles missing data natively  
3. **Feedforward Neural Network**
   - Multi-layer architecture with dropout and batch normalization  
   - Flexible representation learning on augmented features  

### 4.2 Model Training Setup

- **Evaluation metrics:**  

  - Predictive R-squared (cross-sectional)  
  - Mean Squared Error (MSE)  
  - Rank IC (Spearman correlation between predicted and realized returns)  
  - Portfolio backtest (optional, decile spread returns)  

## 5. Feature Group Importance Experiment

To evaluate **economic interpretability**, the following **group ablation procedure** will be conducted:

1. Train the baseline model with **all features**.  
2. Sequentially **remove one feature group** (e.g., valuation, momentum, etc.) at a time.  
3. Re-train and evaluate the model’s performance on test data.  
4. Compute **Change in Performance = Baseline - Reduced Model** for each group.  
5. Repeat across models (Elastic Net, GBT, NN) and multiple time periods to assess **temporal stability**.  

The resulting performance drop quantifies **each group’s marginal contribution** to prediction accuracy — providing an interpretable “economic feature importance ranking.”

## 6. Expected Contributions

- **Methodological:** Introduces structural (fixed-effect) and relational (peer-based) augmentation to traditional firm characteristic datasets.  
- **Empirical:** Quantifies the predictive importance of different economic feature families over time.  
- **Pedagogical:** Demonstrates a replicable, interpretable machine learning pipeline bridging academic factor research and modern ML methods.

## 7. Timeline


| Week | Task |
|------|------|
| 1 | Data cleaning and feature grouping |
| 2 | Peer clustering and feature augmentation |
| 3 | Model setup and baseline training |
| 4 | Group ablation experiments |
| 5 | Performance comparison and visualization |
| 6 | Report writing and presentation preparation |

## 8. Deliverables

- Python notebooks (data processing, modeling, visualization)  
- Feature importance and ablation analysis report  
- Presentation slides summarizing model results and insights

## 9. References

- Gu, Kelly, & Xiu (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies.  
- Green, Hand, & Zhang (2017). *The Characteristics that Provide Independent Information about Average U.S. Stock Returns.*  
- Bryzgalova, Pelger, & Zhu (2023). *Principal Components of Characteristic Portfolios.*  
- Kelly, Pruitt, & Su (2019). *Characteristics Are Covariances: A Unified Model of Risk and Return.*

