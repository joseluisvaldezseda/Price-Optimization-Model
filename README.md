# Optimal Price Optimization using Demand and Profit Curves

This Jupyter Notebook implements a price optimization workflow based on **demand curves** and **profit curves**, leveraging multiple machine learning models to estimate the optimal selling price.

## Overview

The project uses historical sales and pricing data to:
1. **Analyze demand behavior** in relation to price changes.
2. **Model the demand curve** using different regression techniques.
3. **Calculate profit at various price points**.
4. **Identify the optimal price** that maximizes profit.

## Data

The dataset is loaded from an Excel file (`Base X Semana - V3.xlsx`), specifically from the sheet `00.BASEXSEM`.  
It includes information such as:
- Product prices
- Units sold
- Revenue
- Additional attributes useful for modeling demand.

## Methodology

1. **Exploratory Data Analysis (EDA)**  
   - View and understand the structure of the dataset.
   - Identify relevant features for price-demand modeling.

2. **Model Training**  
   Several models are used to estimate the demand curve:
   - Linear Regression
   - Ridge Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor

3. **Prediction and Optimization**  
   - Predict demand for a range of possible prices.
   - Calculate profit as:
     ```
     Profit = (Price - Cost) * Predicted_Demand
     ```
   - Find the price that yields maximum profit.

4. **Visualization**  
   - Demand curve plots for different models.
   - Profit curve highlighting the optimal price.

## Requirements

The following Python libraries are required:
```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

## How to Run

1. Place the dataset (`Base X Semana - V3.xlsx`) in the appropriate directory.
2. Update the file path in the notebook to match your system.
3. Run all cells to execute the analysis.
4. Review the generated plots and optimal price results.

## Output

- Demand curves for each model.
- Profit curves indicating the maximum profit point.
- Suggested optimal selling price.

---
**Author:** José Luis Valdéz Seda  
**Purpose:** Price optimization using predictive modeling of demand and profit curves.
