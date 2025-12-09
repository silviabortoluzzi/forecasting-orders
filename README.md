# forecasting-orders
Machine learning solution for forecasting raw material deliveries for Hydro, as part of the TDT4173 course project at NTNU. Focus on conservative forecasts using asymmetric quantile loss.

# Forecasting Orders — Machine Learning Project (TDT4173)

This repository contains the implementation of a **machine learning forecasting solution** developed for the course **TDT4173 – Modern Machine Learning in Practice (NTNU)**.

The goal of the project is to predict the **cumulative delivered weight** of raw materials over a future period, using a conservative forecasting approach aligned with the evaluation metric based on **Quantile Loss (α = 0.2)**.  
The full methodology, experiments, and results are documented in **Report.pdf**.

---

## Project Overview

The task consists of forecasting the total received weight for each raw material (`rm_id`) between:

- **Start of forecast:** 2025-01-01  
- **End date:** any date between **2025-01-01 and 2025-05-31**

Forecasts must be *conservative*, meaning that **overestimation is penalized more heavily** than underestimation.  
The official metric follows the asymmetric Pinball Loss defined in the task description.

Two models were implemented and compared:

- **Prophet** : per-material cumulative forecasting  
- **XGBoost** : feature-based regression with extensive engineered features  

Both approaches generate final prediction files stored in the `submissions/` directory.

For the complete explanation of model design, preprocessing pipeline, EDA, and results, refer to **Report.pdf**.

---

## Repository Structure


```text
FORECASTING-ORDERS/
│
├── data/
│   ├── kernel/
│   │   ├── receivals.csv
│   │   ├── purchase_orders.csv
│   │   └── prediction_mapping.csv
│   └── extended/
│       ├── materials.csv
│       └── transportation.csv
│
├── models/
│   ├── prophet.ipynb
│   └── xgboost.ipynb
│
├── submissions/
│   ├── final_submission_prophet.csv
│   └── final_submission_xgboost.csv
│
├── README.md
└── Report.pdf
```

### Dataset Documentation  
A detailed description of each dataset (kernel + extended) is available in the dataset explanation file included in the project.

---

## Model Approaches (Summary)

### **Prophet Model**
- Trained separately for each **active** raw material (`rm_id`)
- Uses **cumulative weights** instead of daily weights to reduce sparsity
- Captures:
  - weekly seasonality  
  - Norwegian holidays  
  - working-day regressors  
- Applies monotonic post-processing to ensure cumulative forecasts never decrease

This model achieved the **best performance** for the task.

---

### **XGBoost Model**
- Feature-rich regression pipeline with:
  - temporal window statistics (7/30/90 days)
  - rolling means and trends
  - supplier reliability information
  - Croston intermittent demand baseline
  - month/week cyclic encodings
  - purchase order–based features
- Hyperparameters tuned with Optuna  
- Competitive performance, though slightly below Prophet on final leaderboard

All methodological details and visualizations are available in **Report.pdf**.

---

## Final Submissions

The directory `submissions/` contains the final CSV outputs for the task:

- `final_submission_prophet.csv`
- `final_submission_xgboost.csv`

Each file follows the required structure specified in `prediction_mapping.csv`.

---

## How to Run the Notebooks

1. Ensure the dataset is placed in the `data/` directory using the structure above.
2. Open one of the notebooks in `models/`:
   - `prophet.ipynb`
   - `xgboost.ipynb`
3. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn prophet xgboost optuna
```

## Full Report

All analysis — including EDA, feature engineering, model selection, hyperparameter tuning, evaluation, and discussion — is documented in Report.pdf.
Readers interested in the technical reasoning behind each choice should refer to that document.


---
