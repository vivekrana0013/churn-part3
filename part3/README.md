# Part 3 — Churn Prediction Model & Model Card

## D2C Customer Churn Intelligence Capstone

This repository contains the full churn prediction pipeline: feature preparation, baseline + final model training, evaluation, error analysis, and a model card.

---

## Repository Structure

```
part3/
├── churn_model.ipynb      # Full modeling notebook
├── model.pkl              # Saved Gradient Boosting model + preprocessing artifacts
├── metrics.json           # Test set metrics for baseline and final model
├── error_analysis.md      # FP/FN analysis with 14 specific customer examples
├── model_card.md          # Structured model card for stakeholder review
├── requirements.txt
└── README.md
```

---

## Dataset Setup

Place dataset CSVs in `../data/` relative to this repo:
```
data/
├── rfm_modeling_snapshot.csv   # primary modeling input
├── churn_labels.csv            # target variable + splits
└── customers.csv, orders.csv (for reference)
```

---

## Setup & Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook churn_model.ipynb
```

Run all cells in order. `model.pkl` and `metrics.json` will be saved to the working directory.

---

## Loading the Model

```python
import pickle
with open('model.pkl', 'rb') as f:
    artifact = pickle.load(f)

model       = artifact['model']         # GradientBoostingClassifier
le_dict     = artifact['le_dict']       # LabelEncoders per categorical column
feature_cols= artifact['feature_cols']  # List of 25 feature names
threshold   = artifact['threshold']     # 0.29
cat_cols    = artifact['cat_cols']      # Categorical columns to encode

# Predict
proba = model.predict_proba(X)[:, 1]
preds = (proba >= threshold).astype(int)
```

---

## Model Performance (Test Set — 336 customers)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.837 | 0.792 | 0.814 | 0.885 |
| **Gradient Boosting (final)** | **0.701** | **0.893** | **0.785** | **0.862** |

**Why Gradient Boosting with threshold=0.29:** Recall is prioritised. Missing a churner (~₹2,000 lost revenue) costs far more than an unnecessary retention message (~₹30). The model catches 89.3% of true churners.

**Top Feature:** `recency_days` accounts for 51.4% of model importance.

---

## Key Files

| File | Description |
|---|---|
| `model.pkl` | Trained GradientBoostingClassifier + LabelEncoders + threshold |
| `metrics.json` | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| `error_analysis.md` | 14 customer-level FP/FN examples with business interpretation |
| `model_card.md` | Intended use, limitations, ethical risks, monitoring plan |
