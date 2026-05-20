# Model Card — D2C Customer Churn Prediction Model

**Version:** 1.0  
**Created:** September 30, 2025  
**Model File:** `model.pkl`  
**Intended Audience:** Product, Marketing, CRM & Customer Success Teams  

---

## 1. Model Summary

This model predicts whether a customer will churn (make no purchase) in the 60-day window following the snapshot date of 2025-09-30. It is intended to help the retention team prioritise which customers should receive proactive outreach campaigns.

---

## 2. Model Details

| Detail | Value |
|---|---|
| Model Type | Gradient Boosting Classifier (scikit-learn `GradientBoostingClassifier`) |
| Number of Trees | 300 |
| Learning Rate | 0.05 |
| Max Depth | 4 |
| Subsample | 0.8 |
| Decision Threshold | 0.29 (tuned on validation set to maximise F1) |
| Output | Churn probability (0.0–1.0) + binary class (0 or 1) |

**Baseline Comparison:** A Logistic Regression model was trained as a baseline. The Gradient Boosting model achieves higher recall (0.893 vs 0.792) at a lower precision, making it better suited for the retention use case where catching true churners is the priority.

---

## 3. Data Used

### Training Data
- **Source:** `rfm_modeling_snapshot.csv` (pre-built feature table)
- **Snapshot Date:** 2025-09-30
- **Train/Val/Test Split:** 1,728 / 336 / 336 (provided pre-split in `churn_labels.csv`)
- **Target:** `churn_next_60d` — 1 if customer made no purchase from 2025-10-01 to 2025-11-29

### Features Used (25 total)

**RFM & Order Behaviour**
- `recency_days` — days since last pre-snapshot order
- `frequency_180d` — orders in the 180 days before snapshot
- `monetary_180d` — total spend in 180 days (INR)
- `return_rate_180d` — proportion of orders returned
- `avg_discount_pct_180d` — average discount fraction
- `avg_rating_180d` — average order satisfaction rating
- `category_diversity_180d` — distinct product categories purchased

**Support Interactions**
- `ticket_count_90d` — number of support tickets (90 days)
- `negative_ticket_rate_90d` — proportion with negative sentiment
- `avg_resolution_hours_90d` — average time to ticket resolution

**Web/App Activity (30 days)**
- `sessions_30d`, `product_views_30d`, `cart_adds_30d`
- `wishlist_adds_30d`, `abandoned_carts_30d`
- `email_opens_30d`, `campaign_clicks_30d`, `last_visit_days_ago`

**Customer Profile**
- `days_since_signup`, `city_tier`, `age_group`
- `acquisition_channel`, `loyalty_tier`, `preferred_category`, `marketing_consent`

### Leakage Prevention
All features are derived from data on or before 2025-09-30. Post-snapshot order rows (2025-10-01 onwards) in `orders.csv` were never used as features. The target label was constructed separately from the post-snapshot order window.

---

## 4. Performance (Test Set — 336 customers)

| Metric | Logistic Regression (Baseline) | Gradient Boosting (Final) |
|---|---|---|
| Accuracy | 0.819 | 0.756 |
| Precision | 0.837 | 0.701 |
| Recall | 0.792 | **0.893** |
| F1 Score | 0.814 | 0.785 |
| ROC-AUC | 0.885 | 0.862 |
| PR-AUC | 0.878 | 0.837 |
| **False Negatives** | 35 | **18** |
| **False Positives** | 26 | 64 |

**Why we chose Gradient Boosting despite lower overall accuracy:** In a retention setting, False Negatives (missed churners) are far more costly than False Positives (unnecessary retention messages). The GB model reduces FNs from 35 to 18, representing 17 additional at-risk customers correctly identified. The Logistic Regression has higher precision but misses too many true churners to be useful as a retention trigger.

---

## 5. Top Features by Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | `recency_days` | 0.514 |
| 2 | `monetary_180d` | 0.106 |
| 3 | `days_since_signup` | 0.051 |
| 4 | `last_visit_days_ago` | 0.044 |
| 5 | `product_views_30d` | 0.044 |
| 6 | `avg_discount_pct_180d` | 0.028 |
| 7 | `return_rate_180d` | 0.024 |
| 8 | `avg_rating_180d` | 0.020 |
| 9 | `email_opens_30d` | 0.019 |
| 10 | `avg_resolution_hours_90d` | 0.018 |

**Interpretation:** Purchase recency dominates all other signals, accounting for 51% of model decisions. The combination of purchase inactivity and declining digital engagement (last_visit_days_ago, product_views_30d) are the clearest indicators of an impending churner.

---

## 6. Threshold Selection

**Selected threshold: 0.29**

The threshold was chosen by maximising F1 score on the validation set across thresholds 0.25–0.75. At 0.29:
- Precision: 0.70 (30% of flagged customers are false alarms)
- Recall: 0.89 (89% of actual churners are caught)

**Business justification:** The cost of a retention campaign (₹15–40/customer) is far lower than the average customer monetary value (~₹2,000+). The acceptable false alarm rate is high; the acceptable miss rate is low. A threshold of 0.29 reflects this asymmetry — the model errs on the side of caution by flagging borderline customers rather than missing them.

**If precision is prioritised:** Raise threshold to 0.45–0.50. Precision rises to ~0.83, but recall drops to ~0.79, missing approximately 35 churners per 336-customer batch.

---

## 7. Limitations

1. **Recency dominance:** The model is heavily influenced by recency. Customers with naturally long purchase cycles (quarterly buyers) will often be incorrectly flagged.
2. **No competitor signal:** The model cannot observe whether a customer has moved to a competitor. "Surprise churners" (deceptively active at snapshot) represent the majority of false negatives.
3. **Snapshot-based:** The model produces a static score at a single point in time. It does not update as customer behaviour changes mid-window.
4. **Cold start problem:** New customers with fewer than 2 orders have thin feature vectors. Their churn probability estimates are less reliable than for established customers.
5. **Discount sensitivity:** Average discount is a proxy for price sensitivity, but the model cannot distinguish between a customer who churns because they got a bad deal and one who churns despite a good deal.
6. **No causal inference:** A high churn probability score does not explain *why* a customer will churn. The retention team should use segment context (from Part 2) alongside the model score.

---

## 8. Ethical Risks & Responsible Use

### Appropriate Uses
- Prioritising which customers should receive proactive retention outreach
- Sizing campaign budgets based on estimated at-risk population
- Identifying customer cohorts for qualitative research and survey outreach

### Inappropriate Uses
- **Do NOT use scores to deny service.** A high churn probability does not justify reducing service quality, limiting product access, or deprioritising support for that customer.
- **Do NOT use scores as the sole basis for permanent customer decisions.** Scores should be one input among several, not an automatic trigger for drastic action.
- **Do NOT target customers differently based on demographic features.** While `age_group`, `city_tier`, and `acquisition_channel` are included as features, retention actions should be campaign-type-based, not demographic-profile-based.
- **Do NOT assume causality.** High recency alone does not mean a customer is "bad." Some loyal customers simply have longer purchase cycles.

### Fairness Considerations
The model was evaluated on overall metrics but was not audited for differential performance across demographic subgroups (age group, city tier). Before deployment, the team should verify that false negative rates are comparable across customer segments to avoid systematically under-serving any group.

---

## 9. Monitoring & Retraining

### What to Monitor After Deployment
1. **Churn rate actuals vs predictions:** At end of each 60-day window, compare predicted churn rate to actual churn rate. Flag if gap > 5%.
2. **Feature distribution drift:** Monitor the distribution of top features (`recency_days`, `monetary_180d`, `last_visit_days_ago`) monthly. Alert if any feature mean shifts > 20% from training distribution.
3. **Precision and Recall tracking:** Re-evaluate precision and recall on each fresh cohort. If recall drops below 0.80, consider retraining.
4. **False Negative audit:** For every confirmed churner who was not flagged, capture their pre-snapshot features for model improvement analysis.

### When to Retrain
- After a major product catalogue change or pricing shift
- After a significant marketing or seasonality event that alters purchasing patterns
- When model ROC-AUC on a fresh validation cohort drops below 0.80
- At minimum every 6 months with refreshed training data

### Recommended Retraining Cadence
Monthly batch scoring + quarterly model refresh.
