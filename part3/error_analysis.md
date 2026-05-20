# Error Analysis — Churn Prediction Model

**Model:** Gradient Boosting Classifier  
**Test Set:** 336 customers (churn rate = 50.0%)  
**Decision Threshold:** 0.29 (selected by maximising F1 on validation set)  
**Test Set Results:** TP=150 | FP=64 | FN=18 | TN=104

---

## Confusion Matrix

|  | Predicted: Not Churn | Predicted: Churn |
|---|---|---|
| **Actual: Not Churn** | TN = 104 | FP = 64 |
| **Actual: Churn** | FN = 18 | TP = 150 |

**Precision:** 0.701 | **Recall:** 0.893 | **F1:** 0.785 | **ROC-AUC:** 0.862

---

## Why Recall Is Prioritised Over Precision

In a retention context, **missing a churner (FN) is significantly more expensive than incorrectly flagging a loyal customer (FP)**:

- **FN Cost:** A missed churner generates ₹0 in the next cycle. The brand loses the full monetary value of that customer (~₹2,202 average for At Risk segment).
- **FP Cost:** A falsely flagged loyal customer receives an unnecessary retention message. The cost is the campaign budget (₹15–40 per customer) plus a slight annoyance risk.

At threshold=0.29, the model catches **89.3% of all true churners** (recall) while accepting a precision of 70.1% — meaning roughly 3 in 10 flagged customers are false alarms. This is an acceptable trade-off in a low-cost outreach setting.

---

## False Positive Analysis (Model Predicted Churn, Customer Did NOT Churn)

**Total FPs on test set:** 64

### Pattern: Misleading Recency

The most common FP profile is a customer with **high recency (long since last order)** who appears at-risk based on purchase inactivity but actually returned to buy in the 60-day target window. These customers may have been in a "consideration phase" during the snapshot period.

### Specific FP Cases

| customer_id | Pred Prob | Recency | Frequency | Monetary | Sessions | Tickets | Why FP? |
|---|---|---|---|---|---|---|---|
| CUST01370 | 0.97 | 161 days | 2 | ₹1,246 | 2 | 0 | High recency, but was a delayed buyer — purchased post-snapshot |
| CUST01246 | 0.96 | 262 days | 0 | ₹0 | 1 | 0 | No 180d orders, but had prior purchase history and returned in window |
| CUST01017 | 0.96 | 133 days | 2 | ₹1,167 | 3 | 0 | Moderate recency, still web-active, resumed purchasing |
| CUST00437 | 0.96 | 151 days | 1 | ₹729 | 0 | 0 | Inactive-looking profile, but made a purchase in target window |
| CUST01614 | 0.95 | 103 days | 2 | ₹1,352 | 4 | 0 | Borderline recency, good sessions, purchased post-snapshot |
| CUST01405 | 0.94 | 140 days | 1 | ₹1,013 | 2 | 0 | Low frequency but non-zero digital activity — eventual buyer |
| CUST01325 | 0.92 | 186 days | 0 | ₹0 | 1 | 0 | Extremely inactive profile but re-engaged in target window |

**Business Risk of FPs:** These customers will receive an unnecessary retention campaign. This is a low-severity error — a loyalty nudge or free-shipping offer to an already-active customer is unlikely to harm the relationship and may even accelerate their next purchase. The main cost is campaign budget (₹15–40 per customer × 64 = ₹960–₹2,560 of misdirected spend).

**Model Implication:** The model slightly over-relies on recency as a churn signal. Some customers have long purchase cycles (e.g., bulk buyers or seasonal purchasers) that look inactive at snapshot time but are not truly churned. Adding a "purchase cycle" feature (average days between orders) in a future model iteration could reduce FP rate for this cohort.

---

## False Negative Analysis (Model Predicted Not Churn, Customer DID Churn)

**Total FNs on test set:** 18

### Pattern: Deceptively Active at Snapshot

These are customers who appear healthy at snapshot date — recent orders, multiple purchases, active sessions — but then stop buying entirely in the 60-day window. The model's features don't capture the "why" of the upcoming departure.

### Specific FN Cases

| customer_id | Pred Prob | Recency | Frequency | Monetary | Sessions | Tickets | Why FN? |
|---|---|---|---|---|---|---|---|
| CUST00184 | 0.014 | 14 days | 3 | ₹2,456 | 6 | 0 | Very recent buyer, active sessions — model had no signal of impending churn |
| CUST01990 | 0.027 | 59 days | 4 | ₹3,877 | 11 | 0 | High value, highly engaged, 11 sessions — total surprise churn |
| CUST01655 | 0.055 | 13 days | 2 | ₹1,358 | 2 | 0 | Ordered just 13 days before snapshot — model considered safe |
| CUST01253 | 0.066 | 99 days | 2 | ₹2,035 | 13 | 0 | 13 sessions (highest) but stopped buying — severe browse-to-purchase gap |
| CUST01826 | 0.071 | 57 days | 3 | ₹1,929 | 0 | 0 | No sessions but recent order history; offline churn signal |
| CUST00592 | 0.090 | 20 days | 1 | ₹627 | 3 | 0 | New-ish customer, recent purchase, low risk profile — churned anyway |
| CUST00866 | 0.115 | 26 days | 1 | ₹1,280 | 5 | 0 | Recent + browsing; single order may not have met expectations |

**Business Risk of FNs:** These are the highest-risk errors. Each FN is a churned customer who received no retention intervention because the model did not flag them. With avg monetary ~₹1,900 for this group, 18 FNs represent approximately ₹34,200 in undefended revenue in this test cohort alone.

**Root Causes Identified:**
1. **Surprise churners (CUST00184, CUST01990):** These customers gave no behavioural warning. They may have found a competitor or had an unrecorded negative experience (in-store, social, word-of-mouth). The model cannot predict what it cannot observe.
2. **High browser, no buyer (CUST01253, CUST01990):** High session counts that do not convert to purchases are a warning sign that was underweighted in this model iteration. A "session-to-purchase ratio" feature could improve recall for this cohort.
3. **New customers with 1–2 orders (CUST00592, CUST00866):** Very new customers who have only ordered once are hard to classify — their recency looks good but their commitment is not yet established.

**Model Implication:** At the current threshold of 0.29, FN rate is 10.7% (18 of 168 actual churners). Lowering the threshold further (e.g., to 0.20) would capture more of these but at the cost of a significant increase in FPs. The current threshold represents the best F1 balance on the validation set.

---

## Summary of Error Types

| Error Type | Count | Avg Pred Prob | Avg Recency | Avg Monetary | Primary Cause |
|---|---|---|---|---|---|
| False Positives | 64 | 0.78 | 155 days | ₹832 | Long purchase cycle customers look inactive at snapshot |
| False Negatives | 18 | 0.06 | 38 days | ₹1,901 | Deceptively active customers with unobserved departure triggers |

---

## Recommendations for Model Improvement

1. **Add purchase cycle feature:** Customers with naturally long gaps between purchases (30–90 day cycles) should not be penalised as heavily on recency alone.
2. **Add session-to-purchase ratio:** High browse activity with no conversion is a meaningful signal that is currently underweighted.
3. **Segment-aware thresholds:** Apply a lower threshold (more aggressive) for High-Value customers where FN cost is highest, and a higher threshold for low-value segments where FP campaign cost matters more.
4. **Monitor FNs after deployment:** Implement a post-cycle review process — for every confirmed churner who was not flagged, audit their pre-snapshot signals to identify patterns for future model updates.
