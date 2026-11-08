# Hotel Reservation Cancellation Prediction

Predicts whether a hotel booking will be **cancelled**, so a hotel chain can plan staffing (outsourced cleaning crews) and avoid both empty rooms and overbooking.

## Results
**XGBoost** was selected after comparing five tuned classifiers on accuracy *and* refit time.

| Metric (test set, 10,883 bookings) | Score |
|---|---|
| Accuracy | **89.2%** |
| Weighted F1 | 0.89 |
| F1 (class 0 / class 1) | 0.84 / 0.92 |

## Approach
1. Exploration and cleaning of the Hotel Reservations dataset (lead time, room type, price, special requests, market segment, …).
2. Encoding and class balancing with **SMOTE**.
3. Hyper-parameter search (`GridSearchCV`, 5-fold CV) across five classifiers, including KNN, Decision Tree, Random Forest and XGBoost.
4. Model choice by accuracy-vs-refit-time trade-off, then final evaluation with a classification report.

## Tech stack
Python · pandas · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

## Repository contents
| File | Description |
|------|-------------|
| `Hotel Reservation Prediction.html` | Full rendered notebook: code, outputs and discussion |

Download the HTML file and open it in a browser to read the full analysis.
