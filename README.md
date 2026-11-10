# Hotel Reservation Cancellation Prediction

A classifier that predicts whether a hotel booking will be cancelled, so a hotel chain can plan cleaning staff and room availability.

## Results

XGBoost scores **89.2% accuracy** on 10,883 held-out bookings. A majority-class guess would score 67.3% on the same set, since 7,320 of the 10,883 bookings are not cancelled.

| Metric (test set, 10,883 bookings) | Score |
|---|---|
| Accuracy | 0.8916 |
| Weighted F1 | 0.8918 |
| F1, Canceled (class 0) | 0.8355 |
| F1, Not_Canceled (class 1) | 0.9191 |

5-fold cross-validation accuracy on the training data, for the five models compared:

| Model | CV accuracy |
|---|---|
| XGBoost | 0.9168 |
| KNN | 0.8902 |
| Decision Tree | 0.8835 |
| Random Forest | 0.8741 |
| SVC | 0.8566 |

These CV scores are optimistic. See Limitations.

## Problem

A hotel chain had surprise cancellations. They caused over- or under-staffing of outsourced cleaning crews, paid hours with no work, and empty rooms that could have been sold. The goal was a model that estimates each booking's chance of cancellation.

## Data

- **Source:** [Hotel Reservations Classification Dataset on Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) (`Hotel Reservations.csv`). The CSV is not in this repo.
- **Size:** 36,275 bookings, 19 columns. The target is `booking_status` (Canceled or Not_Canceled).
- **Split:** 70% train, 30% test, `random_state=20`. The test set has 10,883 bookings: 3,563 Canceled and 7,320 Not_Canceled.

## Approach

- Drop `Booking_ID`. Scale the numeric features with `StandardScaler` and one-hot encode the three categorical columns, using a `ColumnTransformer` fitted on the training set only.
- Balance the training classes with SMOTE. The test set is not resampled, so test scores reflect the real class mix.
- Tune five classifiers with `GridSearchCV` (5-fold, accuracy scoring): KNN, SVC, Decision Tree, Random Forest and XGBoost.
- Pick XGBoost for the best cross-validation accuracy at a short refit time (0.34 s on the training set).
- Evaluate once on the test set with a classification report.

## Repository contents

| File | Description |
|------|-------------|
| `Hotel Reservation Prediction.html` | Export of the analysis notebook: code, outputs and written notes |
| `README.md` | This file |

## How to run

This repo has no notebook or script. The only analysis file is the HTML export, which shows the code and outputs but cannot be re-run from this repo.

To read it, open [the rendered version](https://htmlpreview.github.io/?https://github.com/EFaridRahimli/hotel-cancellation-prediction/blob/main/Hotel%20Reservation%20Prediction.html) or download the file and open it in a browser.

To reproduce the work, you would need the Kaggle CSV and the packages pandas, scikit-learn, XGBoost, imbalanced-learn, Matplotlib and Seaborn. The export reads the CSV from the local path `Hotel Reservations.csv`.

## Limitations and next steps

- **SMOTE runs before cross-validation.** Synthetic copies of training rows can end up in both the training and validation folds, so the CV scores are optimistic. The test accuracy of 0.8916 is the more reliable figure.
- **The classification report has its arguments reversed.** The call passes predictions before true labels, which swaps the precision and recall columns. Accuracy and F1 are not affected, so this README reports only those.
- **Next steps:** move SMOTE inside a scikit-learn `Pipeline` so it is fitted per fold, fix the argument order and report per-class precision and recall, and add the notebook and data loading as a script so the results can be re-run.

Coursework for module M5050B.
