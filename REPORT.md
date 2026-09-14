# Project Report: Bank Marketing Term Deposit Prediction

## 1. Problem
[2–3 sentences: what you're predicting and why it matters to the bank.]

## 2. Data
[The dataset: source, ~45,000 rows, mix of numeric and categorical
features, target `y`. Mention that "unknown" appears as a category.]

## 3. The imbalance problem
[Explain the 88/12 split and why accuracy is misleading here. State that
you chose F1 and recall as your main metrics, and why.]

## 4. Preventing data leakage
[Explain that you dropped the `duration` column because it's only known
after the call ends — including it would leak the outcome. This is a key
methodological choice.]

## 5. Approach
[Preprocessing: scaling numeric features, one-hot encoding categoricals
via a ColumnTransformer, all inside a pipeline. Stratified train/test
split. class_weight="balanced" to handle imbalance. Two models compared.]

## 6. Results
[Your metrics table. Then the key insight: the precision/recall trade-off.
Logistic Regression = high recall, low precision (catches most subscribers
but many false alarms). Random Forest = higher precision, lower recall
(fewer, higher-quality leads). Which to choose depends on whether calls
are cheap or expensive.]

## 7. Limitations & next steps
[Honest limitations: no hyperparameter tuning, "unknown" values not
specially handled, single train/test split rather than cross-validation.
Next steps: tune thresholds, try gradient boosting, cross-validate.]