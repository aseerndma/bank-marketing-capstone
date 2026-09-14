# Bank Marketing — Term Deposit Prediction

Predicting whether a bank customer will subscribe to a term deposit,
using the UCI Bank Marketing dataset (Portuguese bank, phone campaigns).

## The problem
A bank wants to focus its call campaigns on customers likely to subscribe.
This project builds and evaluates models to predict subscription (yes/no)
from customer and campaign features.

## Key challenge: class imbalance
Only ~11.7% of customers subscribed. A model that always predicts "no"
scores ~88% accuracy while being useless — so this project evaluates on
**F1 and recall**, not accuracy.

## Project structure
- `src/` — pipeline, training script
- `data/` — the dataset (bank-full.csv)
- `configs/` — configuration (model choice, test size)
- `tests/` — data and model tests
- `outputs/` — generated model, report, experiment log
- `.github/workflows/` — CI pipeline

## How to run
1. Create and activate a virtual environment
2. `pip install -r requirements.txt`
3. Train: `python src/train.py --config configs/config.yaml`
4. Test: `python -m pytest`

## Results (test set)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | [fill] | [fill] | [fill] |
| Random Forest | [fill] | [fill] | [fill] |

See `REPORT.md` for full analysis.