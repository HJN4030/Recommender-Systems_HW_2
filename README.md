# HW2 Recommender System

Two models trained on `train.csv`: EASE (Q1) and NeuMF (Q2).

- **EASE**: input values are rating/5. Instead of inverting the 8,237 x 8,237 item matrix, I solve a 610 x 610 user-side system with the Woodbury identity.
- **NeuMF**: PyTorch, separate GMF and MLP embeddings, 4 negatives per positive, loss is `BCEWithLogitsLoss`.

## Validation

Each user's interactions are sorted by `time_idx`. The latest 20% is the validation set and the older 80% is used for training. Items seen in training are removed from the recommendations, and the metrics are computed over all items (no sampled candidates).

Results with seed 42:

| Model | Settings | NDCG@20 | Recall@20 |
|---|---|---:|---:|
| EASE | rating/5, lambda=100 | 0.1622 | 0.1639 |
| NeuMF | 15 epochs, 4 negatives, Adam 1e-3 | 0.1244 | 0.1233 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run validation

```bash
python hw2_recommender.py --mode validate --model ease \
  --train train.csv --ease-regularization 100 --ease-weights rating

python hw2_recommender.py --mode validate --model neumf \
  --train train.csv --epochs 15 --negatives 4 --seed 42
```

## Make submission files

```bash
python hw2_recommender.py --mode submit --model ease \
  --train train.csv --test test.csv --output submission_ease.csv

python hw2_recommender.py --mode submit --model neumf \
  --train train.csv --test test.csv --output submission_neumf.csv \
  --checkpoint neumf.pt --epochs 15 --seed 42
```

Each row has 20 space-separated `movieId`s, none of which the user has already seen.
