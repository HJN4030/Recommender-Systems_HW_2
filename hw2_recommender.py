#!/usr/bin/env python3
"""HW2 recommenders: dual-form EASE and NeuMF.

The script supports a chronological per-user validation split and creates a
Kaggle-compatible top-20 submission.  It intentionally uses only NumPy and
PyTorch so that the experiment is easy to reproduce without a recommender
framework.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Interaction:
    user: int
    item: int
    rating: float
    time_idx: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_interactions(path: Path) -> List[Interaction]:
    rows: List[Interaction] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            rows.append(
                Interaction(
                    user=int(row["userId"]),
                    item=int(row["movieId"]),
                    rating=float(row["rating"]),
                    time_idx=int(row["time_idx"]),
                )
            )
    if not rows:
        raise ValueError(f"No interactions found in {path}")
    return rows


def read_test_users(path: Path) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            rows.append((int(row["id"]), int(row["userId"])))
    return rows


def chronological_split(
    rows: Sequence[Interaction], validation_ratio: float
) -> Tuple[List[Interaction], Dict[int, Set[int]]]:
    """Keep the most recent validation_ratio interactions for each user."""
    by_user: Dict[int, List[Interaction]] = defaultdict(list)
    for row in rows:
        by_user[row.user].append(row)

    train: List[Interaction] = []
    validation: Dict[int, Set[int]] = defaultdict(set)
    for user, events in by_user.items():
        events.sort(key=lambda event: event.time_idx)
        n_validation = max(1, int(math.ceil(len(events) * validation_ratio)))
        if n_validation >= len(events):
            n_validation = 1
        train.extend(events[:-n_validation])
        validation[user].update(event.item for event in events[-n_validation:])
    return train, validation


class IdMap:
    def __init__(self, rows: Sequence[Interaction]):
        self.users = sorted({row.user for row in rows})
        self.items = sorted({row.item for row in rows})
        self.user_to_idx = {value: idx for idx, value in enumerate(self.users)}
        self.item_to_idx = {value: idx for idx, value in enumerate(self.items)}


def build_dense_matrix(
    rows: Sequence[Interaction], id_map: IdMap, weight_mode: str
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(id_map.users), len(id_map.items)), dtype=np.float64)
    seen = np.zeros(values.shape, dtype=bool)
    for row in rows:
        user_idx = id_map.user_to_idx[row.user]
        item_idx = id_map.item_to_idx[row.item]
        if weight_mode == "binary":
            value = 1.0
        elif weight_mode == "rating":
            value = row.rating / 5.0
        elif weight_mode == "confidence":
            value = 1.0 + 0.25 * (row.rating - 3.0)
        else:
            raise ValueError(f"Unknown weight mode: {weight_mode}")
        values[user_idx, item_idx] = value
        seen[user_idx, item_idx] = True
    return values, seen


def top_k_indices(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    if scores.shape != seen.shape:
        raise ValueError("scores and seen must have the same shape")
    if k > scores.shape[1]:
        raise ValueError("k exceeds the number of candidate items")
    masked = scores.copy()
    masked[seen] = -np.inf
    candidates = np.argpartition(masked, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(masked, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    return np.take_along_axis(candidates, order, axis=1)


def ranking_metrics(
    top_items: np.ndarray,
    users: Sequence[int],
    ground_truth: Dict[int, Set[int]],
    id_map: IdMap,
    k: int,
) -> Dict[str, float]:
    ndcgs: List[float] = []
    recalls: List[float] = []
    for row_idx, user in enumerate(users):
        relevant = {
            id_map.item_to_idx[item]
            for item in ground_truth.get(user, set())
            if item in id_map.item_to_idx
        }
        if not relevant:
            continue
        ranked = top_items[row_idx, :k]
        dcg = sum(
            1.0 / math.log2(rank + 2.0)
            for rank, item_idx in enumerate(ranked)
            if int(item_idx) in relevant
        )
        ideal = sum(
            1.0 / math.log2(rank + 2.0)
            for rank in range(min(k, len(relevant)))
        )
        hits = sum(int(item_idx) in relevant for item_idx in ranked)
        ndcgs.append(dcg / ideal)
        recalls.append(hits / len(relevant))
    if not ndcgs:
        raise ValueError("No users had validation items in the item vocabulary")
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        "evaluated_users": len(ndcgs),
    }


class DualEASE:
    """EASE computed in user space with the Woodbury identity.

    The ordinary EASE solution requires the inverse of an item-by-item matrix.
    Here the same closed-form prediction is calculated from a user-by-user
    system, which is much smaller for this data set (610 users vs. 8,237 items).
    """

    def __init__(self, regularization: float):
        self.regularization = regularization
        self.x: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.denominator: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "DualEASE":
        self.x = np.asarray(x, dtype=np.float64)
        # Some Accelerate-backed NumPy builds emit spurious floating-point
        # warnings for finite matrix products.  The explicit finite checks
        # below still fail fast if the data or result is genuinely invalid.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = self.x @ self.x.T
        if not np.isfinite(gram).all():
            raise FloatingPointError("Non-finite EASE Gram matrix")
        system = gram + self.regularization * np.eye(gram.shape[0], dtype=gram.dtype)
        self.z = np.linalg.solve(system, self.x)
        denominator = 1.0 - np.sum(self.x * self.z, axis=0)
        self.denominator = np.maximum(denominator, 1e-12)
        return self

    def predict(self, query: np.ndarray) -> np.ndarray:
        if self.x is None or self.z is None or self.denominator is None:
            raise RuntimeError("fit must be called before predict")
        query = np.asarray(query, dtype=np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            projected = (query @ self.x.T) @ self.z
        if not np.isfinite(projected).all():
            raise FloatingPointError("Non-finite EASE projection")
        x_times_p_without_lambda = query - projected
        return query - x_times_p_without_lambda / self.denominator


class NeuMF(nn.Module):
    """Neural Matrix Factorization with GMF and MLP branches."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_dim: int = 16,
        mlp_dim: int = 32,
        mlp_layers: Sequence[int] = (64, 32, 16),
    ):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, gmf_dim)
        self.gmf_item = nn.Embedding(n_items, gmf_dim)
        self.mlp_user = nn.Embedding(n_users, mlp_dim)
        self.mlp_item = nn.Embedding(n_items, mlp_dim)

        layers: List[nn.Module] = []
        input_dim = 2 * mlp_dim
        for output_dim in mlp_layers:
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
            input_dim = output_dim
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(gmf_dim + input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in [self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item]:
            nn.init.normal_(embedding.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp_input = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)
        mlp = self.mlp(mlp_input)
        return self.output(torch.cat([gmf, mlp], dim=1)).squeeze(1)


def indexed_positives(
    rows: Sequence[Interaction], id_map: IdMap
) -> Tuple[np.ndarray, np.ndarray]:
    users = np.fromiter(
        (id_map.user_to_idx[row.user] for row in rows), dtype=np.int64, count=len(rows)
    )
    items = np.fromiter(
        (id_map.item_to_idx[row.item] for row in rows), dtype=np.int64, count=len(rows)
    )
    return users, items


def sample_training_epoch(
    positive_users: np.ndarray,
    positive_items: np.ndarray,
    known: np.ndarray,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    negative_users = np.repeat(positive_users, negatives_per_positive)
    negative_items = rng.integers(0, known.shape[1], size=len(negative_users), dtype=np.int64)
    invalid = known[negative_users, negative_items]
    while np.any(invalid):
        negative_items[invalid] = rng.integers(
            0, known.shape[1], size=int(np.sum(invalid)), dtype=np.int64
        )
        invalid = known[negative_users, negative_items]

    users = np.concatenate([positive_users, negative_users])
    items = np.concatenate([positive_items, negative_items])
    labels = np.concatenate(
        [
            np.ones(len(positive_users), dtype=np.float32),
            np.zeros(len(negative_users), dtype=np.float32),
        ]
    )
    order = rng.permutation(len(users))
    return (
        torch.from_numpy(users[order]),
        torch.from_numpy(items[order]),
        torch.from_numpy(labels[order]),
    )


def fit_neumf(
    rows: Sequence[Interaction],
    id_map: IdMap,
    seen: np.ndarray,
    epochs: int,
    batch_size: int,
    negatives_per_positive: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> Tuple[NeuMF, List[float]]:
    model = NeuMF(len(id_map.users), len(id_map.items))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    positive_users, positive_items = indexed_positives(rows, id_map)
    rng = np.random.default_rng(seed)
    losses: List[float] = []

    for epoch in range(1, epochs + 1):
        epoch_users, epoch_items, epoch_labels = sample_training_epoch(
            positive_users,
            positive_items,
            seen,
            negatives_per_positive,
            rng,
        )
        loader = DataLoader(
            TensorDataset(epoch_users, epoch_items, epoch_labels),
            batch_size=batch_size,
            shuffle=False,
        )
        model.train()
        total_loss = 0.0
        total_count = 0
        for users, items, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(users, items)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_count += len(labels)
        mean_loss = total_loss / total_count
        losses.append(mean_loss)
        print(f"epoch={epoch}/{epochs} loss={mean_loss:.6f}", flush=True)
    return model, losses


@torch.no_grad()
def predict_neumf(
    model: NeuMF,
    user_indices: Sequence[int],
    n_items: int,
    user_batch_size: int = 8,
    pair_batch_size: int = 65536,
) -> np.ndarray:
    model.eval()
    result = np.empty((len(user_indices), n_items), dtype=np.float32)
    all_items = torch.arange(n_items, dtype=torch.long)
    for begin in range(0, len(user_indices), user_batch_size):
        batch_users = np.asarray(user_indices[begin : begin + user_batch_size], dtype=np.int64)
        users = torch.from_numpy(batch_users).repeat_interleave(n_items)
        items = all_items.repeat(len(batch_users))
        values: List[torch.Tensor] = []
        for pair_begin in range(0, len(users), pair_batch_size):
            pair_end = pair_begin + pair_batch_size
            values.append(model(users[pair_begin:pair_end], items[pair_begin:pair_end]).cpu())
        logits = torch.cat(values).reshape(len(batch_users), n_items)
        result[begin : begin + len(batch_users)] = logits.numpy()
        print(
            f"scored_users={min(begin + len(batch_users), len(user_indices))}/{len(user_indices)}",
            flush=True,
        )
    return result


def popularity_scores(rows: Sequence[Interaction], id_map: IdMap) -> np.ndarray:
    counts = np.zeros(len(id_map.items), dtype=np.float64)
    for row in rows:
        counts[id_map.item_to_idx[row.item]] += 1.0
    return counts


def write_submission(
    path: Path,
    test_rows: Sequence[Tuple[int, int]],
    recommendations: Dict[int, Sequence[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "recommendations"])
        for row_id, user in test_rows:
            writer.writerow([row_id, " ".join(str(item) for item in recommendations[user])])


def save_checkpoint(path: Path, model: NeuMF, id_map: IdMap, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "users": id_map.users,
            "items": id_map.items,
            "config": vars(args),
        },
        path,
    )


def run_validation(args: argparse.Namespace, rows: List[Interaction]) -> None:
    train_rows, ground_truth = chronological_split(rows, args.validation_ratio)
    id_map = IdMap(rows)
    train_values, train_seen = build_dense_matrix(train_rows, id_map, args.ease_weights)
    start = time.perf_counter()

    if args.model == "ease":
        model = DualEASE(args.ease_regularization).fit(train_values)
        scores = model.predict(train_values)
        losses: List[float] = []
    else:
        model, losses = fit_neumf(
            train_rows,
            id_map,
            train_seen,
            args.epochs,
            args.batch_size,
            args.negatives,
            args.learning_rate,
            args.weight_decay,
            args.seed,
        )
        scores = predict_neumf(model, range(len(id_map.users)), len(id_map.items))

    ranked = top_k_indices(scores, train_seen, args.top_k)
    metrics = ranking_metrics(ranked, id_map.users, ground_truth, id_map, args.top_k)
    metrics.update(
        {
            "model": args.model,
            "train_interactions": len(train_rows),
            "validation_interactions": len(rows) - len(train_rows),
            "users": len(id_map.users),
            "items": len(id_map.items),
            "elapsed_seconds": time.perf_counter() - start,
        }
    )
    if losses:
        metrics["final_training_loss"] = losses[-1]
    print("METRICS " + json.dumps(metrics, sort_keys=True), flush=True)


def run_submission(args: argparse.Namespace, rows: List[Interaction]) -> None:
    if args.test is None or args.output is None:
        raise ValueError("--test and --output are required in submit mode")
    test_rows = read_test_users(args.test)
    id_map = IdMap(rows)
    train_values, train_seen = build_dense_matrix(rows, id_map, args.ease_weights)
    start = time.perf_counter()

    known_test_users = sorted({user for _, user in test_rows if user in id_map.user_to_idx})
    known_indices = [id_map.user_to_idx[user] for user in known_test_users]
    if args.model == "ease":
        model = DualEASE(args.ease_regularization).fit(train_values)
        known_scores = model.predict(train_values[known_indices])
    else:
        model, _ = fit_neumf(
            rows,
            id_map,
            train_seen,
            args.epochs,
            args.batch_size,
            args.negatives,
            args.learning_rate,
            args.weight_decay,
            args.seed,
        )
        known_scores = predict_neumf(model, known_indices, len(id_map.items))
        if args.checkpoint is not None:
            save_checkpoint(args.checkpoint, model, id_map, args)

    known_seen = train_seen[known_indices]
    known_ranked = top_k_indices(known_scores, known_seen, args.top_k)
    recommendations: Dict[int, Sequence[int]] = {}
    for row_idx, user in enumerate(known_test_users):
        recommendations[user] = [id_map.items[item_idx] for item_idx in known_ranked[row_idx]]

    popular = np.argsort(popularity_scores(rows, id_map))[::-1]
    for _, user in test_rows:
        if user not in recommendations:
            recommendations[user] = [id_map.items[item_idx] for item_idx in popular[: args.top_k]]

    write_submission(args.output, test_rows, recommendations)
    summary = {
        "model": args.model,
        "output": str(args.output),
        "rows": len(test_rows),
        "top_k": args.top_k,
        "elapsed_seconds": time.perf_counter() - start,
    }
    print("SUBMISSION " + json.dumps(summary, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["validate", "submit"], required=True)
    parser.add_argument("--model", choices=["ease", "neumf"], required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ease-regularization", type=float, default=100.0)
    parser.add_argument(
        "--ease-weights", choices=["binary", "rating", "confidence"], default="rating"
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rows = read_interactions(args.train)
    if args.mode == "validate":
        run_validation(args, rows)
    else:
        run_submission(args, rows)


if __name__ == "__main__":
    main()
