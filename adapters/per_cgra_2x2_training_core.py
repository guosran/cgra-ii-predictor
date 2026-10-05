"""Frozen v1 compact61 training and evaluation routines for the 2x2 model."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any

import torch

from cgra_ii_predictor.mapper_model import (
    DirectMapperIIEnsemble,
    DirectMapperIIModel,
    MAPPER_FEATURE_NAMES,
    MapperModelConfig,
    mapper_feature_names,
)
from cgra_ii_predictor.shape_protocol import SHAPE_PROTOCOL_2X2, SHAPE_PROTOCOL_2X2_ID


SPLIT_SEED = 20261004
DEFAULT_SEEDS = (17, 41, 113, 239)
DEFAULT_EPOCHS = 80
HIDDEN_DIMENSIONS = (64, 32)
MAPPER_SHAPES = SHAPE_PROTOCOL_2X2.mapper_shapes
FEATURE_NAMES_2X2 = mapper_feature_names(SHAPE_PROTOCOL_2X2_ID)
COMPACT_INDICES = tuple(range(65, 79)) + tuple(range(93, 118)) + tuple(range(134, 156))
COMPACT_FEATURE_NAMES = tuple(MAPPER_FEATURE_NAMES[index] for index in COMPACT_INDICES)
if len(COMPACT_FEATURE_NAMES) != 61 or not set(COMPACT_FEATURE_NAMES).issubset(FEATURE_NAMES_2X2):
    raise AssertionError("frozen compact61 feature contract changed")


def _split_groups(groups, seed: int = SPLIT_SEED) -> dict[str, str]:
    import random

    ordered = sorted(set(groups))
    if len(ordered) < 3:
        raise ValueError("at least three unexcluded source groups are required")
    random.Random(seed).shuffle(ordered)
    train_count = max(1, int(len(ordered) * 0.70))
    validation_count = max(1, int(len(ordered) * 0.15))
    if train_count + validation_count >= len(ordered):
        train_count, validation_count = len(ordered) - 2, 1
    assignment = {}
    for group in ordered[:train_count]:
        assignment[group] = "train"
    for group in ordered[train_count:train_count + validation_count]:
        assignment[group] = "validation"
    for group in ordered[train_count + validation_count:]:
        assignment[group] = "test"
    if set(assignment.values()) != {"train", "validation", "test"}:
        raise AssertionError("group split omitted one of its three folds")
    return assignment


def _balanced_group_weights(rows: list[dict[str, Any]]) -> torch.Tensor:
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_group[row["group"]].append(index)
    random_groups = {
        group for group, indices in by_group.items()
        if rows[indices[0]]["group_families"] and all(
            family.startswith("random-dfg/") for family in rows[indices[0]]["group_families"]
        )
    }
    real_groups = set(by_group) - random_groups
    if random_groups and real_groups:
        mass_by_group = {
            **{group: 0.5 / len(real_groups) for group in real_groups},
            **{group: 0.5 / len(random_groups) for group in random_groups},
        }
    else:
        mass_by_group = {group: 1.0 / len(by_group) for group in by_group}
    values = [0.0] * len(rows)
    for group, indices in by_group.items():
        each = mass_by_group[group] / len(indices)
        for index in indices:
            values[index] = each
    return torch.tensor(values, dtype=torch.float32)


def _rank_queries(
    rows: list[dict[str, Any]], weights: torch.Tensor,
) -> list[tuple[list[int], float]]:
    by_query: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_query[(row["group"], row["query"])].append(index)
    complete = []
    expected = set(MAPPER_SHAPES)
    for (group, _query), indices in by_query.items():
        shapes = {rows[index]["shape"] for index in indices}
        if shapes == expected and len(indices) == len(expected):
            if {rows[index]["group"] for index in indices} != {group}:
                raise ValueError("rank query crosses source groups")
            complete.append((indices, float(weights[indices].sum())))
    total = sum(mass for _indices, mass in complete)
    return [(indices, mass / total) for indices, mass in complete] if total else []


def _set_top1_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    logits = -predicted
    best = truth == truth.min()
    return torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[best], dim=0)


def _fit_member(
    training: list[dict[str, Any]], seed: int = 17,
    epochs: int = DEFAULT_EPOCHS,
) -> DirectMapperIIModel:
    raw = torch.tensor([row["full_features"] for row in training], dtype=torch.float32)
    mean = raw.mean(0)
    scale = raw.std(0, unbiased=False)
    scale = torch.where(scale < 1e-5, torch.ones_like(scale), scale)
    config = MapperModelConfig(
        hidden_dimensions=HIDDEN_DIMENSIONS,
        shape_protocol=SHAPE_PROTOCOL_2X2_ID,
        enabled_feature_names=COMPACT_FEATURE_NAMES,
    ).validate()
    torch.manual_seed(seed)
    model = DirectMapperIIModel(config, mean, scale)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    lower = torch.tensor([row["lower_bound"] for row in training], dtype=torch.float32)
    truth = torch.tensor([row["ii"] for row in training], dtype=torch.float32)
    weights = _balanced_group_weights(training)
    queries = _rank_queries(training, weights)
    better: list[int] = []
    worse: list[int] = []
    pair_weights: list[float] = []
    for indices, mass in queries:
        pairs = [(left, right) for left in indices for right in indices if truth[left] < truth[right]]
        if pairs:
            better.extend(left for left, _right in pairs)
            worse.extend(right for _left, right in pairs)
            pair_weights.extend([mass / len(pairs)] * len(pairs))
    pair_weight_tensor = torch.tensor(pair_weights, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        predicted = model(raw, lower)
        point = (torch.nn.functional.smooth_l1_loss(predicted, truth, reduction="none") * weights).sum()
        pair = (
            torch.nn.functional.relu(0.5 - (predicted[worse] - predicted[better])) * pair_weight_tensor
        ).sum() if pair_weights else predicted.sum() * 0.0
        top1 = sum(
            _set_top1_loss(predicted[indices], truth[indices]) * mass
            for indices, mass in queries
        ) if queries else predicted.sum() * 0.0
        loss = point + 0.1 * pair + 0.3 * top1
        if not torch.isfinite(loss):
            raise ValueError("training loss became non-finite")
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict(model: torch.nn.Module, rows: list[dict[str, Any]]) -> list[float]:
    if not rows:
        return []
    features = torch.tensor([row["full_features"] for row in rows], dtype=torch.float32)
    lower = torch.tensor([row["lower_bound"] for row in rows], dtype=torch.float32)
    torch.set_num_threads(1)
    with torch.inference_mode():
        return [float(value) for value in model(features, lower).tolist()]


def _score(rows: list[dict[str, Any]], predictions: list[float]) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError("evaluation prediction count changed")
    if not rows:
        return {
            "row_count": 0, "group_count": 0, "row_mae": None,
            "rounded_exact_ii": None, "rounded_within_one_ii": None,
            "complete_query_count": 0, "partial_query_count": 0,
            "shape_hit": None, "mean_shape_regret": None,
        }
    by_query: dict[tuple[str, str], list[tuple[dict[str, Any], float]]] = defaultdict(list)
    absolute_errors, exact, within_one = [], [], []
    shape_order = {shape: index for index, shape in enumerate(MAPPER_SHAPES)}
    for row, prediction in zip(rows, predictions):
        if not math.isfinite(prediction):
            raise ValueError("model produced a non-finite evaluation prediction")
        absolute_errors.append(abs(prediction - row["ii"]))
        rounded = int(round(prediction))
        exact.append(rounded == int(row["ii"]))
        within_one.append(abs(rounded - int(row["ii"])) <= 1)
        by_query[(row["group"], row["query"])].append((row, prediction))
    complete = []
    partial_count = 0
    for items in by_query.values():
        if {row["shape"] for row, _prediction in items} != set(MAPPER_SHAPES):
            partial_count += 1
            continue
        selected_row, _prediction = min(
            items, key=lambda item: (item[1], shape_order[item[0]["shape"]]),
        )
        oracle = min(row["ii"] for row, _value in items)
        complete.append({"hit": selected_row["ii"] == oracle, "regret": selected_row["ii"] - oracle})
    return {
        "row_count": len(rows),
        "group_count": len({row["group"] for row in rows}),
        "row_mae": statistics.mean(absolute_errors),
        "rounded_exact_ii": statistics.mean(exact),
        "rounded_within_one_ii": statistics.mean(within_one),
        "complete_query_count": len(complete),
        "partial_query_count": partial_count,
        "shape_hit": statistics.mean(row["hit"] for row in complete) if complete else None,
        "mean_shape_regret": statistics.mean(row["regret"] for row in complete) if complete else None,
    }


def ensemble_from_members(members: list[DirectMapperIIModel]) -> DirectMapperIIEnsemble:
    if len(members) != len(DEFAULT_SEEDS):
        raise ValueError("original recipe requires exactly four ensemble members")
    model = DirectMapperIIEnsemble(len(members), members[0].config)
    for destination, source in zip(model.members, members):
        destination.load_state_dict(source.state_dict(), strict=True)
    model.eval()
    return model
