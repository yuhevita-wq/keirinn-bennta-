#!/usr/bin/env python3
"""Independent first-win probability for F1 S-class using adopted frozen V2 weights.

Derived from the exact feature definitions in the pinned source repository:
research/benter_fundamental_v1/train_v1.py,
research/benter_structural_discovery_v2/discover_v2.py,
research/benter_walkforward_v7/freeze_prior_win_term_unscaled.py.

No odds, finishing results or future outcomes may enter this inference.
This is NOT a live-data acquisition tool.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path

BASE_FEATURES = (
    "score", "age", "term", "s_count", "b_count", "nige_count",
    "makuri_count", "sashi_count", "mark_count", "first_count",
    "second_count", "third_count", "outside_count", "win_rate",
    "top2_rate", "top3_rate", "gear",
)
STRUCT_FEATURES = (
    "leader", "second", "third_or_later", "solo", "line_size",
    "leader_b", "follower_mark", "lead_follow_synergy",
    "line_leader_first", "line_follow_first",
    "line_leader_pressure_share", "rival_front_b",
    "lead_attack_versatility", "line_mate_count",
    "line_head_vs_follower_win_gap", "own_role_front_b",
    "solo_front_pressure", "teammate_score_support",
    "line_score_imbalance",
)
ALL_FEATURES = BASE_FEATURES + STRUCT_FEATURES + ("term_999_foreign_category",)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "first_win_2026sep_manifest.json"
THETA = 0.6412587661746468
LINE_RE = re.compile(r"^[1-9](?:-[1-9])*$")
FORBIDDEN_RACE_KEYS = {"finish_position", "result", "results", "odds", "payout"}
FORBIDDEN_RIDER_KEYS = {"finish_position", "result", "odds", "payout", "observed_win"}


def number(value):
    """Faithful to v1.as_number: commas and percent symbols are stripped."""
    if value is None:
        return None
    try:
        f = float(str(value).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None
    return f if math.isfinite(f) else None


def require_scope(card):
    if FORBIDDEN_RACE_KEYS.intersection(card):
        raise ValueError("Results and market odds must not be inference inputs")
    grade = unicodedata.normalize("NFKC", str(card.get("meeting_grade", "")).strip()).upper()
    race_type = unicodedata.normalize("NFKC", str(card.get("race_type", "")).strip()).upper()
    if grade != "F1" or not race_type.startswith("S級"):
        raise ValueError("Research universe locked to F1・S級")
    riders = card.get("riders")
    if not isinstance(riders, list) or not 5 <= len(riders) <= 9:
        raise ValueError("Require 5-9 actual entrants")
    ids = []
    for rider in riders:
        if not isinstance(rider, dict) or FORBIDDEN_RIDER_KEYS.intersection(rider):
            raise ValueError("Rider fields invalid or contain outcome/odds")
        car = rider.get("car_no")
        if isinstance(car, bool):
            raise ValueError("Invalid car number")
        try:
            car = int(car)
        except (ValueError, TypeError):
            raise ValueError("Invalid car number") from None
        if not 1 <= car <= 9:
            raise ValueError("Car number must be 1-9")
        ids.append(car)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate car numbers")
    return sorted(zip(ids, riders), key=lambda x: x[0])


def parse_lines(formation, cars):
    text = str(formation or "").strip()
    groups = text.split("/")
    if not text or any(not LINE_RE.fullmatch(g.strip()) for g in groups):
        raise ValueError("Invalid or missing predicted line formation")
    lines = [[int(n) for n in g.strip().split("-")] for g in groups]
    flat = [n for line in lines for n in line]
    if len(set(flat)) != len(flat) or set(flat) != set(cars):
        raise ValueError("Line formation must contain each entrant exactly once")
    return lines


def build_features(ordered_riders, lines):
    raw = {}
    foreign = {}
    for car, rider in ordered_riders:
        vals = {key: number(rider.get(key)) for key in BASE_FEATURES}
        foreign[car] = float(vals["term"] == 999.0)
        if foreign[car]:
            vals["term"] = None  # Special category, not numerical ordinal 999.
        raw[car] = vals

    def val(car, field):
        v = raw[car][field]
        return float(v) if v is not None else 0.0

    def l1(car, field):
        return math.log1p(max(val(car, field), 0.0))

    heads = [group[0] for group in lines]
    extra = {}
    for group in lines:
        leader = group[0]
        front_b = l1(leader, "b_count")
        mark = l1(group[1], "mark_count") if len(group) >= 2 else 0.0
        first = l1(leader, "first_count")
        follower_first = l1(group[1], "first_count") if len(group) >= 2 else 0.0
        competitors = [h for h in heads if h != leader]
        pressure = sum(max(val(h, "b_count"), 0.0) for h in competitors)
        pressure_max = max((max(val(h, "b_count"), 0.0) for h in competitors), default=0.0)
        all_front = sum(max(val(h, "b_count"), 0.0) for h in heads)
        front_share = max(val(leader, "b_count"), 0.0) / (1.0 + all_front)
        leader_score = val(leader, "score")
        followers = [val(c, "score") for c in group[1:]]
        teammate_score = sum(followers) / len(followers) if followers else 0.0
        head_vs_fol = first - follower_first
        for pos, car in enumerate(group):
            lead_b = front_b if pos >= 1 else 0.0
            support_mark = mark if pos == 0 else 0.0
            synergy = math.sqrt(max(0.0, front_b * mark)) if pos <= 1 else 0.0
            own_front = math.log1p(max(val(car, "b_count"), 0.0)) if pos == 0 else 0.0
            teammate_strength = teammate_score if pos == 0 else leader_score
            extra[car] = [
                float(pos == 0), float(pos == 1),
                float(pos >= 2), float(len(group) == 1),
                float(len(group)), lead_b, support_mark, synergy,
                first if pos >= 1 else 0.0,
                follower_first if pos == 0 else 0.0,
                front_share if pos == 0 else 0.0,
                math.log1p(pressure_max) if pos == 0 else 0.0,
                math.log1p(max(val(leader, "nige_count"), 0.0) +
                           max(val(leader, "makuri_count"), 0.0)) if pos == 0 else 0.0,
                float(len(group) - 1),
                head_vs_fol if pos <= 1 else 0.0,
                own_front,
                float(len(group) == 1) * math.log1p(pressure),
                teammate_strength,
                abs(teammate_strength - val(car, "score")),
            ]
    return {
        car: [raw[car][key] for key in BASE_FEATURES] +
             extra[car] + [foreign[car]]
        for car, _ in ordered_riders
    }


def softmax(logits):
    maximum = max(logits)
    exponentials = [math.exp(z - maximum) for z in logits]
    denom = sum(exponentials)
    return [x / denom for x in exponentials]


def model_probabilities(ordered_riders, feature_map, manifest):
    if tuple(manifest["feature_names"]) != ALL_FEATURES:
        raise ValueError("Manifest feature order does not match adopted V2")
    trans = manifest["model"]["transform"]
    median, mean, std = (trans[k] for k in ("median", "mean", "std"))
    coefficients = manifest["model"]["coefficients"]
    n = len(ALL_FEATURES)
    if any(len(a) != n for a in (median, mean, std)) or len(coefficients) != 2 * n:
        raise ValueError("Frozen model dimensions invalid")
    if mean[-1] != 0 or std[-1] != 1:
        raise ValueError("Foreign category not safely encoded as 0/1")

    scores = []
    for car, _ in ordered_riders:
        values = feature_map[car]
        missing = [float(v is None) for v in values]
        filled = [median[j] if v is None else v for j, v in enumerate(values)]
        standardized = [(filled[j] - mean[j]) / std[j] for j in range(n)]
        score = sum(standardized[j] * coefficients[j] for j in range(n))
        score += sum(missing[j] * coefficients[n+j] for j in range(n))
        scores.append(score)
    return softmax(scores)


def predict(card, manifest):
    ordered = require_scope(card)
    cars = [c for c, _ in ordered]
    lines = parse_lines(card.get("predicted_line_formation"), cars)
    features = build_features(ordered, lines)
    p = model_probabilities(ordered, features, manifest)
    # Adopted development-period high-probability recalibration.
    z = [math.log(x) - THETA * max(x - 0.4, 0.0) for x in p]
    q = softmax(z)
    assert abs(sum(q) - 1.0) < 1e-12
    return {
        "scope": "F1 S級",
        "source_model_period": manifest["period"],
        "source_model_last_training_date": manifest["train_max_date"],
        "note": "Frozen independent-win model, not a market-bet expected value",
        "predictions": [
            {
                "car_no": car,
                "raw_independent_win_probability": round(float(x), 12),
                "calibrated_first_place_probability": round(float(y), 12),
            }
            for car, x, y in zip(cars, p, q)
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="F1 S級 adopted independent win probability")
    parser.add_argument("--input", required=True, type=Path, help="JSON racecard, not result data")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, type=Path)
    args = parser.parse_args()
    card = json.loads(args.input.read_text(encoding="utf-8"))
    model = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps(predict(card, model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
