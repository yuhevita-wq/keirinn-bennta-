#!/usr/bin/env python3
"""Read-only Benter-style comparison of the FIXED F1 S-class first-win model
with the 3-ren-tan final-odds-derived first-place marginal on IDENTICAL races.

No model coefficients, probabilities, predictor source, or purchase rules are changed.
Historical final odds are retrospective, NOT actionable pre-deadline quotes.
"""
import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def load_stage(path):
    races = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            rid = str(x["race_id"])
            cars = {int(v["car_no"]): float(v["adopted_q"]) for v in x["first_win"]}
            assert len(cars) in (5, 6, 7, 8, 9)
            assert abs(sum(cars.values()) - 1) < 1e-8
            assert int(x["first"]) in cars and rid not in races
            races[rid] = dict(date=x["race_date"], period=x["period"],
                              winner=int(x["first"]), model=cars)
    assert len(races) == 14081, len(races)
    return races


def list_odds(origin):
    out = []
    for year in (2024, 2025):
        for quarter in (1, 2, 3, 4):
            f = origin / f"data/{year}/s_class_f1_all_parts/{year}_q{quarter}/trifecta_final_odds.csv"
            if f.is_file():
                out.append(f)
    f = origin / "data/2026_h1/s_class_f1_all"
    out.extend(sorted(f.glob("trifecta_final_odds_part*.csv")))
    if not out:
        raise RuntimeError("No archived final-odds source files were checked out")
    return out


def load_market(origin, races):
    sums = defaultdict(lambda: defaultdict(float))
    counts = Counter()
    seen = defaultdict(set)
    problems = defaultdict(set)
    source_stats = []
    for file in list_odds(origin):
        stats = Counter()
        with file.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"race_id", "combination", "odds", "odds_status", "odds_phase"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected columns in {file}: {reader.fieldnames}")
            for row in reader:
                stats["rows_all"] += 1
                rid = row["race_id"]
                if rid not in races:
                    continue
                stats["rows_in_fixed_cohort"] += 1
                counts[rid] += 1
                combination = row["combination"]
                try:
                    order = tuple(map(int, combination.split("-")))
                    if len(order) != 3 or len(set(order)) != 3 or not set(order).issubset(races[rid]["model"]):
                        raise ValueError("Invalid combination")
                    if order in seen[rid]:
                        raise ValueError("Duplicate combination")
                    seen[rid].add(order)
                    odd = float(row["odds"])
                    if not math.isfinite(odd) or odd <= 1:
                        raise ValueError("Invalid odds")
                    if row["odds_status"] != "available" or row["odds_phase"] != "final":
                        raise ValueError("Not a final available quote")
                    sums[rid][order[0]] += 1.0 / odd
                except (ValueError, TypeError, OverflowError) as exc:
                    problems[rid].add(str(exc))
        source_stats.append(dict(file=str(file.relative_to(origin)), **stats))
    eligible = {}
    exclusion = Counter()
    for rid, race in races.items():
        n = len(race["model"])
        expected = n * (n - 1) * (n - 2)
        if counts[rid] == 0:
            exclusion["no_trifecta_market_rows"] += 1
            continue
        if problems[rid]:
            exclusion["invalid_or_duplicate_market_rows"] += 1
            continue
        if counts[rid] != expected or len(seen[rid]) != expected:
            exclusion["incomplete_trifecta_market"] += 1
            continue
        total = sum(sums[rid].values())
        if not 0 < total < float("inf"):
            exclusion["invalid_market_normalization"] += 1
            continue
        m = {car: sums[rid].get(car, 0) / total for car in race["model"]}
        if not all(0 < p < 1 for p in m.values()) or abs(sum(m.values()) - 1) > 1e-8:
            exclusion["invalid_market_first_marginal"] += 1
            continue
        eligible[rid] = {**race, "market": m, "trifecta_overround": total}
    return eligible, dict(exclusion), source_stats


def row_data(eligible):
    rows = []
    for rid, r in sorted(eligible.items(), key=lambda x: (x[1]["date"], x[0])):
        cars = sorted(r["model"])
        m = np.array([r["model"][c] for c in cars], dtype=float)
        q = np.array([r["market"][c] for c in cars], dtype=float)
        win = cars.index(r["winner"])
        rows.append(dict(rid=rid, date=r["date"], period=r["period"], n=len(cars),
                         win=win, model=m, market=q,
                         subset="fit" if r["date"] < "2025-04-01" else "later"))
    return rows


def fit_alpha_beta(rows):
    def fn_ab(ab):
        loss = 0.0
        grad = np.zeros(2)
        for r in rows:
            features = np.array([np.log(r["model"]), np.log(r["market"])])
            logits = ab @ features
            maxlogit = np.max(logits)
            weights = np.exp(logits - maxlogit)
            p = weights / np.sum(weights)
            loss += -math.log(max(p[r["win"]], 1e-300))
            grad += (features @ p - features[:, r["win"]])
        return loss, grad
    res = minimize(fn_ab, np.array([1.0, 1.0]), jac=True,
                   bounds=[(0, 5), (0, 5)], method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Combined model fit failed: {res.message}")
    return [float(v) for v in res.x], float(res.fun)


def combined(m, market, alpha, beta):
    logs = alpha * np.log(m) + beta * np.log(market)
    w = np.exp(logs - max(logs))
    return w / np.sum(w)


def calc(rows, ab):
    if not rows:
        return None
    metrics = {k: dict(loss=0., brier=0., hits=0) for k in ("independent", "market", "combined")}
    base = 0.
    daily = defaultdict(lambda: {"n": 0, "model_minus_market": 0.,
                                  "combined_minus_market": 0.})
    for r in rows:
        win = r["win"]
        base += math.log(r["n"])
        probs = {"independent": r["model"], "market": r["market"],
                 "combined": combined(r["model"], r["market"], *ab)}
        d = {}
        for key, p in probs.items():
            ll = -math.log(max(p[win], 1e-300))
            d[key] = ll
            metrics[key]["loss"] += ll
            target = np.zeros(len(p))
            target[win] = 1.
            metrics[key]["brier"] += float(np.sum((p - target) ** 2))
            metrics[key]["hits"] += int(np.argmax(p) == win)
        daily[r["date"]]["n"] += 1
        daily[r["date"]]["model_minus_market"] += d["independent"] - d["market"]
        daily[r["date"]]["combined_minus_market"] += d["combined"] - d["market"]
    n = len(rows)
    for k, v in metrics.items():
        v["races"] = n
        v["log_loss"] = v.pop("loss") / n
        v["brier"] /= n
        v["top1_accuracy"] = v.pop("hits") / n
        v["benter_pseudo_r2"] = 1 - v["log_loss"] * n / base
    rng = np.random.default_rng(1994)
    vals = list(daily.values())
    sizes = np.array([v["n"] for v in vals])
    ix = rng.integers(0, len(vals), size=(1500, len(vals)))
    ci = {}
    for k in ("model_minus_market", "combined_minus_market"):
        z = np.array([v[k] for v in vals])
        boots = z[ix].sum(axis=1) / sizes[ix].sum(axis=1)
        ci[k] = {"delta_mean_loss": float(z.sum() / n),
                 "date_block_bootstrap_95ci": [float(v) for v in np.quantile(boots, [0.025, 0.975])]}
    return dict(n=n, date_start=min(r["date"] for r in rows),
                date_end=max(r["date"] for r in rows),
                models=metrics, paired=ci, unique_days=len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    races = load_stage(a.stage)
    usable, reasons, stats = load_market(a.source_root, races)
    rows = row_data(usable)
    train = [r for r in rows if r["subset"] == "fit"]
    test = [r for r in rows if r["subset"] == "later" and r["date"] <= "2026-06-30"]
    if len(train) < 500 or len(test) < 500:
        raise RuntimeError(f"Not enough matched data: {len(train)} fit; {len(test)} holdout")
    ab, fitted_ll = fit_alpha_beta(train)
    result = {
        "study": "FIXED_INDEPENDENT_F1S_VS_MARKET_FIRST_FROM_FINAL_TRIFECTA",
        "model_modified": False,
        "source_locked_git_commit": "8da8f4f8e1125eb522df5546d07cd6657fab7dda",
        "stage_races": len(races),
        "matched_races": len(usable),
        "excluded": reasons,
        "market_method": "Normalize inverse final 3-ren-tan odds over full ordered combination space; marginalize by first-position car number",
        "market_warning": "Final odds are NOT pre-deadline odds. Retrospective alignment, not actionable profitability; this is not native win market.",
        "integrated_model": {
            "fit_dates": "2024-01-01 to 2025-03-31",
            "fit_coefficients": {"alpha_frozen_independent": ab[0], "beta_final_market": ab[1]},
            "fit_loss_total": fitted_ll,
            "note": "Auxiliary diagnostic combination only. Frozen independent model not changed.",
        },
        "fit": calc(train, ab),
        "later": calc(test, ab),
        "periods": {p: calc([r for r in test if r["period"] == p], ab)
                    for p in sorted(set(r["period"] for r in test))},
        "source_stats": stats,
        "limitations": [
          "Historic racecards/lineups were acquired retrospectively, their exact 15-minute-prior status is unproven.",
          "Model calibration changes were discovered in previously examined portions of the evaluation period: not a fully blind holdout.",
          "Final odds provide market information formed after the pre-deadline purchase snapshot.",
          "One ticket class (3-ren-tan) supplies a proxy for first-win market probabilities; not the single-win pool.",
          "This estimates relative probability-information gain, not EV or realized return from bets.",
        ]
    }
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("stage_races", "matched_races", "excluded", "integrated_model", "fit", "later", "periods")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
