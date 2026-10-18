#!/usr/bin/env python3
"""Stage 14,081 F1 S-class races from pinned ORIGIN for 2nd/3rd research.

Read frozen historical independent first-win CSV, podium labels and published
line formation. Verify SHA256 and exact race membership. No raw-data duplication
in this repository: a local old-repo checkout at SOURCE_LOCK's commit is needed.

Output is a local .jsonl research table. It is not a certified predeadline
snapshot and is not yet an estimated second/third probability model.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data" / "source_ledger.json"
LOCK = ROOT / "config" / "scope_lock.json"
THETA = 0.6412587661746468
LINE_RE = re.compile(r"^[1-9](?:-[1-9])*$")


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def source_groups(origin):
    groups = []
    for year in (2023, 2024, 2025):
        for quarter in range(1, 5):
            d = origin / f"data/{year}/s_class_f1_all_parts/{year}_q{quarter}"
            if (d / "entries.csv").exists() and (d / "races.csv").exists():
                groups.append(d)
    for relative in (
        "data/2026_h1/s_class_f1_all",
        "data/2026_future_block1/s_class_f1_20260701_20260830",
        "research/benter_calibration_v4/sep_holdout",
    ):
        d = origin / relative
        if (d / "entries.csv").exists() and (d / "races.csv").exists():
            groups.append(d)
    return groups


def line_of(card, cars):
    formation = (card.get("predicted_line_formation") or "").strip()
    strings = formation.split("/")
    if not formation or any(not LINE_RE.fullmatch(g) for g in strings):
        raise ValueError(f"Missing/invalid lineup for {card.get('race_id')}")
    lines = [[int(x) for x in s.split("-")] for s in strings]
    flat = [x for line in lines for x in line]
    if len(set(flat)) != len(flat) or set(flat) != set(cars):
        raise ValueError(f"Lineup not equal to the field for {card.get('race_id')}")
    return lines


def ids_hash(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def stage(origin, output):
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert ledger["origin_commit_sha"] == lock["source_commit_sha"]
    assert lock["scope_id"] == "F1_S_CLASS_ONLY"
    forecasts = {}
    for period in ledger["forecasts"]:
        path = origin / period["forecast_csv_path"]
        if not path.is_file():
            raise RuntimeError(f"Pinned input unavailable: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != period["predicted_win_csv_sha256"]:
            raise RuntimeError(f"Forecast hash mismatch: {path}")
        count = 0
        for row in read_csv(path):
            count += 1
            rid, date, car = row["race_id"], row["race_date"], int(row["car_no"])
            p = float(row["independent_win_probability"])
            if not 0 < p < 1 or not math.isfinite(p):
                raise RuntimeError(f"Bad probability: {rid}:{car}")
            rec = forecasts.setdefault(rid, {"date":date, "period":period["period"], "cars":{}})
            if rec["date"] != date or rec["period"] != period["period"] or car in rec["cars"]:
                raise RuntimeError(f"Forecast race collision: {rid}:{car}")
            rec["cars"][car] = p
        if count != period["target_rider_rows"]:
            raise RuntimeError(f"Forecast candidate count mismatch: {period['period']}")
    if len(forecasts) != lock["generated_forecast_races"]:
        raise RuntimeError("Forecast race count mismatch")

    outcomes = defaultdict(lambda:defaultdict(list))
    racecards = {}
    for d in source_groups(origin):
        for row in read_csv(d / "races.csv"):
            rid = row.get("race_id")
            if rid in forecasts:
                card = racecards.setdefault(rid, row)
                grade = unicodedata.normalize("NFKC", str(card.get("meeting_grade", ""))).upper()
                typ = unicodedata.normalize("NFKC", str(card.get("race_type", ""))).upper()
                if grade != "F1" or not typ.startswith("S級"):
                    raise RuntimeError(f"Race outside F1 S-class: {rid}")
        result_path = d / "results.csv"
        if not result_path.exists():
            continue
        for row in read_csv(result_path):
            rid = row.get("race_id")
            if rid in forecasts:
                place = str(row.get("finish_position", "")).strip()
                if place in ("1","2","3"):
                    outcomes[rid][place].append(row.get("car_no"))

    out_rows = []
    skipped = Counter()
    bysplit = {"fit":{"ids":[],"seconds":0,"thirds":0}, "chronological_check":{"ids":[],"seconds":0,"thirds":0}}
    for rid, rec in forecasts.items():
        all_places = outcomes.get(rid, {})
        if any(len(all_places.get(str(k), [])) != 1 for k in (1, 2, 3)):
            skipped["missing_ambiguous_podium"] += 1
            continue
        try:
            podium = [int(all_places[str(k)][0]) for k in (1, 2, 3)]
        except (ValueError, TypeError):
            skipped["missing_ambiguous_podium"] += 1
            continue
        cars = sorted(rec["cars"])
        if len(set(podium)) != 3 or not set(podium).issubset(cars):
            skipped["missing_ambiguous_podium"] += 1
            continue
        if rid not in racecards:
            raise RuntimeError(f"Missing source racecard: {rid}")
        p = [rec["cars"][car] for car in cars]
        if abs(sum(p) - 1.0) > 1e-8:
            raise RuntimeError(f"Forecast does not sum to one: {rid}")
        weights = [v * math.exp(-THETA * max(v - .4, 0.0)) for v in p]
        norm = sum(weights)
        q = [v / norm for v in weights]
        formation = line_of(racecards[rid], cars)
        split = "fit" if rec["date"] < "2025-04-01" else "chronological_check"
        bysplit[split]["ids"].append(rid)
        bysplit[split]["seconds"] += len(cars)-1
        bysplit[split]["thirds"] += len(cars)-2
        out_rows.append({
            "race_id":rid, "race_date":rec["date"], "period":rec["period"],
            "race_type":racecards[rid].get("race_type"),
            "split_provenance_only":split,
            "predicted_line_formation":formation,
            "first":podium[0], "second":podium[1], "third":podium[2],
            "first_win": [
                {"car_no": car, "raw_p": a, "adopted_q": b}
                for car,a,b in zip(cars,p,q)
            ],
        })
    if len(out_rows) != lock["all_eligible_races"]:
        raise RuntimeError(f"Eligible sample is not fixed: {len(out_rows)}")
    if sum(skipped.values()) != lock["podium_invalid_excluded"]:
        raise RuntimeError(f"Unexpected skipped podium count: {dict(skipped)}")
    for split, stats in bysplit.items():
        required = lock["partitions_provenance_only"][
            "exploratory_fit_to_2025_03_31" if split=="fit" else "chronology_check_from_2025_04_01"
        ]
        if (len(stats["ids"]) != required["races"] or
                stats["seconds"] != required["second_candidates"] or
                stats["thirds"] != required["third_candidates"] or
                ids_hash(stats["ids"]) != required["race_ids_sha256"]):
            raise RuntimeError(f"Fixed cohort hash or count changed: {split}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in sorted(out_rows, key=lambda r:(r["race_date"],r["race_id"])):
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    meta = {
        "status":"DATA_STAGE_READY_NOT_2ND_OR_3RD_FIT",
        "source_commit_expected":ledger["origin_commit_sha"],
        "eligible_races":len(out_rows),
        "skipped":dict(skipped),
        "development_count":len(bysplit["fit"]["ids"]),
        "chronology_count":len(bysplit["chronological_check"]["ids"]),
        "second_candidate_rows":sum(x["seconds"] for x in bysplit.values()),
        "third_candidate_rows":sum(x["thirds"] for x in bysplit.values()),
        "output_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "note":"Development samples have retrospectively fitted calibration. Historical racecard timestamps are not certified predeadline snapshots."
    }
    report = output.with_suffix(".audit.json")
    report.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root",type=Path,required=True,help="Old repository checked out at pinned SHA")
    ap.add_argument("--output",type=Path,default=ROOT/"stage/f1_s_class_14081.jsonl")
    a=ap.parse_args()
    stage(a.source_root.resolve(),a.output.resolve())
