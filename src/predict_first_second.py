#!/usr/bin/env python3
"""Joint first-second probabilities from adopted F1 S-class independent win q.

q comes from the adopted first-win model. The conditional second formula was
fitted from real outcomes of 14,081 F1 S-class races, but uses ONLY pre-race
q and lineup at inference time. Never condition on the actual future winner.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from predict_first import DEFAULT_MANIFEST, parse_lines, predict

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SECOND=ROOT/"models/second_conditional_v1.json"


def predict_first_second(card, first_manifest, second_manifest):
    if second_manifest.get("status")!="SELECTED_RESEARCH_REFERENCE_NOT_MARKET_OR_WAGER_MODEL":
        raise ValueError("Unknown or nonadopted second-place parameter version")
    if second_manifest.get("scope")!="F1 S級 only":
        raise ValueError("Second-place model scope must be F1 S級")
    first=predict(card,first_manifest)
    qs={int(x["car_no"]):float(x["calibrated_first_place_probability"])
        for x in first["predictions"]}
    scale=sum(qs.values())
    qs={car:value/scale for car,value in qs.items()}
    lines=parse_lines(card["predicted_line_formation"],list(qs))
    placement={}
    for n,line in enumerate(lines):
        for k,car in enumerate(line):
            placement[car]=(n,k)
    co=second_manifest["coefficients"]
    g=float(co["gamma"])
    pairs=[]
    second_prob={car:0.0 for car in qs}
    for i,qi in sorted(qs.items()):
        line_i,pos_i=placement[i]
        weights={}
        for j,qj in qs.items():
            if i==j:continue
            line_j,pos_j=placement[j]
            if line_j!=line_i:
                effect=float(co["other_line"])
            elif pos_j==pos_i+1:
                effect=float(co["directly_behind_winner"])
            elif pos_j<pos_i:
                effect=float(co["ahead_of_winner"])
            elif pos_j>pos_i+1:
                effect=float(co["later_behind_winner"])
            else:
                raise RuntimeError("Invalid duplicated line position")
            weights[j]=qj**g * math.exp(effect)
        norm=sum(weights.values())
        for j,w in sorted(weights.items()):
            conditional=w/norm
            joint=qi*conditional
            second_prob[j]+=joint
            pairs.append({
                "first_car_no":i,
                "second_car_no":j,
                "conditional_second_given_first":conditional,
                "joint_first_second_probability":joint,
            })
    if abs(sum(row["joint_first_second_probability"] for row in pairs)-1)>1e-10:
        raise RuntimeError("First-second probability not normalized")
    if abs(sum(second_prob.values())-1)>1e-10:
        raise RuntimeError("Second marginal not normalized")
    for car,p in qs.items():
        implied=sum(x["joint_first_second_probability"] for x in pairs if x["first_car_no"]==car)
        if abs(implied-p)>1e-10:raise RuntimeError("Joint not consistent with q")
    return {
        "scope":"F1 S級 only",
        "model":"adopted first-win q plus second_conditional_v1",
        "caveat":"Mathematical probability output; no market odds or expected value analysis",
        "first_marginal":[{"car_no":i,"probability":v} for i,v in sorted(qs.items())],
        "second_marginal":[{"car_no":j,"probability":v} for j,v in sorted(second_prob.items())],
        "ordered_first_second_pairs":pairs,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True,type=Path,help="Pre-race F1 S級 entrants and formation in JSON")
    parser.add_argument("--first",type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument("--second",type=Path,default=DEFAULT_SECOND)
    args=parser.parse_args()
    card=json.loads(args.input.read_text(encoding="utf-8"))
    first=json.loads(args.first.read_text(encoding="utf-8"))
    second=json.loads(args.second.read_text(encoding="utf-8"))
    result=predict_first_second(card,first,second)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
