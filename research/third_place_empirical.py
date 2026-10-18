#!/usr/bin/env python3
"""Results-first third-finisher location census. No fitted coefficients.

For every of the fixed 14,081 F1 S級 races:
- condition on the observed first AND second,
- consider each remaining candidate, flag actual third,
- compare to Harville q_k / (1-q_first-q_second),
- classify the rider's *predicted* line position against BOTH finishers.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict,Counter
from pathlib import Path
from hashlib import sha256

BINS=(0,.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5,.6,.7,1.00000001)

def classify(c, a, b, positions):
    line,pos=positions[c];la,pa=positions[a];lb,pb=positions[b]
    with_a=line==la
    with_b=line==lb
    if with_a and with_b:
        relation="line_of_both"
        direction=("ahead_of_both" if pos<min(pa,pb)
                   else "between_first_second" if pos<max(pa,pb)
                   else "behind_both")
    elif with_a:
        relation="first_line_only"
        direction=("ahead_of_first" if pos<pa else
                   "immediately_behind_first" if pos==pa+1 else
                   "further_behind_first")
    elif with_b:
        relation="second_line_only"
        direction=("ahead_of_second" if pos<pb else
                   "immediately_behind_second" if pos==pb+1 else
                   "further_behind_second")
    else:
        relation="neither_line"
        direction="neither_line"
    role=("solo" if len(line)==1 else "leader" if pos==0 else "second" if pos==1 else "third_or_later")
    return relation,direction,role

def summary(values):
    n=len(values)
    if not n:return {"candidates":0,"observed":0,"expected":0}
    exp=sum(x["baseline"] for x in values)
    observed=sum(x["hit"] for x in values)
    return {"candidates":n,"races_with_candidate":len({x["race_id"] for x in values}),
            "observed":observed,"expected":exp,
            "actual_rate":observed/n,"harville_rate":exp/n,
            "observed_minus_expected":observed-exp}

def analyze(path):
    events=[];fields=Counter();races=0;pair_configuration=Counter()
    for raw in path.open(encoding="utf-8"):
        if not raw.strip():continue
        r=json.loads(raw);races+=1
        qi={int(x["car_no"]):float(x["adopted_q"]) for x in r["first_win"]}
        i,j,k=(int(r[p]) for p in ("first","second","third"))
        if len({i,j,k})!=3 or not {i,j,k}.issubset(qi):
            raise ValueError(f"Invalid podium: {r['race_id']}")
        if abs(sum(qi.values())-1)>1e-8:raise ValueError("First probabilities not normalized")
        positions={}
        for line in r["predicted_line_formation"]:
            for pos,car in enumerate(line):
                if car in positions: raise ValueError("Duplicate line position")
                positions[car]=(tuple(line),pos)
        if set(positions)!=set(qi):raise ValueError("Line-map coverage changed")
        same_first_second=(positions[i][0]==positions[j][0])
        config="first_second_same_line" if same_first_second else "first_second_different_lines"
        pair_configuration[config]+=1
        denom=1-qi[i]-qi[j]
        if denom<=0:raise ValueError("Nonpositive conditional third denominator")
        fields[len(qi)]+=1
        for car,q in qi.items():
            if car in (i,j):continue
            relation,direction,role=classify(car,i,j,positions)
            events.append({
                "race_id":r["race_id"],"date":r["race_date"],"period":r["period"],
                "split":r["split_provenance_only"],"n":len(qi),
                "config":config,"car":car,"relation":relation,
                "direction":direction,"role":role,
                "baseline":q/denom,"q":q,"hit":int(car==k),
                "winner_line_length":len(positions[i][0]),
                "second_line_length":len(positions[j][0]),
            })
    if races!=14081 or len(events)!=69300 or sum(x["hit"] for x in events)!=races:
        raise RuntimeError("Frozen sample changed")
    return races,fields,pair_configuration,events

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    races,fields,configs,events=analyze(args.input)
    out={
      "status":"THIRD_PLACE_RESULT_FIRST_EMPIRICAL_ONLY",
      "scope":"F1 S級 only",
      "races":races,"third_candidate_rows":len(events),
      "field_sizes":dict(sorted(fields.items())),
      "observed_first_second_configuration_races":dict(configs),
      "baseline_equation":"P(third=k | actual first=i, actual second=j)=q_k/(1-q_i-q_j)",
      "no_3rd_coefficients_fitted":True,
      "whole":summary(events),
      "by_configuration":{},"by_relation":{},"by_direction":{},"by_role":{},
      "configuration_x_relation":{},
      "configuration_x_direction":{},
      "time_period_x_relation":{},
      "split_x_relation":{},
      "probability_bins":[],
    }
    for p,name in (("config","by_configuration"),("relation","by_relation"),
                   ("direction","by_direction"),("role","by_role")):
        for key in sorted({x[p] for x in events}):
            out[name][key]=summary([x for x in events if x[p]==key])
    for cfg in sorted(configs):
        rows=[x for x in events if x["config"]==cfg]
        out["configuration_x_relation"][cfg]={key:summary([x for x in rows if x["relation"]==key])
                                    for key in sorted({x["relation"] for x in rows})}
        out["configuration_x_direction"][cfg]={key:summary([x for x in rows if x["direction"]==key])
                                    for key in sorted({x["direction"] for x in rows})}
    for period in sorted({e["period"] for e in events}):
        rows=[e for e in events if e["period"]==period]
        out["time_period_x_relation"][period]={key:summary([e for e in rows if e["relation"]==key])
                                           for key in sorted({x["relation"] for x in rows})}
    for split in ("fit","chronological_check"):
        rows=[e for e in events if e["split"]==split]
        out["split_x_relation"][split]={key:summary([e for e in rows if e["relation"]==key])
                                       for key in sorted({x["relation"] for x in rows})}
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        rows=[e for e in events if lo<=e["baseline"]<hi]
        out["probability_bins"].append({"range":[lo,min(hi,1)],**summary(rows)})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("RESULT_FIRST_THIRD_POSITION_CENSUS",json.dumps({
        "races":races,"candidate_rows":len(events),"first_second_configuration":out["observed_first_second_configuration_races"],
        "relation":out["by_relation"],"directions":out["by_direction"],
        "by_configuration":out["configuration_x_relation"]},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
