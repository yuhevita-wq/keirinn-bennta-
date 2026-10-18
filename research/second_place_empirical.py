#!/usr/bin/env python3
"""Stage 1: empirical 2nd-place conditional structure from real F1 S級 outcomes.

Uses confirmed FIRST-place finisher to condition SECOND-place candidate outcomes.
No coefficient fitting; all conditional expected values use the adopted frozen q.
All 14,081 eligible races are retained, including foreign riders, 5-7 car races.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BINS=(0,.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5,.6,.7,1.00000001)

def summary(events):
    n=len(events)
    if not n:return {"candidate_count":0}
    expected=sum(e["p"] for e in events)
    observed=sum(e["y"] for e in events)
    variance=sum(e["p"]*(1-e["p"]) for e in events)
    return {
        "candidate_count":n,
        "unique_races":len({e["race"] for e in events}),
        "mean_predicted_probability":expected/n,
        "actual_frequency":observed/n,
        "expected_2nd":expected,
        "actual_2nd":observed,
        "observed_minus_expected":observed-expected,
        "z_normal_reference_not_independence_adjusted":(observed-expected)/math.sqrt(variance) if variance else None
    }

def build_events(path):
    events=[]
    race_count=0
    fields=Counter()
    for line in path.open("r",encoding="utf-8"):
        if not line.strip(): continue
        r=json.loads(line);race_count+=1
        q={x["car_no"]:x["adopted_q"] for x in r["first_win"]}
        first,second=r["first"],r["second"]
        assert first in q and second in q and first!=second
        assert math.isclose(sum(q.values()),1.,abs_tol=1e-8)
        lines=r["predicted_line_formation"]
        assert len(lines)>0
        placement={}
        for group in lines:
            for k,car in enumerate(group):
                assert car not in placement
                placement[car]=(tuple(group),k)
        assert set(placement)==set(q)
        first_line,first_pos=placement[first]
        fields[len(q)]+=1
        denom=1-q[first]
        for car,v in q.items():
            if car==first: continue
            group,pos=placement[car]
            same=group==first_line
            if len(group)==1:role="solo"
            elif pos==0:role="leader"
            elif pos==1:role="second"
            else:role="third_or_later"
            direction=("directly_behind_winner" if same and pos==first_pos+1
                else "behind_winner_later" if same and pos>first_pos+1
                else "ahead_of_winner" if same and pos<first_pos
                else "other_line")
            events.append({
                "race":r["race_id"],"date":r["race_date"],"period":r["period"],
                "n":len(q),"first":first,"car":car,
                "first_q":q[first],"q":v,"p":v/denom,
                "y":int(car==second),
                "same_line":same,"role":role,"direction":direction,
                "winner_role":("solo" if len(first_line)==1 else
                                "leader" if first_pos==0 else
                                "second" if first_pos==1 else "third_or_later"),
                "split":r["split_provenance_only"]
            })
    if race_count!=14081 or len(events)!=83381:
        raise RuntimeError(f"Fixed universe drift: {race_count} races, {len(events)} candidates")
    if sum(e["y"] for e in events)!=race_count:
        raise RuntimeError("Exactly one actual second finisher must exist per race")
    return race_count,fields,events

def empirical(races,fields,events):
    out={
        "research_stage":"EMPIRICAL_RESULT_FIRST_BEFORE_MODEL_FITTING",
        "race_count":races,
        "candidate_rows":len(events),
        "field_sizes":dict(sorted(fields.items())),
        "reference":"Harville conditional baseline P(second=j | first=i)=q_j/(1-q_i). Same rows as Benter Table 9; q is corrected independently estimated F1 S級 first-place probability.",
        "no_second_model_fitted":True,
        "all":summary(events),
        "bins":[],
        "winner_line_relation":{},
        "second_candidate_role":{},
        "directed_line_position":{},
        "winner_role_x_relation":{},
        "first_probability_bins":{},
        "split_x_relation":{},
        "per_period_relation":{},
    }
    for low,high in zip(BINS[:-1],BINS[1:]):
        group=[e for e in events if low<=e["p"]<high]
        out["bins"].append({"range":[low,min(high,1)],**summary(group)})
    for key,outname,make in [
       ("same_line","winner_line_relation",lambda x:"same_line" if x else "other_line"),
       ("role","second_candidate_role",lambda x:x),
       ("direction","directed_line_position",lambda x:x),
    ]:
        for v in sorted({make(e[key]) for e in events}):
            out[outname][v]=summary([e for e in events if make(e[key])==v])
    for role in sorted({e["winner_role"] for e in events}):
        sub=[e for e in events if e["winner_role"]==role]
        out["winner_role_x_relation"][role]={
          key:summary([e for e in sub if e["same_line"]==yes])
          for key,yes in (("same_line",True),("other_line",False))}
    for tag,rng in (
       ("first_q_below_10pct",(0,.1)),("first_q_10_to_20pct",(.1,.2)),
       ("first_q_20_to_40pct",(.2,.4)),("first_q_40_to_60pct",(.4,.6)),
       ("first_q_over_60pct",(.6,1.00001))
    ):
        sub=[e for e in events if rng[0]<=e["first_q"]<rng[1]]
        out["first_probability_bins"][tag]={
          "same_line":summary([e for e in sub if e["same_line"]]),
          "other_line":summary([e for e in sub if not e["same_line"]])}
    for split in ("fit","chronological_check"):
        sub=[e for e in events if e["split"]==split]
        out["split_x_relation"][split]={
          "same_line":summary([e for e in sub if e["same_line"]]),
          "other_line":summary([e for e in sub if not e["same_line"]])}
    for period in sorted({e["period"] for e in events}):
        sub=[e for e in events if e["period"]==period]
        out["per_period_relation"][period]={
          "same_line":summary([e for e in sub if e["same_line"]]),
          "other_line":summary([e for e in sub if not e["same_line"]])}
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    n,fields,events=build_events(args.input)
    result=empirical(n,fields,events)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    important={k:result[k] for k in ("race_count","candidate_rows","field_sizes","winner_line_relation","second_candidate_role","directed_line_position","split_x_relation")}
    print("SECOND_EMPIRICAL_FINISHED",json.dumps(important,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
