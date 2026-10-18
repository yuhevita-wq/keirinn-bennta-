#!/usr/bin/env python3
"""All ordered 1-2-3 finishing probabilities from independently fitted F1 S models.

This module does not access actual finishing results or market odds.
The same pre-race first probabilities q are fed to second and third models.
Third was estimated conditionally given observed first/second in history, but
INFERENCE marginalizes over all possible first+second outcomes.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from predict_first import DEFAULT_MANIFEST,parse_lines
from predict_first_second import DEFAULT_SECOND,predict_first_second

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_THIRD=ROOT/"models/third_conditional_v1.json"

def third_position(k,i,j,lines):
    places={}
    for line in lines:
        for pos,car in enumerate(line):
            places[car]=(tuple(line),pos)
    lk,pk=places[k];li,pi=places[i];lj,pj=places[j]
    if lk==li and lk==lj:
        if pk<min(pi,pj):return "ahead_of_both"
        if pk>max(pi,pj):return "behind_both"
        return "between_first_second"
    if lk==li:
        if pk<pi:return "ahead_of_first"
        if pk==pi+1:return "immediately_behind_first"
        return "further_behind_first"
    if lk==lj:
        if pk<pj:return "ahead_of_second"
        if pk==pj+1:return "immediately_behind_second"
        return "further_behind_second"
    return "neither_line"

def third_role(k,lines):
    for line in lines:
        if k in line:
            pos=line.index(k)
            if len(line)==1:return "role_solo"
            if pos==0:return "role_leader"
            if pos==1:return "role_second"
            return "role_third_or_later"
    raise ValueError("Third car not in lineup")

def infer_triples(card,first_manifest,second_manifest,third_manifest):
    if third_manifest.get("status")!="SELECTED_RESEARCH_REFERENCE_NOT_MARKET_OR_WAGER_MODEL":
        raise ValueError("Unrecognized third conditional model")
    if third_manifest.get("scope")!="F1 S級 only":
        raise ValueError("Wrong third-model scope")
    second=predict_first_second(card,first_manifest,second_manifest)
    q={int(x["car_no"]):float(x["probability"]) for x in second["first_marginal"]}
    lines=parse_lines(card.get("predicted_line_formation"),list(q))
    params=third_manifest["coefficients"]
    delta=float(params["delta"])
    triples=[]
    third_marginal={k:0. for k in q}
    pair_marginal={}
    for pair in second["ordered_first_second_pairs"]:
        i=int(pair["first_car_no"]);j=int(pair["second_car_no"])
        joint2=float(pair["joint_first_second_probability"])
        scores={}
        for k,qk in q.items():
            if k in (i,j):continue
            name=third_position(k,i,j,lines)
            role=third_role(k,lines)
            score=delta*math.log(qk)+float(params[name])+float(params[role])
            scores[k]=score
        maxscore=max(scores.values())
        weights={k:math.exp(z-maxscore) for k,z in scores.items()}
        norm=sum(weights.values())
        if not norm>0:raise ValueError("Invalid third normalization")
        acc=0.0
        for k,w in weights.items():
            cond=w/norm
            prob=joint2*cond
            acc+=prob
            third_marginal[k]+=prob
            triples.append({
                "first_car_no":i,
                "second_car_no":j,
                "third_car_no":k,
                "probability":prob,
                "third_conditional_given_first_second":cond
            })
        if abs(acc-joint2)>1e-12:
            raise RuntimeError("First-second joint probability changed during third stage")
        pair_marginal[(i,j)]=acc
    if abs(sum(t["probability"] for t in triples)-1)>1e-10:
        raise RuntimeError("Ordered triple distribution not normalized")
    if abs(sum(third_marginal.values())-1)>1e-10:
        raise RuntimeError("Third marginal not normalized")
    n=len(q)
    if len(triples)!=n*(n-1)*(n-2):
        raise RuntimeError("Not all ordered 3-car combinations evaluated")
    first_joint={(int(p["first_car_no"]),int(p["second_car_no"])):float(p["joint_first_second_probability"])
                 for p in second["ordered_first_second_pairs"]}
    if set(first_joint)!=set(pair_marginal):
        raise RuntimeError("Lost ordered 1st+2nd pair")
    for pair,p in first_joint.items():
        if abs(pair_marginal[pair]-p)>1e-12:
            raise RuntimeError("Triple did not preserve existing first-second model")
    triples.sort(key=lambda v:(-v["probability"],v["first_car_no"],v["second_car_no"],v["third_car_no"]))
    return {
        "scope":"F1 S級 only",
        "model":"adopted first model, second_conditional_v1, third_conditional_v1",
        "scope_note":"No payout/odds or expected value computed; not a betting recommendation",
        "all_ordered_triples_count":len(triples),
        "first_marginal":second["first_marginal"],
        "second_marginal":second["second_marginal"],
        "third_marginal":[{"car_no":k,"probability":v} for k,v in sorted(third_marginal.items())],
        "ordered_first_second_third_triples":triples
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True,help="Pre-race only, do not include outcomes")
    p.add_argument("--first",type=Path,default=DEFAULT_MANIFEST)
    p.add_argument("--second",type=Path,default=DEFAULT_SECOND)
    p.add_argument("--third",type=Path,default=DEFAULT_THIRD)
    args=p.parse_args()
    card=json.loads(args.input.read_text(encoding="utf-8"))
    first=json.loads(args.first.read_text(encoding="utf-8"))
    second=json.loads(args.second.read_text(encoding="utf-8"))
    third=json.loads(args.third.read_text(encoding="utf-8"))
    result=infer_triples(card,first,second,third)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
