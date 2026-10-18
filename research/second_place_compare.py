#!/usr/bin/env python3
"""Compare what second-place model complexity actually buys on same race set."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from second_place_fit import (read,coefficients,probs,indices,evaluate,pair_bootstrap,MODELS,COLS)

def grouped_calibration(a,ix,pr):
    M=a["M"][ix];F=a["F"][ix];y=a["y"][ix]
    n=len(ix)
    actual=np.zeros_like(M,dtype=bool)
    actual[np.arange(n),y]=True
    groups={
      "same_line":F[:,:,COLS.index("same_line")]>.5,
      "other_line":F[:,:,COLS.index("same_line")]<.5,
      "directly_behind_winner":F[:,:,COLS.index("directly_behind_winner")]>.5,
      "ahead_of_winner":F[:,:,COLS.index("ahead_of_winner")]>.5,
      "later_behind_winner":F[:,:,COLS.index("later_behind_winner")]>.5,
    }
    report={}
    for name,b in groups.items():
        choose=M&b
        nrows=int(choose.sum())
        exp=float(pr[choose].sum())
        act=int(actual[choose].sum())
        report[name]={
           "candidate_rows":nrows,
           "expected_second":exp,
           "actual_second":act,
           "predicted_rate":exp/nrows if nrows else None,
           "observed_rate":act/nrows if nrows else None,
           "expected_minus_actual":exp-act
        }
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    a=read(args.input)
    dev=np.array([i for i,s in enumerate(a["split"]) if s=="fit"],dtype=int)
    check=np.array([i for i,s in enumerate(a["split"]) if s=="chronological_check"],dtype=int)
    allidx=np.arange(len(a["id"]),dtype=int)
    models=("harville","benter_1994_gamma_081","estimated_gamma",
            "gamma_plus_same_line","gamma_plus_directed_line",
            "gamma_plus_directed_and_role")
    report={
        "status":"SECOND_PLACE_PAIRWISE_POSITION_AND_ROLE_DIAGNOSTICS",
        "sample_all_races":len(allidx),
        "chronology_evaluation_races":len(check),
        "purpose":"Decide whether directed line position and candidate role add reliable improvement beyond same-line adjustment; never mutate adopted first model.",
        "comparison":{"dev_train_evaluation":{},"all_data_descriptive_in_sample":{}},
        "paired_day_cluster_on_chronological_check":{},
        "no_market_odds":True,
        "reproducibility":"models fit only 2024 to 2025Q1 for chronology, full sample refit reported separately."
    }
    losses={}
    for name in models:
        b,col=coefficients(name,a,dev)
        full,col2=coefficients(name,a,allidx)
        pe=probs(name,b,a,check)
        pf=probs(name,full,a,allidx)
        met,loss=evaluate(name,b,a,check)
        losses[name]=loss
        byper={}
        for period in sorted(set(a["period"][i] for i in check)):
            selection=np.array([i for i in check if a["period"][i]==period],dtype=int)
            pm,ll=evaluate(name,b,a,selection)
            byper[period]={"races":len(selection),"log_loss":pm["conditional_second_log_loss"],
                           "top1_second_accuracy":pm["top_candidate_second_accuracy"]}
        report["comparison"]["dev_train_evaluation"][name]={
            "coefficient_names":col,"coefficients":[float(x) for x in b],
            "conditional_second_log_loss":met["conditional_second_log_loss"],
            "calendar_subgroups":byper,
            "observed_vs_predicted_line_relation":grouped_calibration(a,check,pe)
        }
        full_met,_=evaluate(name,full,a,allidx)
        report["comparison"]["all_data_descriptive_in_sample"][name]={
            "coefficient_names":col2,"coefficients":[float(x) for x in full],
            "conditional_second_log_loss":full_met["conditional_second_log_loss"],
            "observed_vs_predicted_line_relation":grouped_calibration(a,allidx,pf)
        }
    dates=[a["date"][i] for i in check]
    for base,test in (("gamma_plus_same_line","gamma_plus_directed_line"),
                      ("gamma_plus_same_line","gamma_plus_directed_and_role"),
                      ("gamma_plus_directed_line","gamma_plus_directed_and_role"),
                      ("estimated_gamma","gamma_plus_same_line")):
        report["paired_day_cluster_on_chronological_check"][test+"_OVER_"+base]=pair_bootstrap(
            losses[base],losses[test],dates
        )
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("PAIRWISE_SECOND_REPORT",json.dumps({
      "pairs":report["paired_day_cluster_on_chronological_check"],
      "line_comparison":{k:{v["candidate_count"] if "candidate_count" in v else n:0 for n,v in []}
                         for k in []}
      },ensure_ascii=False),flush=True)
if __name__=="__main__":
    main()
