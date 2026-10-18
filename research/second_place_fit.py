#!/usr/bin/env python3
"""Stage 2: MLE conditional-second mathematical rules, after empirical audit.

All fixed 14,081 F1 S-class cases included. The baseline is Harville.
Equations are candidate log-linear softmax rules, NOT asserted in advance to
be correct. All candidates use the SAME race IDs and frozen first-win q.
"""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

COLS=["log_q","same_line","directly_behind_winner","ahead_of_winner","later_behind_winner",
      "candidate_is_second_role","candidate_is_third_or_later_role","candidate_is_solo"]
MODELS={
    "harville":[],
    "benter_1994_gamma_081":["log_q"],
    "estimated_gamma":["log_q"],
    "gamma_plus_same_line":["log_q","same_line"],
    "gamma_plus_directed_line":["log_q","directly_behind_winner","ahead_of_winner","later_behind_winner"],
    "gamma_plus_candidate_role":["log_q","same_line","candidate_is_second_role","candidate_is_third_or_later_role","candidate_is_solo"],
    "gamma_plus_directed_and_role":["log_q","directly_behind_winner","ahead_of_winner","later_behind_winner","candidate_is_second_role","candidate_is_third_or_later_role","candidate_is_solo"],
}
# The Benter fixed gamma is a reference only; it was fitted to horse data.
LOG_EPS=1e-15
BINS=(0,.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5,.6,.7,1.00000001)

def read(path):
    records=[]
    for raw in path.open(encoding="utf-8"):
        if not raw.strip():continue
        records.append(json.loads(raw))
    if len(records)!=14081:raise RuntimeError("Fixed 14081-race universe changed")
    records.sort(key=lambda r:(r["race_date"],r["race_id"]))
    n=len(records);p=len(COLS)
    features=np.zeros((n,9,p),dtype=np.float64)
    mask=np.zeros((n,9),dtype=bool)
    observed=np.zeros(n,dtype=int)
    firsts=np.zeros(n,dtype=int)
    ids=[];dates=[];periods=[];splits=[]
    fields=Counter()
    for idx,r in enumerate(records):
        pcs={int(row["car_no"]):float(row["adopted_q"]) for row in r["first_win"]}
        cars=sorted(pcs)
        first=int(r["first"]);second=int(r["second"])
        line_map={}
        for line in r["predicted_line_formation"]:
            for pos,car in enumerate(line):
                line_map[car]=(line,pos)
        if first not in pcs or second not in pcs or first==second or set(line_map)!=set(pcs):
            raise RuntimeError("Invalid race: "+r["race_id"])
        if abs(sum(pcs.values())-1)>1e-8:raise RuntimeError("Win q not normalized")
        winline,winpos=line_map[first]
        for j,car in enumerate(cars):
            q=pcs[car]
            line,pos=line_map[car]
            same=line==winline
            mask[idx,j]=car!=first
            features[idx,j]=[
                math.log(max(q,LOG_EPS)),
                float(same),
                float(same and pos==winpos+1),
                float(same and pos<winpos),
                float(same and pos>winpos+1),
                float(pos==1 and len(line)>=2),
                float(pos>=2),
                float(len(line)==1)
            ]
        observed[idx]=cars.index(second);firsts[idx]=cars.index(first)
        ids.append(r["race_id"]);dates.append(r["race_date"])
        periods.append(r["period"]);splits.append(r["split_provenance_only"])
        fields[len(cars)]+=1
    if sum(fields.values())!=n or int(mask.sum())!=83381:raise RuntimeError("Candidate count mismatch")
    return {"F":features,"M":mask,"y":observed,"i":firsts,"date":dates,
            "id":ids,"period":periods,"split":splits,"fields":dict(fields)}

def indices(name):
    return [COLS.index(x) for x in MODELS[name]]

def coefficients(name,arr,which):
    if name=="harville":return np.array([1.]), ["log_q"]
    if name=="benter_1994_gamma_081":return np.array([.81]), ["log_q"]
    ix=indices(name)
    X=arr["F"][which][:,:,ix]
    M=arr["M"][which]
    y=arr["y"][which]
    n=len(y)
    yc=np.arange(n)
    def objective(b):
        z=(X@b)
        z=np.where(M,z,-1e9)
        v=logsumexp(z,axis=1)
        ll=np.mean(v-z[yc,y])
        pr=np.exp(z-v[:,None])
        grad=np.mean((pr[:,:,None]*X).sum(axis=1)-X[yc,y],axis=0)
        return ll,grad
    start=np.zeros(len(ix))
    start[0]=1.0
    bounds=[(.05,2.5)]+[(-5,5)]*(len(ix)-1)
    res=minimize(objective,start,jac=True,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":400,"ftol":1e-12,"gtol":1e-9})
    if not res.success and not np.isfinite(res.fun):raise RuntimeError("Fit failed: "+name)
    return res.x,MODELS[name]

def probs(name,b,arr,which):
    X=arr["F"][which]
    M=arr["M"][which]
    if name in ("harville","benter_1994_gamma_081"):
        z=X[:,:,0]*b[0]
    else:
        z=X[:,:,indices(name)]@b
    z=np.where(M,z,-1e9)
    pp=np.exp(z-logsumexp(z,axis=1)[:,None])
    return pp

def evaluate(name,b,arr,which,calibration=False):
    pp=probs(name,b,arr,which)
    y=arr["y"][which];n=len(y); row=np.arange(n)
    loss=-np.log(np.maximum(pp[row,y],LOG_EPS))
    top1=(np.argmax(pp,axis=1)==y)
    res={"races":n,"conditional_second_log_loss":float(np.mean(loss)),
         "top_candidate_second_accuracy":float(top1.mean()),
         "reference_uniform_log_loss":float(np.mean(np.log(arr["M"][which].sum(axis=1))))}
    if calibration:
        p=pp[arr["M"][which]]
        ground=np.zeros_like(pp,dtype=bool);ground[row,y]=True
        yy=ground[arr["M"][which]]
        res["calibration_by_predicted_probability"]=[]
        for lo,hi in zip(BINS[:-1],BINS[1:]):
            select=(p>=lo)&(p<hi)
            if select.sum():
                res["calibration_by_predicted_probability"].append({
                    "range":[lo,min(hi,1.)],"candidates":int(select.sum()),
                    "mean_predicted":float(p[select].mean()),
                    "actual_second_rate":float(yy[select].mean()),
                    "expected_second":float(p[select].sum()),
                    "actual_second":int(yy[select].sum()),
                })
    return res,loss

def pair_bootstrap(loss_a,loss_b,dates,seed=20261018,reps=1200):
    grouped=defaultdict(list)
    for a,b,d in zip(loss_a,loss_b,dates):
        grouped[d].append(float(a-b))
    ds=sorted(grouped)
    totals=np.array([sum(grouped[d]) for d in ds])
    lengths=np.array([len(grouped[d]) for d in ds])
    if len(ds)<3:return None
    random=np.random.default_rng(seed)
    ix=random.integers(len(ds),size=(reps,len(ds)))
    samples=totals[ix].sum(axis=1)/lengths[ix].sum(axis=1)
    return {"mean_improvement_positive_is_second_model_better":float(totals.sum()/lengths.sum()),
            "day_cluster_bootstrap_95_interval":[float(t) for t in np.quantile(samples,[.025,.975])],
            "days":len(ds)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    a=read(args.input)
    train=np.array([k for k,s in enumerate(a["split"]) if s=="fit"],dtype=int)
    check=np.array([k for k,s in enumerate(a["split"]) if s=="chronological_check"],dtype=int)
    allrows=np.arange(len(a["id"]),dtype=int)
    if len(train)!=6393 or len(check)!=7688:raise RuntimeError("Partition mismatch")
    report={
        "research_stage":"CONDITIONAL_SECOND_MLE_AFTER_EMPIRICAL_AUDIT",
        "scope":"F1 S級 only",
        "equation":"P(second=j | first=i) = q_j^gamma * exp(A*directly_behind + B*ahead_of_winner + C*later_behind_winner + role_adjustments) / sum_{h != i}(q_h^gamma * exp(A*directly_behind(h) + B*ahead(h) + C*later_behind(h) + role_adjustments(h))). A, B, C are mutually exclusive line-position effects; the simpler same-line family substitutes one common term.",
        "race_count":len(allrows),
        "second_candidate_rows":int(a["M"].sum()),
        "result_conditioned_on_actual_first":True,
        "predictions_in_advance_require_sum_over_each_possible_first":True,
        "independent_first_q":"F1 S-class 12 period corrected foreign term999 unscaled + high-hinge calibration, pinned source",
        "benter_1994_gamma_081":"Reference fitted to Hong Kong racing: not transferred as keirin optimum",
        "dev_races":len(train),"chronology_races":len(check),
        "candidate_families":{},
        "notes":["Fit over entire eligible 14081 races is for empirical rule discovery. Fit on development and evaluate on later interval is a generalization diagnostic, not a globally virgin test.",
                 "No candidate is imposed as true before comparing residuals and predicted vs actual; no 3rd-place or wager model chosen.",
                 "Market odds not used. No claim of positive expected value.",
                 "same_line = directly_behind + ahead_of_winner + later_behind_winner; including all four in one equation would produce non-identifiable coefficients. Directed candidates use the three mutually exclusive directed categories WITHOUT a redundant same_line indicator.",
                 "For 2024-2025Q1 the 1st-win calibrator has seen development labels. Interpretation is retrospective descriptive."]
    }
    loss_check={}
    for name in MODELS:
        devb,cols=coefficients(name,a,train)
        allb,cols2=coefficients(name,a,allrows)
        eval_dev,_=evaluate(name,devb,a,train)
        eval_check,ll=evaluate(name,devb,a,check,calibration=True)
        eval_full,_=evaluate(name,allb,a,allrows,calibration=True)
        loss_check[name]=ll
        report["candidate_families"][name]={
            "coefficient_names":cols,
            "development_fitted_coefficients":[float(x) for x in devb],
            "all_14081_fitted_coefficients":[float(x) for x in allb],
            "development_fit":eval_dev,
            "chronological_check":eval_check,
            "all_races_description_in_sample":eval_full,
            "parameter_count":len(cols),
            "all_data_BIC":(2*len(allrows)*eval_full["conditional_second_log_loss"]
                            +len(cols)*math.log(len(allrows)))
        }
        print("SECOND_FIT",name,"coefs",list(zip(cols,np.round(devb,5))),
              "check_ll",round(eval_check["conditional_second_log_loss"],8),flush=True)
    # Compare changes on the same 7688 race IDs with block-by-day bootstrap.
    report["pairwise_check_improvement_over_harville"]={}
    dates=[a["date"][k] for k in check]
    for name in MODELS:
        if name=="harville":continue
        report["pairwise_check_improvement_over_harville"][name]=pair_bootstrap(
            loss_check["harville"],loss_check[name],dates)
    report["complexity_ranking_all_data_BIC"]=sorted(MODELS,key=lambda name:report["candidate_families"][name]["all_data_BIC"])
    report["chronology_logloss_ranking"]=sorted(MODELS,key=lambda name:report["candidate_families"][name]["chronological_check"]["conditional_second_log_loss"])
    report["not_automatic_adoption"]=True
    report["fixed_first_model_unmodified"]=True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SECOND_MODEL_REPORT_SAVED",json.dumps({
        "all_races":len(allrows),"train":len(train),"check":len(check),
        "rank_by_check":report["chronology_logloss_ranking"],
        "rank_by_BIC":report["complexity_ranking_all_data_BIC"]
    }),flush=True)

if __name__=="__main__":main()
