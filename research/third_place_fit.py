#!/usr/bin/env python3
"""Conditional third position model families, selected only after observed 3rd census.

P(third=k | first=i, second=j) = softmax(delta log q_k + positional correction).
Compare Harville, Benter horse-racing delta, F1 S-specific exponent, memberships,
exclusive directional positions, and optional rider-line-role adjustment.

No role or direction coefficient copied from original second-place research.
"""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from third_place_empirical import classify

DIRECTION_CLASSES=(
 "ahead_of_both", "between_first_second", "behind_both",
 "ahead_of_first", "immediately_behind_first", "further_behind_first",
 "ahead_of_second", "immediately_behind_second", "further_behind_second")
MEMBERSHIP_CLASSES=("line_of_both","first_line_only","second_line_only")
ROLE_CLASSES=("second","third_or_later","solo")
COLUMNS=("log_q",)+MEMBERSHIP_CLASSES+DIRECTION_CLASSES+tuple("role_"+s for s in ROLE_CLASSES)
FAMILIES={
 "harville":("log_q",),
 "benter_horse_delta_065":("log_q",),
 "estimated_delta":("log_q",),
 "delta_plus_line_ownership":("log_q",)+MEMBERSHIP_CLASSES,
 "delta_plus_exact_position":("log_q",)+DIRECTION_CLASSES,
 "delta_plus_exact_position_and_role":("log_q",)+DIRECTION_CLASSES+tuple("role_"+s for s in ROLE_CLASSES),
}
MINP=1e-15

def read(path):
    source=[json.loads(s) for s in path.open(encoding="utf-8") if s.strip()]
    source.sort(key=lambda x:(x["race_date"],x["race_id"]))
    if len(source)!=14081:raise RuntimeError("Unexpected number of races")
    nr=len(source);c=len(COLUMNS)
    X=np.zeros((nr,9,c),dtype=float)
    M=np.zeros((nr,9),dtype=bool)
    y=np.zeros(nr,dtype=int)
    fields=Counter();dates=[];periods=[];splits=[];ids=[];config=[]
    for ridx,r in enumerate(source):
        q={int(x["car_no"]):float(x["adopted_q"]) for x in r["first_win"]}
        cars=sorted(q);i=int(r["first"]);j=int(r["second"]);third=int(r["third"])
        assert len({i,j,third})==3 and {i,j,third}.issubset(q)
        pos={}
        for line in r["predicted_line_formation"]:
            for n,car in enumerate(line):
                assert car not in pos
                pos[car]=(tuple(line),n)
        assert set(pos)==set(q)
        fields[len(cars)]+=1
        same_12=(pos[i][0]==pos[j][0])
        config.append("same" if same_12 else "different")
        for u,car in enumerate(cars):
            rel,direction,role=classify(car,i,j,pos)
            M[ridx,u]=car not in (i,j)
            row=[math.log(max(q[car],MINP))]
            row += [float(rel==k) for k in MEMBERSHIP_CLASSES]
            row += [float(direction==k) for k in DIRECTION_CLASSES]
            row += [float(role==k) for k in ROLE_CLASSES]
            assert len(row)==c
            X[ridx,u]=row
        y[ridx]=cars.index(third)
        assert M[ridx,y[ridx]]
        ids.append(r["race_id"]);dates.append(r["race_date"])
        periods.append(r["period"]);splits.append(r["split_provenance_only"])
    if int(M.sum())!=69300:raise RuntimeError("Third candidate count changed")
    return dict(X=X,M=M,y=y,ids=ids,dates=dates,periods=periods,
                splits=splits,config=config,field_sizes=dict(fields))

def subsetidx(name):return [COLUMNS.index(k) for k in FAMILIES[name]]

def estimate(name,a,selection):
    if name=="harville":return np.array([1.0])
    if name=="benter_horse_delta_065":return np.array([.65])
    sub=subsetidx(name)
    X=a["X"][selection][:,:,sub]
    M=a["M"][selection]
    yy=a["y"][selection];rr=np.arange(len(selection))
    def fun(beta):
        v=X@beta
        v=np.where(M,v,-1.e8)
        norm=logsumexp(v,axis=1)
        nll=np.mean(norm-v[rr,yy])
        pp=np.exp(v-norm[:,None])
        grad=np.mean((pp[:,:,None]*X).sum(axis=1)-X[rr,yy,:],axis=0)
        return nll,grad
    start=np.zeros(len(sub));start[0]=1.0
    bounds=[(.05,2.5)]+[(-6,6)]*(len(sub)-1)
    res=minimize(fun,start,jac=True,method="L-BFGS-B",bounds=bounds,
                 options={"maxiter":600,"ftol":1e-13,"gtol":1e-9})
    if not np.isfinite(res.fun) or not np.isfinite(res.x).all():
        raise RuntimeError(f"Nonfinite MLE for {name}")
    return res.x

def predict_matrix(name,b,a,selection):
    M=a["M"][selection];F=a["X"][selection]
    scores=(F[:,:,0]*b[0] if name in ("harville","benter_horse_delta_065")
            else F[:,:,subsetidx(name)]@b)
    scores=np.where(M,scores,-1.e8)
    return np.exp(scores-logsumexp(scores,axis=1)[:,None])

def metrics(name,b,a,selection,detail=False):
    pr=predict_matrix(name,b,a,selection)
    row=np.arange(len(selection));yy=a["y"][selection]
    loss=-np.log(np.maximum(pr[row,yy],MINP))
    met={
      "races":len(selection),
      "third_candidates":int(a["M"][selection].sum()),
      "conditional_third_mean_log_loss":float(loss.mean()),
      "uniform_log_loss":float(np.mean(np.log(a["M"][selection].sum(axis=1)))),
      "best_candidate_3rd_hit":float((np.argmax(pr,axis=1)==yy).mean()),
      "mean_brier":float(np.mean(np.sum((pr-(np.arange(9)[None,:]==yy[:,None]))**2,axis=1)))
    }
    if detail:
        met["relation_calibration"]=by_group(pr,a,selection,MEMBERSHIP_CLASSES,"membership")
        met["direction_calibration"]=by_group(pr,a,selection,DIRECTION_CLASSES,"direction")
        met["candidate_role_calibration"]=by_group(pr,a,selection,ROLE_CLASSES,"role")
        met["first_second_same_vs_different"]={}
        for x in ("same","different"):
            cases=np.array([i for i in selection if a["config"][i]==x],dtype=int)
            yprob,_=metrics(name,b,a,cases,detail=False)
            met["first_second_same_vs_different"][x]=yprob
        met["per_period"]={}
        for period in sorted({a["periods"][i] for i in selection}):
            ix=np.array([i for i in selection if a["periods"][i]==period],dtype=int)
            po,_=metrics(name,b,a,ix)
            met["per_period"][period]={k:po[k] for k in ("races","conditional_third_mean_log_loss","best_candidate_3rd_hit")}
    return met,loss

def by_group(pr,a,selection,classes,kind):
    F=a["X"][selection];M=a["M"][selection]
    n=len(selection);y=a["y"][selection]
    hit=np.zeros_like(M,dtype=bool);hit[np.arange(n),y]=True
    result={}
    for k in classes:
        col=("role_"+k if kind=="role" else k)
        matched=M&(F[:,:,COLUMNS.index(col)]>.5)
        pred=float(pr[matched].sum())
        actual=int(hit[matched].sum())
        result[k]={"candidate_rows":int(matched.sum()),
                   "predicted_3rd":pred,"observed_3rd":actual,
                   "predicted_minus_observed":pred-actual}
    if kind in ("membership","direction"):
        matched=M&(~np.any(F[:,:, [COLUMNS.index(k) for k in classes]]>.5,axis=2))
        result["neither_line"]={"candidate_rows":int(matched.sum()),
                                "predicted_3rd":float(pr[matched].sum()),
                                "observed_3rd":int(hit[matched].sum())}
    return result

def day_ci(loss_base,loss_new,days,rep=2000):
    group=defaultdict(list)
    for x,y,d in zip(loss_base,loss_new,days):group[d].append(float(x-y))
    keys=sorted(group)
    su=np.array([sum(group[k]) for k in keys])
    num=np.array([len(group[k]) for k in keys])
    rng=np.random.default_rng(20261018)
    ix=rng.integers(len(keys),size=(rep,len(keys)))
    bootstrap=su[ix].sum(axis=1)/num[ix].sum(axis=1)
    return {"mean_improvement_positive_better":float(su.sum()/num.sum()),
            "day_cluster_95_interval":[float(x) for x in np.quantile(bootstrap,[.025,.975])],
            "number_of_days":len(keys)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--empirical",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    empirical=json.loads(args.empirical.read_text(encoding="utf-8"))
    if empirical["races"]!=14081 or empirical["third_candidate_rows"]!=69300:
        raise RuntimeError("Must observe third results before third MLE")
    a=read(args.input)
    allidx=np.arange(len(a["ids"]),dtype=int)
    fit=np.array([i for i,x in enumerate(a["splits"]) if x=="fit"],dtype=int)
    chronological=np.array([i for i,x in enumerate(a["splits"]) if x=="chronological_check"],dtype=int)
    if len(fit)!=6393 or len(chronological)!=7688:raise RuntimeError("Sample split changed")
    out={
      "status":"THIRD_CONDITIONAL_MLE_COMPARED_AFTER_REAL_RESULT_CENSUS",
      "scope":"F1 S級 only",
      "races":len(allidx),"candidate_rows":int(a["M"].sum()),"field_sizes":a["field_sizes"],
      "equation":"P(k third | i first, j second)=q_k^delta * exp(position_effect(i,j,k)) / sum_{h not in (i,j)} q_h^delta * exp(position_effect(i,j,h))",
      "first_second_actual_used_only_for_historical_conditional_study":True,
      "pre_race_requires_marginalize_all_first_second":True,
      "baseline":"Harville conditional third q_k/(1-q_i-q_j)",
      "benter_delta_065":"Horse-racing reference only; do not transfer as Keirin coefficient.",
      "family_features":{n:list(c) for n,c in FAMILIES.items()},
      "coefficient_identifiability":"Ownership and exclusive direction terms are separate nested models, not combined; role may be included with direction as an independent incremental question.",
      "discovery_all_races":14081,
      "chronological_fit_races":len(fit),"chronological_evaluation_races":len(chronological),
      "models":{},"pairwise_daily_block_bootstrap":{},
      "source_note":"Historical lineup snapshots not certified predeadline; same research had inspected later dates elsewhere."
    }
    losses={}
    for name in FAMILIES:
        devb=estimate(name,a,fit)
        fullb=estimate(name,a,allidx)
        dev,_=metrics(name,devb,a,fit)
        check,ll=metrics(name,devb,a,chronological,detail=True)
        full,_=metrics(name,fullb,a,allidx,detail=True)
        losses[name]=ll
        out["models"][name]={
            "coefficient_names":list(FAMILIES[name]),
            "fitted_to_6393":[float(x) for x in devb],
            "fitted_to_all_14081":[float(x) for x in fullb],
            "development_in_sample":dev,
            "later_7688_check":check,
            "full_cohort_descriptive_in_sample":full,
            "all_cohort_BIC":2*len(allidx)*full["conditional_third_mean_log_loss"]+len(fullb)*math.log(len(allidx)),
        }
        print("THIRD_FIT",name,"dev_coeff",list(zip(FAMILIES[name],np.round(devb,5))),
              "later_logloss",round(check["conditional_third_mean_log_loss"],8),flush=True)
    comparisons=(
        ("estimated_delta","harville"),
        ("delta_plus_line_ownership","estimated_delta"),
        ("delta_plus_exact_position","delta_plus_line_ownership"),
        ("delta_plus_exact_position_and_role","delta_plus_exact_position"),
        ("delta_plus_exact_position","harville"),
    )
    days=[a["dates"][i] for i in chronological]
    for new,base in comparisons:
        out["pairwise_daily_block_bootstrap"][new+"_OVER_"+base]=day_ci(losses[base],losses[new],days)
    out["rank_by_later_log_loss"]=sorted(FAMILIES,key=lambda k:out["models"][k]["later_7688_check"]["conditional_third_mean_log_loss"])
    out["rank_by_all_cohort_BIC"]=sorted(FAMILIES,key=lambda k:out["models"][k]["all_cohort_BIC"])
    out["not_automatic_model_adoption"]=True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("THIRD_RESULTS_DONE",json.dumps({"races":len(allidx),"model_ranking":out["rank_by_later_log_loss"],
          "bootstrap":out["pairwise_daily_block_bootstrap"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
