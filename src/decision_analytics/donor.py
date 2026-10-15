"""Compare proxy labels, subgroup transfer, and distance-constrained outreach."""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .common import ROOT, SEED, environment, model_pipeline, read_dataset, write_json
from .routing import build_route, distance_matrix, route_projection, select_threshold

FEATURES = ["age", "workclass", "education_level", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]
DEPOT = (31.1508, 121.4770)


def fit_stage(data, target, stage):
    x, y = data[FEATURES], data[target].astype(int)
    tr, va = train_test_split(np.arange(len(data)), test_size=.2, stratify=y, random_state=SEED)
    estimators = {"Logistic regression":LogisticRegression(max_iter=2000, random_state=SEED),
                  "Random forest":RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=2),
                  "Gradient boosting":GradientBoostingClassifier(n_estimators=200, random_state=SEED)}
    candidates, rows = {}, []
    for name, est in estimators.items():
        print(f"[donor] {stage}: {name}", flush=True)
        model = model_pipeline(x.iloc[tr], est)
        model.fit(x.iloc[tr], y.iloc[tr])
        p = model.predict_proba(x.iloc[va])[:,1]
        rows.append({"stage":stage, "model":name, "target":target, "split":"internal validation", "roc_auc":float(roc_auc_score(y.iloc[va],p)), "average_precision":float(average_precision_score(y.iloc[va],p)), "brier_score":float(brier_score_loss(y.iloc[va],p))})
        candidates[name] = model
    winner = sorted(rows, key=lambda r:(-r["roc_auc"],r["model"]))[0]["model"]
    fitted = candidates[winner]
    fitted.fit(x, y)
    split_ids = pd.DataFrame({"source_train_row":np.concatenate([data.index.to_numpy()[tr],data.index.to_numpy()[va]]), "stage":stage, "split":["fit"]*len(tr)+["internal validation"]*len(va)})
    return fitted, winner, rows, split_ids


def run():
    started=time.monotonic()
    folder=ROOT / "projects" / "donor-routing"
    results,figures=folder/"results",folder/"figures"
    results.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    train=read_dataset("donor-routing","train.csv")
    test=read_dataset("donor-routing","test.csv")
    known_train=train[(train.previously_contacted==1) & train.if_donated_last_year.notna()]
    m1,w1,r1,s1=fit_stage(train,"income_gt50k","Income proxy")
    m2,w2,r2,s2=fit_stage(known_train,"if_donated_last_year","Past donation")
    pd.DataFrame(r1+r2).to_csv(results/"model_selection_validation.csv",index=False)
    pd.concat([s1,s2],ignore_index=True).to_csv(results/"internal_splits.csv.gz",index=False,compression={"method":"gzip", "mtime":0})
    p1=m1.predict_proba(test[FEATURES])[:,1]
    p2=m2.predict_proba(test[FEATURES])[:,1]
    known=test.previously_contacted.to_numpy()==1
    hybrid=np.where(known,p2,p1)
    # This inherited hybrid rule is a retrospective hypothesis. Income scores
    # on cold prospects are not calibrated probabilities of future donation.
    scores={"Income proxy":p1,"Past donation":p2,"Hybrid heuristic":hybrid}
    rows=[]
    for name,p in scores.items():
        for group,mask in [("All",np.ones(len(test),bool)),("Known",known),("Cold",~known)]:
            yt=test.donated.to_numpy()[mask]
            rows.append({"score":name,"group":group,"n":int(mask.sum()),"future_donation_auc":float(roc_auc_score(yt,p[mask])),"future_donation_ap":float(average_precision_score(yt,p[mask]))})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(results/"future_donation_test.csv",index=False)
    sample=test.sample(n=200,random_state=SEED)
    positions=test.index.get_indexer(sample.index)
    routing_scores=hybrid[positions]
    points=sample[["lat","lon"]].to_numpy()
    distances=distance_matrix(points,DEPOT)
    threshold,combined,search=select_threshold(routing_scores,distances,15.,np.arange(.05,.55,.05))
    pd.DataFrame(search).to_csv(results/"threshold_search.csv",index=False)
    routes={"Top scores first":build_route(distances,15.,np.argsort(-routing_scores,kind="stable")+1,ordered=True),
            "Nearest first":build_route(distances,15.),"Threshold + nearest":combined}
    route_rows=[]
    def evaluate(name,route,budget):
        projected=route_projection(route,distances,routing_scores)
        ix=np.asarray(route[1:-1],dtype=int)-1
        donations=int(sample.donated.to_numpy()[ix].sum())
        return {"strategy":name,"budget_km":budget,**projected,"donations":donations,
                "historical_simulated_net":200*donations-5*len(ix)-10*projected["distance_km"]}
    for name,route in routes.items():
        row=evaluate(name,route,15.)
        assert row["distance_km"]<=15.+1e-8
        route_rows.append(row)
    pd.DataFrame(route_rows).to_csv(results/"routes_test.csv",index=False)
    sensitivity=[]
    for budget in [2.,5.,8.,10.,15.,20.,30.]:
        route=build_route(distances,budget,np.flatnonzero(routing_scores>=threshold)+1)
        row=evaluate("Threshold + nearest",route,budget)
        assert row["distance_km"]<=budget+1e-8
        sensitivity.append(row)
    pd.DataFrame(sensitivity).to_csv(results/"budget_sensitivity.csv",index=False)
    # Candidate IDs and coordinates are omitted from exports; bundled source
    # files retain their course-provided content and checksums.
    write_json(results/"routes.json",{name:[int(v) for v in route] for name,route in routes.items()})

    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    colors=["#8893a3","#d49537","#245b78"]
    for name,color in zip(scores,colors):
        part=metrics[(metrics.score==name) & (metrics.group!="All")]
        axes[0].plot(part.group,part.future_donation_auc,"o-",label=name,color=color)
    axes[0].set(ylabel="Future-donation ROC-AUC",title="Known vs cold prospects")
    axes[0].legend(fontsize=8)
    route=routes["Threshold + nearest"]
    all_points=np.vstack([DEPOT,points])
    axes[1].scatter(points[:,1],points[:,0],s=9,color="#c6ced5",label="Candidates")
    axes[1].plot(all_points[route,1],all_points[route,0],lw=.9,color="#245b78",label="Selected round trip")
    axes[1].scatter([DEPOT[1]],[DEPOT[0]],marker="*",s=100,color="#d49537",label="Depot")
    axes[1].set(xlabel="Longitude",ylabel="Latitude",title="Score filter + nearest-neighbor route")
    axes[1].ticklabel_format(axis="both", style="plain", useOffset=False)
    axes[1].legend(fontsize=7)
    fig.savefig(figures/"subgroups_and_route.svg",metadata={"Date":None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,3.8),layout="constrained")
    ss=pd.DataFrame(sensitivity)
    ax.plot(ss.budget_km,ss.historical_simulated_net,"o-",color="#245b78",label="Historical simulated net")
    ax.plot(ss.budget_km,ss.score_based_projected_net,"s--",color="#d49537",label="Score-based projection")
    ax.set(xlabel="Round-trip travel budget (km)",ylabel="Assumed monetary units",title="Travel budget and diminishing returns")
    ax.legend(fontsize=8)
    fig.savefig(figures/"budget_sensitivity.svg",metadata={"Date":None});plt.close(fig)
    write_json(results/"run.json",{"environment":environment(),"source_rows":{"train":len(train),"test":len(test),"known_train":len(known_train)},"features":FEATURES,"income_model":w1,"past_donation_model":w2,"selection":"maximum AUC on a stratified 20% internal training holdout; refit selected estimator on source training rows", "routing_candidates":200,"threshold":threshold,"depot":list(DEPOT),"distance":"Haversine great-circle distance, Earth radius 6371.0088 km; excludes road and travel-time constraints", "economics":{"donation_value":200,"visit_cost":5,"travel_cost_per_km":10},"elapsed_seconds":round(time.monotonic()-started,2),"evaluation_scope":"Course scenario; original test results previously inspected. Income proxy scores are not calibrated donation probabilities. Hybrid rule inherited from course analysis; no causal or untouched-holdout claim."})
    print(f"[donor] Complete: {w1}; {w2}; threshold={threshold:.2f}",flush=True)


if __name__=="__main__":
    run()
