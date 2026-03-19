#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.special import ndtr, logit as logit_fn, expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler

ICU_ABBREV = {0: "CCU", 1: "CSRU", 2: "MICU", 3: "SICU"}
ICU_FULL   = {0: "Cardiology / coronary care",
               1: "Cardiac surgery recovery",
               2: "Medical ICU",
               3: "Surgical ICU"}

GBM_PARAMS = dict(
    learning_rate    = 0.03,
    max_depth        = 3,
    min_samples_leaf = 300,
    max_iter         = 450,
    random_state     = 42,
    early_stopping   = False,
    l2_regularization= 25.0,
)
GLM_C = 0.005


def load_outcomes(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with path.open() as f:
        next(f)
        for ln in f:
            parts = ln.strip().split(",")
            out[int(parts[0])] = int(parts[5])
    return out


def labels_for_ids(record_ids: np.ndarray, outcomes: dict) -> np.ndarray:
    return np.array([outcomes[int(r)] for r in record_ids], dtype=int)


def load_cached_features(set_name: str, cache_dir: Path):
    cache = cache_dir / f"{set_name}_features_0257.npz"
    if not cache.exists():
        return None
    d = np.load(str(cache))
    return d["record_ids"], d["icu_idx"], d["X"]


def impute_features(X: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X) | np.isinf(X), 0.0, X)


def icu_onehot(idx: np.ndarray) -> np.ndarray:
    oh = np.zeros((len(idx), 4), dtype=float)
    for i, d in enumerate(idx):
        oh[i, int(d)] = 1.0
    return oh


def load_lm(path: Path) -> dict:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    lm = {k: v for k, v in mat.items() if not k.startswith("__")}
    return {
        "Condit":     lm["Condit"],
        "m":          np.asarray(lm["m"]).reshape(-1),
        "beta":       np.asarray(lm["beta"],    dtype=float).reshape(-1),
        "prob_th":    np.asarray(lm["prob_th"], dtype=float).reshape(-1),
        "use_probit": bool(lm["use_probit"]),
    }


def _lsvm_score(model, feat_row: np.ndarray) -> float:
    x    = feat_row.copy(); x[np.isnan(x)] = 0.0
    sv   = np.asarray(model.SVs,     dtype=float)
    coef = np.asarray(model.sv_coef, dtype=float).reshape(-1)
    rho  = float(model.rho)
    g    = float(model.G)
    deg  = int(model.degree)
    dot  = x @ sv.T
    k    = np.power(g * dot + 1.0, deg) if str(model.kernel).lower() == "poly" else dot
    return float(k @ coef - rho)


def compute_paper_probs(X: np.ndarray, lm: dict,
                         svm_cache: np.ndarray | None = None) -> np.ndarray:
    if svm_cache is not None:
        svm_scores = svm_cache
    else:
        n_svms = len(lm["m"])
        svm_scores = np.empty((len(X), n_svms), dtype=float)
        for i, feat in enumerate(X):
            feat_imp = np.nan_to_num(feat, nan=0.0)
            row = np.array([_lsvm_score(m, feat_imp) for m in lm["m"]], dtype=float)
            svm_scores[i] = np.sort(row)
    beta = lm["beta"]
    probs = np.empty(len(svm_scores))
    for i, scores in enumerate(svm_scores):
        feat_glm = np.concatenate(([1.0], scores))
        p_lin = float(feat_glm @ beta)
        probs[i] = float(ndtr(p_lin)) if lm["use_probit"] else expit(p_lin)
    return np.clip(probs, 0.001, 0.999)


def train_backbone(X_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def backbone_probs(clf: HistGradientBoostingClassifier,
                   X: np.ndarray) -> np.ndarray:
    return np.clip(clf.predict_proba(X)[:, 1], 0.001, 0.999)


def train_routing_glm(
    backbone_p_train: np.ndarray,
    icu_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    logit_p = np.clip(logit_fn(backbone_p_train), -10, 10).reshape(-1, 1)
    X_glm = np.hstack([logit_p, icu_onehot(icu_train)])
    clf = LogisticRegression(C=GLM_C, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_glm, y_train)
    return clf


def routing_probs(
    glm: LogisticRegression,
    backbone_p: np.ndarray,
    icu: np.ndarray,
) -> np.ndarray:
    logit_p = np.clip(logit_fn(backbone_p), -10, 10).reshape(-1, 1)
    X_glm = np.hstack([logit_p, icu_onehot(icu)])
    return np.clip(glm.predict_proba(X_glm)[:, 1], 0.001, 0.999)


def tune_threshold(y: np.ndarray, p: np.ndarray) -> float:
    cands = np.unique(np.concatenate([
        np.quantile(p, np.linspace(0.001, 0.999, 5000)),
        np.linspace(p.min(), p.max(), 5000),
    ]))
    best_t, best_s = 0.5, -1.0
    for t in cands:
        pred = p > t
        tp = np.sum((pred == 1) & (y == 1))
        fn = np.sum((pred == 0) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        s = min(tp / (tp + fn + 1e-12), tp / (tp + fp + 1e-12))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def write_outputs(path: Path, ids: np.ndarray, probs: np.ndarray,
                  threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rid, p in zip(ids, probs):
            f.write(f"{int(rid)},{int(p > threshold)},{p:.6f}\n")


def write_paper_outputs(path: Path, ids: np.ndarray, icu_idx: np.ndarray,
                         probs: np.ndarray, prob_th: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rid, dep, p in zip(ids, icu_idx, probs):
            f.write(f"{int(rid)},{int(p > prob_th[int(dep)])},{p:.6f}\n")


def compile_score(root: Path) -> Path:
    score_bin = root / "score"
    if not score_bin.exists():
        print("Compiling score.c ...")
        subprocess.run(["gcc", "-std=gnu89", "-O2", "-o", "score", "score.c", "-lm"],
                        cwd=root, check=True)
    return score_bin


def run_score(score_bin: Path, outputs: Path, outcomes: Path) -> tuple[float, float]:
    p = subprocess.run([str(score_bin), str(outputs), str(outcomes)],
                        capture_output=True, text=True, check=True)
    txt = p.stdout + "\n" + p.stderr
    m1 = re.search(r"Unofficial Event 1 score:\s*([0-9.]+)", txt)
    m2 = re.search(r"Unofficial Event 2 score:\s*([0-9.]+)", txt)
    if not (m1 and m2):
        raise RuntimeError(f"Score binary parse error:\n{txt}")
    return float(m1.group(1)), float(m2.group(1))


def print_dept_table(set_label: str, y: np.ndarray,
                      p: np.ndarray, icu_idx: np.ndarray) -> None:
    print(f"\nDepartment-routed model on {set_label}, by ICU type:")
    print(f"  {'Abbrev':<6} {'Department':<30} {'n':>5} {'Mort%':>7} {'AUC':>7} {'Brier':>7}")
    print("  " + "-" * 62)
    for dep in range(4):
        m = icu_idx == dep
        yi, pi = y[m], p[m]
        n = int(m.sum())
        if n == 0:
            continue
        mort  = 100.0 * float(yi.mean())
        auc   = float(roc_auc_score(yi, pi)) if len(np.unique(yi)) == 2 else float("nan")
        brier = float(np.mean((pi - yi) ** 2))
        print(f"  {ICU_ABBREV[dep]:<6} {ICU_FULL[dep]:<30} {n:>5} {mort:>6.1f}% {auc:>7.3f} {brier:>7.3f}")


def plot_roc(y_b, pb_dept, pb_paper,
             y_c, pc_dept, pc_paper, out: Path) -> None:
    PAPER_E1 = {"Set B": 0.5200, "Set C": 0.5345}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("ROC Curves", fontsize=13)
    for ax, (y, p_dept, p_paper, title) in zip(axes, [
        (y_b, pb_dept, pb_paper, "Set B"),
        (y_c, pc_dept, pc_paper, "Set C"),
    ]):
        fpr_d, tpr_d, _ = roc_curve(y, p_dept)
        fpr_p, tpr_p, _ = roc_curve(y, p_paper)
        ax.plot(fpr_d, tpr_d, color="tab:orange", lw=2,
                label=f"Dept-routed (AUC={roc_auc_score(y, p_dept):.3f})")
        ax.plot(fpr_p, tpr_p, color="tab:blue", lw=2, linestyle="--",
                label=f"Paper ref (E1={PAPER_E1[title]:.4f}, AUC={roc_auc_score(y, p_paper):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150); plt.close()
    print(f"  -> {out}")


def plot_calibration(y_b, pb_dept, pb_paper,
                     y_c, pc_dept, pc_paper,
                     out: Path, n_bins: int = 10) -> None:
    PAPER_E2 = {"Set B": 13.54, "Set C": 17.88}
    edges = np.linspace(0, 1, n_bins + 1)

    def cal_curve(y, p):
        xp, yp = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() > 0:
                xp.append(p[m].mean()); yp.append(y[m].mean())
        return np.array(xp), np.array(yp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Calibration Curves", fontsize=13)
    for ax, (y, p_dept, p_paper, title) in zip(axes, [
        (y_b, pb_dept, pb_paper, "Set B"),
        (y_c, pc_dept, pc_paper, "Set C"),
    ]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
        xd, yd = cal_curve(y, p_dept)
        xp, yp = cal_curve(y, p_paper)
        ax.plot(xd, yd, "o-", color="tab:orange", lw=2,
                label=f"Dept-routed (Brier={brier_score_loss(y, p_dept):.4f})")
        ax.plot(xp, yp, "o-", color="tab:blue", lw=2, linestyle="--",
                label=f"Paper ref (E2={PAPER_E2[title]:.2f}, Brier={brier_score_loss(y, p_paper):.4f})")
        ax.set(xlabel="Mean predicted probability",
               ylabel="Observed event rate",
               title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150); plt.close()
    print(f"  -> {out}")


def decision_curve(y: np.ndarray, p: np.ndarray,
                   thresholds: np.ndarray) -> np.ndarray:
    n = len(y)
    nb = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = p > t
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        nb[i] = (tp / n) - (fp / n) * (t / (1.0 - t + 1e-12))
    return nb


def plot_dca(y_b, pb_dept, pb_paper,
             y_c, pc_dept, pc_paper, out: Path) -> None:
    thresholds = np.linspace(0.01, 0.50, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Decision Curve Analysis", fontsize=13)
    for ax, (y, p_dept, p_paper, title) in zip(axes, [
        (y_b, pb_dept, pb_paper, "Set B"),
        (y_c, pc_dept, pc_paper, "Set C"),
    ]):
        prev = y.mean()
        nb_all   = prev - (1 - prev) * (thresholds / (1 - thresholds + 1e-12))
        nb_none  = np.zeros(len(thresholds))
        nb_dept  = decision_curve(y, p_dept,  thresholds)
        nb_paper = decision_curve(y, p_paper, thresholds)
        paper_label = {"Set B": "Paper ref (E1=0.5200, E2=13.54)",
                       "Set C": "Paper ref (E1=0.5345, E2=17.88)"}[title]
        ax.plot(thresholds, nb_dept,  color="tab:orange", lw=2, label="Dept-routed")
        ax.plot(thresholds, nb_paper, color="tab:blue",   lw=2, linestyle="--", label=paper_label)
        ax.plot(thresholds, nb_all,   color="grey",       lw=1, linestyle=":",  label="Treat all")
        ax.plot(thresholds, nb_none,  color="black",      lw=1, linestyle="-.", label="Treat none")
        ax.set(xlabel="Decision threshold probability",
               ylabel="Net benefit",
               title=title, xlim=(0, 0.5))
        ax.axhline(0, color="black", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150); plt.close()
    print(f"  -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Department-Routed ICU Mortality Model (PhysioNet 2012)")
    parser.add_argument("--data_dir", type=Path,
                        default=Path(__file__).parent / "data",
                        help="Directory containing data files and feature caches")
    parser.add_argument("--out_dir", type=Path,
                        default=Path(__file__).parent / "figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    out_dir  = args.out_dir.resolve()
    tmp_dir  = data_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_bin = compile_score(data_dir)

    print("Loading cached features ...")
    for name in ("set-a", "set-b", "set-c"):
        if load_cached_features(name, data_dir) is None:
            sys.exit(f"Missing: {data_dir}/{name}_features_0257.npz")

    ida, icua, Xa = load_cached_features("set-a", data_dir)
    idb, icub, Xb = load_cached_features("set-b", data_dir)
    idc, icuc, Xc = load_cached_features("set-c", data_dir)

    print("Loading outcomes ...")
    y_a = labels_for_ids(ida, load_outcomes(data_dir / "Outcomes-a.txt"))
    y_b = labels_for_ids(idb, load_outcomes(data_dir / "Outcomes-b.txt"))
    y_c = labels_for_ids(idc, load_outcomes(data_dir / "Outcomes-c.txt"))
    print(f"  Set A: {len(y_a)} patients, {y_a.sum()} deaths ({100*y_a.mean():.1f}%)")
    print(f"  Set B: {len(y_b)} patients, {y_b.sum()} deaths ({100*y_b.mean():.1f}%)")
    print(f"  Set C: {len(y_c)} patients, {y_c.sum()} deaths ({100*y_c.mean():.1f}%)")

    print("Preprocessing features ...")
    Xa_imp = impute_features(Xa); Xb_imp = impute_features(Xb); Xc_imp = impute_features(Xc)
    scaler = StandardScaler()
    scaler.fit(Xa_imp)
    Xa_s = scaler.transform(Xa_imp)
    Xb_s = scaler.transform(Xb_imp)
    Xc_s = scaler.transform(Xc_imp)

    lm_path = data_dir / "lm_feat_mis0.mat"
    paper_available = lm_path.exists()
    if paper_available:
        print("Loading Citi L, Barbieri R.  model ...")
        lm = load_lm(lm_path)
        svm_cache_path = data_dir / "lm_svm_scores.npz"
        if svm_cache_path.exists():
            c = np.load(str(svm_cache_path))
            svm_a, svm_b, svm_c = c["a"], c["b"], c["c"]
        else:
            svm_a = svm_b = svm_c = None
        pa_paper = compute_paper_probs(Xa, lm, svm_a)
        pb_paper = compute_paper_probs(Xb, lm, svm_b)
        pc_paper = compute_paper_probs(Xc, lm, svm_c)
    else:
        print("WARNING: lm_feat_mis0.mat not found — Citi L, Barbieri R.  model curves skipped.")
        pa_paper = pb_paper = pc_paper = None

    print(f"\nTraining GBM backbone on Set A ...")
    print(f"  lr={GBM_PARAMS['learning_rate']}, max_depth={GBM_PARAMS['max_depth']}, "
          f"min_samples_leaf={GBM_PARAMS['min_samples_leaf']}, "
          f"max_iter={GBM_PARAMS['max_iter']}, l2={GBM_PARAMS['l2_regularization']}")
    backbone = train_backbone(Xa_s, y_a)
    pa_back = backbone_probs(backbone, Xa_s)
    pb_back = backbone_probs(backbone, Xb_s)
    pc_back = backbone_probs(backbone, Xc_s)
    print(f"  Backbone AUC: Set B={roc_auc_score(y_b, pb_back):.3f} | Set C={roc_auc_score(y_c, pc_back):.3f}")

    print(f"\nTraining department routing GLM on Set A ...")
    print(f"  Input: [logit(backbone_prob), ICU_CCU, ICU_CSRU, ICU_MICU, ICU_SICU]  C={GLM_C}")
    glm = train_routing_glm(pa_back, icua, y_a)
    print(f"  GLM coefs: backbone={glm.coef_[0][0]:.4f}, ICU offsets={glm.coef_[0][1:].round(4)}")

    pa_dept = routing_probs(glm, pa_back, icua)
    pb_dept = routing_probs(glm, pb_back, icub)
    pc_dept = routing_probs(glm, pc_back, icuc)

    dept_th = tune_threshold(y_a, pa_dept)
    print(f"  Threshold (tuned on Set A): {dept_th:.4f}")

    out_dept_b = tmp_dir / "Outputs-b-dept-routed.txt"
    out_dept_c = tmp_dir / "Outputs-c-dept-routed.txt"
    write_outputs(out_dept_b, idb, pb_dept, dept_th)
    write_outputs(out_dept_c, idc, pc_dept, dept_th)
    dept_b_e1, dept_b_e2 = run_score(score_bin, out_dept_b, data_dir / "Outcomes-b.txt")
    dept_c_e1, dept_c_e2 = run_score(score_bin, out_dept_c, data_dir / "Outcomes-c.txt")

    paper_b_e1 = paper_b_e2 = paper_c_e1 = paper_c_e2 = None
    if paper_available:
        out_paper_b = tmp_dir / "Outputs-b-paper-ref.txt"
        out_paper_c = tmp_dir / "Outputs-c-paper-ref.txt"
        write_paper_outputs(out_paper_b, idb, icub, pb_paper, lm["prob_th"])
        write_paper_outputs(out_paper_c, idc, icuc, pc_paper, lm["prob_th"])
        paper_b_e1, paper_b_e2 = run_score(score_bin, out_paper_b, data_dir / "Outcomes-b.txt")
        paper_c_e1, paper_c_e2 = run_score(score_bin, out_paper_c, data_dir / "Outcomes-c.txt")

    print("\n" + "=" * 68)
    print("Results  (E1: higher is better | E2: lower is better)")
    print("-" * 68)
    print(f"  {'Model':<22} {'B E1':>8} {'B E2':>8} {'C E1':>8} {'C E2':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    if paper_available:
        print(f"  {'Citi L, Barbieri R. ':<22} {paper_b_e1:>8.4f} {paper_b_e2:>8.2f}"
              f" {paper_c_e1:>8.4f} {paper_c_e2:>8.2f}")
    print(f"  {'Department-routed':<22} {dept_b_e1:>8.4f} {dept_b_e2:>8.2f}"
          f" {dept_c_e1:>8.4f} {dept_c_e2:>8.2f}")
    print()
    print(f"  {'Target (beat paper ref)':<22} {'>0.5200':>8} {'<13.54':>8} {'>0.5345':>8} {'<17.88':>8}")

    if paper_available:
        print(f"\n  AUC paper ref : Set B={roc_auc_score(y_b, pb_paper):.3f} | Set C={roc_auc_score(y_c, pc_paper):.3f}")
    print(f"  AUC dept model: Set B={roc_auc_score(y_b, pb_dept):.3f} | Set C={roc_auc_score(y_c, pc_dept):.3f}")

    print_dept_table("Set B", y_b, pb_dept, icub)
    print_dept_table("Set C", y_c, pc_dept, icuc)

    if paper_available:
        print("\nGenerating figures ...")
        plot_roc(y_b, pb_dept, pb_paper, y_c, pc_dept, pc_paper,
                 out_dir / "figure_roc_curves.png")
        plot_calibration(y_b, pb_dept, pb_paper, y_c, pc_dept, pc_paper,
                         out_dir / "figure_calibration_curves.png")
        plot_dca(y_b, pb_dept, pb_paper, y_c, pc_dept, pc_paper,
                 out_dir / "figure_dca_curves.png")
        print(f"\nDone. Figures saved to: {out_dir}/")
    else:
        print("\nSkipping figures (Citi L, Barbieri R.  not available).")

    print(f"Prediction output files in: {tmp_dir}/")


if __name__ == "__main__":
    main()
