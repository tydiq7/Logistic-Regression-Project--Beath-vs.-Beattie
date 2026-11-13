"""Beath vs. Beattie: which player's per-half performance better predicts a UCSD win?

Evaluation uses game-grouped 5-fold cross-validation (both halves of a game always land
in the same fold) and compares the model against a majority-class baseline.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Paths are relative to this file, so the script runs from a fresh clone on any machine.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Logistic Regression Project Raw Data.csv"
FIG_DIR = ROOT / "Figures"

FEATURES = ["L_Points", "L_Rebounds", "L_Assists", "T_Points", "T_Rebounds", "T_Assists"]
N_SPLITS = 5
SEED = 42


# ----------------------------------------------------------------------------- data
def load_data(path=DATA_PATH):
    """Load the raw CSV, drop non-binary outcomes (ties) and encode L=0, W=1."""
    df = pd.read_csv(path)
    df = df[df["Outcome"].isin(["W", "L"])].copy()
    df["Outcome"] = (df["Outcome"] == "W").astype(int)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- models
def make_model():
    """Standardize each stat, then fit a logistic regression.

    Standardizing inside the pipeline means each CV fold learns its means/stds from its
    training rows only, so nothing about the held-out games leaks into the model.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=0, C=1.0))


def make_baseline():
    """Always predicts the training-set class prior (no skill by construction)."""
    return DummyClassifier(strategy="prior")


# ----------------------------------------------------------------------------- evaluation
def score_predictions(y_true, proba):
    """AUC, accuracy and log-loss for one set of held-out predictions."""
    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "AUC": auc,
        "Accuracy": accuracy_score(y_true, proba >= 0.5),
        "Log-loss": log_loss(y_true, proba, labels=[0, 1]),
    }


def cross_validate(make_estimator, X, y, groups):
    """Game-grouped CV. Returns per-fold metrics and pooled out-of-fold probabilities."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_rows = []
    for train_idx, test_idx in cv.split(X, y, groups):
        est = make_estimator().fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[test_idx] = est.predict_proba(X.iloc[test_idx])[:, 1]
        fold_rows.append(score_predictions(y.iloc[test_idx], oof[test_idx]))
    return pd.DataFrame(fold_rows), oof


def summarize(name, fold_df):
    """Mean and std across folds, as 'mean ± std' strings in a one-row frame."""
    mean, std = fold_df.mean(), fold_df.std(ddof=1)
    return pd.Series({m: f"{mean[m]:.3f} ± {std[m]:.3f}" for m in fold_df.columns}, name=name)


# ----------------------------------------------------------------------------- plotting
def save_fig(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_betas(betas):
    ax = betas.sort_values().plot(kind="barh", title="Beta Values for Each Feature (standardized)", legend=False)
    ax.set_xlabel("Coefficient")
    save_fig(ax.figure, "Beta Values")


def plot_metrics_vs_baseline(model_folds, baseline_folds):
    means = pd.DataFrame({"Model": model_folds.mean(), "Baseline": baseline_folds.mean()})
    errs = pd.DataFrame({"Model": model_folds.std(ddof=1), "Baseline": baseline_folds.std(ddof=1)})
    ax = means.plot(kind="bar", yerr=errs, capsize=4, rot=0, figsize=(6, 4),
                    title="Held-out metrics (5-fold, game-grouped)")
    ax.set_ylabel("Score (log-loss: lower is better)")
    save_fig(ax.figure, "Metrics vs Baseline")


def plot_confusion(y, oof):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y, oof >= 0.5), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix (out-of-fold)")
    save_fig(fig, "Confusion Matrix Heatmap")


def plot_roc(y, oof):
    fpr, tpr, _ = roc_curve(y, oof)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"Model, pooled out-of-fold (AUC = {roc_auc_score(y, oof):.2f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="No Skill")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve for UCSD Game Outcome Classification")
    ax.legend(loc="lower right")
    save_fig(fig, "ROC Curve")


# ----------------------------------------------------------------------------- main
def main():
    df = load_data()
    X, y, groups = df[FEATURES], df["Outcome"], df["GameID"]
    print(f"{len(df)} halves from {groups.nunique()} games; win rate = {y.mean():.2f}")

    model_folds, oof = cross_validate(make_model, X, y, groups)
    baseline_folds, _ = cross_validate(make_baseline, X, y, groups)

    print(f"\nHeld-out metrics, mean ± std over {N_SPLITS} game-grouped folds:")
    print(pd.DataFrame([summarize("Model", model_folds), summarize("Baseline", baseline_folds)]).to_string())

    # Coefficients come from a fit on all data and are for interpretation only, not evaluation.
    final = make_model().fit(X, y)
    betas = pd.Series(final[-1].coef_[0], index=FEATURES, name="beta")
    print("\nStandardized coefficients (full-data fit):")
    print(betas.sort_values(ascending=False).round(3).to_string())

    plot_betas(betas)
    plot_metrics_vs_baseline(model_folds, baseline_folds)
    plot_confusion(y, oof)
    plot_roc(y, oof)
    plt.show()


if __name__ == "__main__":
    main()