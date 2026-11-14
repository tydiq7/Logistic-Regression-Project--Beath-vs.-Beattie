"""Beath vs. Beattie: whose per-half performance better predicts a UCSD win?

Evaluation uses game-grouped 5-fold cross-validation (both halves of a game always land
in the same fold) and compares every model against a majority-class baseline.

Two analyses:
1. The original model (both players' stats) vs. the baseline, with plots.
2. An ablation: models built from different groups of features (Leo only, Tom only,
   opponent strength, combinations), to see which features carry the predictive signal.
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
DATA_PATH = ROOT / "data" / "ucsd_halves_2025_26.csv"
FIG_DIR = ROOT / "Figures"

LEO = ["L_Points", "L_Rebounds", "L_Assists"]
TOM = ["T_Points", "T_Rebounds", "T_Assists"]
OPP = ["Opp_Strength"]
FEATURES = LEO + TOM

N_SPLITS = 5
SEED = 42
N_REPEATS = 30  # ablation: average over this many different fold shuffles

# La Verne (D3) and Stanton (NAIA) have no Division I record to compare. We give them a
# strength well below the weakest D1 opponent (Cal State Bakersfield, 0.267). This is an
# assumption, so the ablation is also re-run with those two games dropped.
NON_D1_STRENGTH = 0.20


# ----------------------------------------------------------------------------- data
def load_data(path=DATA_PATH):
    """Load the CSV, drop non-binary outcomes (ties), encode L=0 / W=1, add Opp_Strength.

    Opp_Strength is the opponent's season win % with all of its games vs. UCSD removed,
    so the outcome we're predicting is not baked into the feature.
    """
    df = pd.read_csv(path)
    df = df[df["Outcome"].isin(["W", "L"])].copy()
    df["Outcome"] = (df["Outcome"] == "W").astype(int)
    df["Opp_Strength"] = df["Opp_WinPct_exUCSD"].fillna(NON_D1_STRENGTH)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- models
def make_model():
    """Standardize each feature, then fit a logistic regression.

    Standardizing inside the pipeline means each CV fold learns its means/stds from its
    training rows only, so nothing about the held-out games leaks into the model.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0))


def make_baseline():
    """Always predict the training-set win rate (no skill by construction)."""
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


def cross_validate(make_estimator, X, y, groups, seed=SEED):
    """Game-grouped CV. Returns per-fold metrics and pooled out-of-fold probabilities."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
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


def repeated_auc(df, cols):
    """Mean CV AUC for a feature set, averaged over N_REPEATS different fold shuffles.

    One 5-fold split of 23 games is noisy, so we reshuffle the folds many times and
    average. This keeps one lucky (or unlucky) split from deciding the comparison.
    """
    aucs = [
        cross_validate(make_model, df[cols], df["Outcome"], df["GameID"], seed=s)[0]["AUC"].mean()
        for s in range(N_REPEATS)
    ]
    return np.mean(aucs), np.min(aucs), np.max(aucs)


def ablation(df):
    """Compare feature groups: which ones carry the signal, and what each adds."""
    feature_sets = {
        "Opponent strength only": OPP,
        "Leo only": LEO,
        "Tom only": TOM,
        "Leo + opponent": LEO + OPP,
        "Tom + opponent": TOM + OPP,
        "Both players": LEO + TOM,
        "Both players + opponent": LEO + TOM + OPP,
    }
    rows = []
    for name, cols in feature_sets.items():
        mean, lo, hi = repeated_auc(df, cols)
        rows.append({"Features": name, "Mean AUC": round(mean, 3), "Range": f"{lo:.2f}-{hi:.2f}"})
    return pd.DataFrame(rows).set_index("Features")


# ----------------------------------------------------------------------------- plotting
def save_fig(fig, name):
    """Save a figure to Figures/ as a PNG."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def plot_betas(betas):
    """Horizontal bar chart of the standardized coefficients."""
    ax = betas.sort_values().plot(kind="barh", title="Beta Values for Each Feature (standardized)", legend=False)
    ax.set_xlabel("Coefficient")
    save_fig(ax.figure, "Beta Values")


def plot_metrics_vs_baseline(model_folds, baseline_folds):
    """Bar chart of mean held-out metrics for the model vs. the baseline, with fold std as error bars."""
    means = pd.DataFrame({"Model": model_folds.mean(), "Baseline": baseline_folds.mean()})
    errs = pd.DataFrame({"Model": model_folds.std(ddof=1), "Baseline": baseline_folds.std(ddof=1)})
    ax = means.plot(kind="bar", yerr=errs, capsize=4, rot=0, figsize=(6, 4),
                    title="Held-out metrics (5-fold, game-grouped)")
    ax.set_ylabel("Score (log-loss: lower is better)")
    save_fig(ax.figure, "Metrics vs Baseline")


def plot_confusion(y, oof):
    """Confusion matrix of out-of-fold predictions at a 0.5 threshold."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y, oof >= 0.5), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix (out-of-fold)")
    save_fig(fig, "Confusion Matrix Heatmap")


def plot_roc(y, oof):
    """ROC curve of pooled out-of-fold predictions, with the no-skill diagonal for reference."""
    fpr, tpr, _ = roc_curve(y, oof)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="blue", label=f"Model, pooled out-of-fold (AUC = {roc_auc_score(y, oof):.2f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="No Skill")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve for UCSD Game Outcome Classification")
    ax.legend(loc="lower right")
    save_fig(fig, "ROC Curve")


def plot_ablation(table):
    """Bar chart of mean CV AUC for each feature set, with the 0.5 no-skill line."""
    ax = table["Mean AUC"].sort_values().plot(kind="barh", figsize=(7, 4), color="steelblue",
                                              title=f"Ablation: mean CV AUC over {N_REPEATS} fold shuffles")
    ax.axvline(0.5, color="red", linestyle="--", label="No skill")
    ax.set_xlabel("AUC")
    ax.legend(loc="lower right")
    save_fig(ax.figure, "Ablation")


# ----------------------------------------------------------------------------- main
def main():
    """Run the baseline comparison, the coefficients, the ablation, and save all figures."""
    df = load_data()
    X, y, groups = df[FEATURES], df["Outcome"], df["GameID"]
    print(f"{len(df)} halves from {groups.nunique()} games; win rate = {y.mean():.2f}")

    # 1. Original model (both players) vs. baseline
    model_folds, oof = cross_validate(make_model, X, y, groups)
    baseline_folds, _ = cross_validate(make_baseline, X, y, groups)
    print(f"\nHeld-out metrics, mean ± std over {N_SPLITS} game-grouped folds:")
    print(pd.DataFrame([summarize("Model", model_folds), summarize("Baseline", baseline_folds)]).to_string())

    # Coefficients come from a fit on all data and are for interpretation only, not evaluation.
    final = make_model().fit(X, y)
    betas = pd.Series(final[-1].coef_[0], index=FEATURES, name="beta")
    print("\nStandardized coefficients (full-data fit):")
    print(betas.sort_values(ascending=False).round(3).to_string())

    # 2. Ablation, plus a robustness check without the two non-D1 games
    table = ablation(df)
    print(f"\nAblation (all games; non-D1 opponents set to strength {NON_D1_STRENGTH}):")
    print(table.to_string())
    d1_only = df[df["Division"] == "D1"].reset_index(drop=True)
    print(f"\nRobustness check, D1 opponents only ({d1_only['GameID'].nunique()} games):")
    print(ablation(d1_only).to_string())

    # How related is opponent strength to each player's scoring? (near 0 = not confounded)
    print("\nCorrelation of opponent strength with:")
    print(df[["Outcome", "L_Points", "T_Points"]].corrwith(df["Opp_Strength"]).round(2).to_string())

    plot_betas(betas)
    plot_metrics_vs_baseline(model_folds, baseline_folds)
    plot_confusion(y, oof)
    plot_roc(y, oof)
    plot_ablation(table)
    plt.show()


if __name__ == "__main__":
    main()
