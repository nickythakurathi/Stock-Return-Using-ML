import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

BASE = "/Users/nickythakurathi/Desktop/Dr. Ishita/Independent Study Dataset"
OUTDIR = f"{BASE}/WEEK 3(ML)/outputs"

TRAIN_END = pd.Timestamp("2016-12-31")
Y = "ret_q_next"

DATASETS = {
    "A_Accounting": f"{OUTDIR}/modelA_accounting_ml_ready.csv",
    "M_Market":     f"{OUTDIR}/modelM_market_ml_ready.csv",
    "B_Combined":   f"{OUTDIR}/modelB_combined_ml_ready.csv"
}

MODELS = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001, max_iter=20000),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
}


def make_model(name, obj):
    if name in ["Ridge", "Lasso", "ElasticNet"]:
        return Pipeline([("scaler", StandardScaler()), ("model", obj)])
    return obj


def plot_lasso(pipe, X_cols, title, save_path):
    coefs = pipe.named_steps["model"].coef_
    imp = pd.Series(np.abs(coefs), index=X_cols).sort_values(ascending=False)

    plt.figure()
    imp.sort_values().tail(min(20, len(X_cols))).plot(kind="barh")
    plt.title(title)
    plt.xlabel("|Standardized Coefficient|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rf_perm(model, X_test, y_test, X_cols, title, save_path):
    pi = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_cols).sort_values(ascending=False)

    plt.figure()
    imp.sort_values().tail(min(20, len(X_cols))).plot(kind="barh")
    plt.title(title)
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    for dname, path in DATASETS.items():
        df = pd.read_csv(path, parse_dates=["qdate"], low_memory=False)

        lo, hi = df[Y].quantile([0.01, 0.99])
        df[Y] = df[Y].clip(lo, hi)

        train = df[df["qdate"] <= TRAIN_END].copy()
        test  = df[df["qdate"] > TRAIN_END].copy()

        X_cols = [c for c in df.columns if c not in ["permno", "qdate", Y]]
        X_train, y_train = train[X_cols], train[Y]
        X_test, y_test   = test[X_cols], test[Y]

        lasso = make_model("Lasso", MODELS["Lasso"])
        rf    = make_model("RandomForest", MODELS["RandomForest"])

        lasso.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        plot_lasso(
            lasso, X_cols,
            f"{dname}: Lasso Feature Importance",
            f"{OUTDIR}/{dname}_Lasso_feature_importance.png"
        )

        plot_rf_perm(
            rf, X_test, y_test, X_cols,
            f"{dname}: RandomForest Permutation Importance (Test)",
            f"{OUTDIR}/{dname}_RandomForest_perm_importance.png"
        )

        print("Saved plots for:", dname)

    print("\nAll plots saved to:", OUTDIR)


if __name__ == "__main__":
    main()
