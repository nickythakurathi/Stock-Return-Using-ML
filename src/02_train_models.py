import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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


def main():
    results = []

    for dname, path in DATASETS.items():
        df = pd.read_csv(path, parse_dates=["qdate"], low_memory=False)

        lo, hi = df[Y].quantile([0.01, 0.99])
        df[Y] = df[Y].clip(lo, hi)

        train = df[df["qdate"] <= TRAIN_END].copy()
        test  = df[df["qdate"] > TRAIN_END].copy()

        X_cols = [c for c in df.columns if c not in ["permno", "qdate", Y]]
        X_train, y_train = train[X_cols], train[Y]
        X_test, y_test   = test[X_cols], test[Y]

        print("\nDataset:", dname, "Train:", len(train), "Test:", len(test), "Features:", len(X_cols))

        for mname, mobj in MODELS.items():
            model = make_model(mname, mobj)
            model.fit(X_train, y_train)

            pred_tr = model.predict(X_train)
            pred_te = model.predict(X_test)

            tr_r2 = r2_score(y_train, pred_tr)
            te_r2 = r2_score(y_test, pred_te)
            tr_mse = mean_squared_error(y_train, pred_tr)
            te_mse = mean_squared_error(y_test, pred_te)

            gap = tr_r2 - te_r2
            overfit = gap > 0.02

            results.append([dname, mname, tr_r2, te_r2, gap, tr_mse, te_mse, overfit])

            print(f"{mname:12s} TrainR2={tr_r2:.4f} TestR2={te_r2:.4f} Gap={gap:.4f} Overfit={overfit}")

    out = pd.DataFrame(results, columns=[
        "Dataset","Model","Train_R2","Test_R2","R2_Gap","Train_MSE","Test_MSE","Overfit_Flag"
    ])

    out = out.sort_values(["Dataset", "Test_R2"], ascending=[True, False])
    out.to_csv(f"{OUTDIR}/model_comparison_table.csv", index=False)

    print("\nSaved:", f"{OUTDIR}/model_comparison_table.csv")


if __name__ == "__main__":
    main()
