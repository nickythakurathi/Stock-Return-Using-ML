import pandas as pd

BASE = "/Users/nickythakurathi/Desktop/Dr. Ishita/Independent Study Dataset"
OUTDIR = f"{BASE}/WEEK 3(ML)/outputs"

ACC_FILE = f"{BASE}/quarterly_Financial_Ratios_2000_2024.csv"
MKT_FILE = f"{BASE}/mkt_quarterly_2000_2024_with_DV.csv"

ACCT_VARS = [
    "roe","roa","npm","gpm","GProf",
    "de_ratio","debt_capital","cash_debt",
    "at_turn","inv_turn","invt_act","rect_act",
    "divyield"
]

MKT_VARS = ["VOL","SHROUT","ret_q","sprtrn","DLRET"]
Y = "ret_q_next"


def impute_by_quarter(df, cols, date_col="qdate"):
    df = df.copy()
    for c in cols:
        med = df.groupby(date_col)[c].transform("median")
        df[c] = df[c].fillna(med)
    return df


def main():
    acc = pd.read_csv(ACC_FILE, parse_dates=["qdate"], low_memory=False)
    mkt = pd.read_csv(MKT_FILE, parse_dates=["qdate"], low_memory=False)

    acc = acc[["permno", "qdate"] + ACCT_VARS].copy()
    mkt = mkt[["permno", "qdate"] + MKT_VARS + [Y]].copy()

    acc["divyield"] = acc["divyield"].astype(str).str.replace("%", "", regex=False)

    for c in ACCT_VARS:
        acc[c] = pd.to_numeric(acc[c], errors="coerce")

    for c in MKT_VARS + [Y]:
        mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    acc = impute_by_quarter(acc, ACCT_VARS)
    mkt = impute_by_quarter(mkt, MKT_VARS)

    mkt = mkt.dropna(subset=[Y]).copy()

    keys = acc[["permno", "qdate"]].merge(
        mkt[["permno", "qdate"]],
        on=["permno", "qdate"],
        how="inner"
    ).drop_duplicates()

    acc = keys.merge(acc, on=["permno", "qdate"], how="inner")
    mkt = keys.merge(mkt, on=["permno", "qdate"], how="inner")

    df_A = acc.merge(mkt[["permno", "qdate", Y]], on=["permno", "qdate"], how="inner")
    df_M = mkt.copy()
    df_B = acc.merge(mkt, on=["permno", "qdate"], how="inner")

    print("Common keys:", keys.shape)
    print("Model A:", df_A.shape, "Missing:", int(df_A.isna().sum().sum()))
    print("Model M:", df_M.shape, "Missing:", int(df_M.isna().sum().sum()))
    print("Model B:", df_B.shape, "Missing:", int(df_B.isna().sum().sum()))

    df_A.to_csv(f"{OUTDIR}/modelA_accounting_ml_ready.csv", index=False)
    df_M.to_csv(f"{OUTDIR}/modelM_market_ml_ready.csv", index=False)
    df_B.to_csv(f"{OUTDIR}/modelB_combined_ml_ready.csv", index=False)

    print("Saved datasets to:", OUTDIR)


if __name__ == "__main__":
    main()
