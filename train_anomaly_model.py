from pathlib import Path
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "jopacc_system_month_november_2022_to_june_2026.csv"
 
FEATURE_NAMES =  ["transactions_total",
                "returned_cheque_value_rate_pct",
                "returned_cheque_value_jod",
                "returned_cheque_count", 
                "returned_cheque_count_rate_pct",
                "off_us_money_transfer_count",
                "customer_legal_entity_count",
                "customer_individual_count",
                "ef_cash_payment_count",
                "ef_digital_payment_count",
                "users_total",
                "tx_cash_in_count",
                "tx_cash_out_count",
                "tx_purchases_count",
                "tx_money_transfers_count"
                ]



def main():
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURE_NAMES) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required features: {missing}")
    if df[FEATURE_NAMES].isna().any().any():
        raise ValueError("Fill or remove missing values before training the anomaly model.")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_NAMES].astype(float))
    model = OneClassSVM(kernel="rbf", nu=0.03, gamma="scale")
    model.fit(X)

    for filename, artifact in (
        ("prediction_schema_ocsvm.pkl", model),
        ("prediction_schema_scaler.pkl", scaler),
        ("prediction_schema_features.pkl", FEATURE_NAMES),
    ):
        with (BASE_DIR / filename).open("wb") as file:
            pickle.dump(artifact, file)
    print("Saved anomaly gate artifacts for the /predict schema.")


if __name__ == "__main__":
    main()
