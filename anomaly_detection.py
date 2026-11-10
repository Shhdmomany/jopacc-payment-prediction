"""Anomaly gate used before any predictive-model or XAI call.

The model must be trained with exactly the same FEATURE_NAMES and order as the
classification/regression form.  See train_anomaly_model.py.
"""
import joblib
from pathlib import Path
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import numpy as np

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


scaler = StandardScaler()
# X = scaler.fit_transform(df[FEATURE_NAMES].astype(float))

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "prediction_schema_iso.pkl"
SCALER_PATH = BASE_DIR / "prediction_schema_scaler.pkl"
FEATURES_PATH = BASE_DIR / "prediction_schema_features.pkl"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

def _load_artifacts():
    missing = [str(path.name) for path in (MODEL_PATH, SCALER_PATH, FEATURES_PATH)
               if not path.exists()]
    if missing:
        raise RuntimeError(
            "The anomaly gate has not been trained. Run train_anomaly_model.py "
            f"first (missing: {', '.join(missing)})."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names




def validate_input(data, max_features=3):
    """Return an anomaly decision and a human-readable feature explanation.

    `data` is one numeric row in the exact form/model feature order.  The
    anomaly score is -decision_function: larger values mean more anomalous.
    """
    model, scaler, feature_names = _load_artifacts()
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape != (1, len(feature_names)):
        raise ValueError(
            f"Expected one row with {len(feature_names)} features; got {values.shape}."
        )

    scaled = scaler.transform(values)
    prediction = int(model.predict(scaled)[0])  # 1 normal, -1 anomalous
    score = float(-model.decision_function(scaled)[0])

    # StandardScaler puts every input on the same scale.  These deviations are
    # an explanation of the unusual entered values, not a claim of OCSVM
    # feature importance (which an RBF SVM does not expose).
    z_scores = scaled[0]
    feature_indices = np.argsort(np.abs(z_scores))[-max_features:][::-1]
    unusual_features = []
    for index in feature_indices:
        z_score = float(z_scores[index])
        direction = "above" if z_score >= 0 else "below"
        unusual_features.append({
            "feature": feature_names[index],
            "value": float(values[0, index]),
            "z_score": round(z_score, 2),
            "explanation": (
                f"{feature_names[index]} is {abs(z_score):.2f} standard deviations "
                f"{direction} the training-data mean."
            ),
        })

    return {
        "is_anomaly": prediction == -1,
        "label": "ANOMALOUS" if prediction == -1 else "NORMAL",
        "score": round(score, 4),
        "unusual_features": unusual_features,
    }
