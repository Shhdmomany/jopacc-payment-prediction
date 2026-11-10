import joblib
import numpy as np
import shap
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import os

# Feature names used across the app
# temp features for both reg and class
FEATURE_NAMES = ["transactions_total",
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

# Get the directory this file lives in (so pickle paths work from anywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model and explainers ONCE at import time ──
 
_model_regression = joblib.load(os.path.join(BASE_DIR, "regression_model.pkl"))
_shap_data = joblib.load(
    os.path.join(BASE_DIR, "shap_regression_explainer.pkl")
)
 
_shap_explainer = _shap_data["explainer"]
FEATURE_NAMES = _shap_data["X_train"].columns.tolist()

# Load training data for LIME
_reg_data = joblib.load(os.path.join(BASE_DIR, "preprocessed_regression.pkl"))
X_train = np.array(_reg_data["X_train"])

_lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=FEATURE_NAMES,
    mode="regression",
    random_state=43,
)


def collect_data(
    transactions_total,
    returned_cheque_value_rate_pct,
    returned_cheque_value_jod,
    returned_cheque_count, 
    returned_cheque_count_rate_pct,
    off_us_money_transfer_count,
    customer_legal_entity_count,
    customer_individual_count,
    ef_cash_payment_count,
    ef_digital_payment_count,                   
    users_total,
    tx_cash_in_count,
    tx_cash_out_count,
    tx_purchases_count,
    tx_money_transfers_count
                    
):
    """Package form inputs into a numpy array. Order MUST match FEATURE_NAMES."""
    X = np.array([[
         transactions_total,
            returned_cheque_value_rate_pct,
            returned_cheque_value_jod,
            returned_cheque_count, 
            returned_cheque_count_rate_pct,
            off_us_money_transfer_count,
            customer_legal_entity_count,
            customer_individual_count,
            ef_cash_payment_count,
            ef_digital_payment_count,
            users_total,
            tx_cash_in_count,
            tx_cash_out_count,
            tx_purchases_count,
            tx_money_transfers_count
    ]], dtype=float)
    return X

def predict_regression(data):
    """Return the predicted transaction value."""
    prediction = _model_regression.predict(data)[0]
    return prediction

def explain_shap_regression(data):

    shap_values = _shap_explainer.shap_values(data)

    print(type(shap_values))

    if isinstance(shap_values, list):
        print("LIST")
        for i, s in enumerate(shap_values):
            print(i, s.shape)
    else:
        print("ARRAY")
        print(shap_values.shape)

    return []


def explain_shap_regression(data):
    """Return SHAP explanation as a list of dicts (easy to render in HTML)."""
    shap_values = _shap_explainer.shap_values(data)[0]
    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Feature Value": data[0],
        "SHAP Value": shap_values.round(4),
    })
    return shap_df.to_dict("records")   # list of dicts


def explain_lime_regression(data):
    """Return LIME explanation as a list of dicts."""
    exp = _lime_explainer.explain_instance(
        data[0],
        lambda x: _model_regression.predict(pd.DataFrame(x, columns=FEATURE_NAMES)),
    )
    lime_list = exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["Feature", "Contribution"])
    lime_df["Contribution"] = lime_df["Contribution"].round(4)
    return lime_df.to_dict("records")