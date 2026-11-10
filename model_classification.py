import joblib
import numpy as np
import shap
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import os

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

_model = joblib.load(os.path.join(BASE_DIR, "classification_model.pkl"))
_shap_data = joblib.load(
    os.path.join(BASE_DIR, "shap_classification_explainer.pkl")
)

_shap_explainer = _shap_data["explainer"]
FEATURE_NAMES = _shap_data["feature_names"]

# Classification also needs the label encoder, to turn the model's 0/1/2
# output back into "Low Activity" / "Normal Activity" / "High Activity"
_label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# Load training data for LIME
_clf_data = joblib.load(os.path.join(BASE_DIR, "preprocessed_classification.pkl"))
X_train = np.array(_clf_data["X_train"])

_lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=FEATURE_NAMES,
    class_names=[str(c) for c in _label_encoder.classes_],
    mode="classification",
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


def predict_(data):
    """Return the predicted activity level label, e.g. 'High Activity'."""
    encoded_prediction = _model.predict(data)[0]
    return _label_encoder.inverse_transform([int(encoded_prediction)])[0]


def predict_probabilities(data):
    """Return prediction probability (%) for each activity level."""
    probabilities = _model.predict_proba(data)[0]
    results = []
    for class_index, probability in enumerate(probabilities):
        class_name = _label_encoder.inverse_transform([class_index])[0]
        results.append({
            "class": str(class_name),
            "probability": round(float(probability) * 100, 2),
        })
    return results


def explain_shap(data):

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

'''
def explain_shap(data):
    """Return SHAP explanation as a list of dicts (easy to render in HTML)."""
    shap_values = _shap_explainer.shap_values(data)[0]
    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Feature Value": data[0],
        "SHAP Value": shap_values.round(4),
    })
    return shap_df.to_dict("records")   # list of dicts
'''


def explain_shap(data):
    """Return SHAP values for the class predicted for this input."""

    all_shap_values = _shap_explainer.shap_values(data)

    # Find the class predicted by the classification model
    predicted_class = _model.predict(data)[0]
    class_index = list(_model.classes_).index(predicted_class)

    # SHAP may return either a list or a 3D NumPy array
    if isinstance(all_shap_values, list):
        # One array per class: shape usually (1, number_of_features)
        shap_values = all_shap_values[class_index][0]
    else:
        # Shape: (rows, features, classes)
        shap_values = all_shap_values[0, :, class_index]

    shap_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Feature Value": data[0],
        "SHAP Value": shap_values.round(4),
    })

    return shap_df.to_dict("records")
'''
def explain_lime(data):
    """Return LIME explanation as a list of dicts."""
    exp = _lime_explainer.explain_instance(
        data[0],
        lambda x: _model.predict(pd.DataFrame(x, columns=FEATURE_NAMES)),
    )
    lime_list = exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["Feature", "Contribution"])
    lime_df["Contribution"] = lime_df["Contribution"].round(4)
    return lime_df.to_dict("records")'''

def explain_lime(data):
    exp = _lime_explainer.explain_instance(
        data[0],
        lambda x: _model.predict_proba(
            pd.DataFrame(x, columns=FEATURE_NAMES)
        ),
    )

    lime_list = exp.as_list()
    lime_df = pd.DataFrame(
        lime_list,
        columns=["Feature", "Contribution"]
    )

    lime_df["Contribution"] = lime_df["Contribution"].round(4)

    return lime_df.to_dict("records")