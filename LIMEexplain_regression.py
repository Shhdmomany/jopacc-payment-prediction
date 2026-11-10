import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

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


_data = joblib.load("preprocessed_regression.pkl")
X_train = _data["X_train"]
_model = joblib.load("regression_model.pkl")
X_train = np.array(X_train)
X_train_df = pd.DataFrame(X_train,columns=FEATURE_NAMES)

 
explainer = LimeTabularExplainer(
    training_data=X_train_df.values,
    feature_names=X_train_df.columns.tolist(),
    mode="regression",
    random_state=43
)

def explain_using_lime(input_values):

    input_values = np.array(input_values).flatten()

    sample_df = pd.DataFrame(
        [input_values],
        columns=X_train_df.columns
    )

    prediction = float(
        _model.predict(sample_df)[0]
    )

    exp = explainer.explain_instance(
        sample_df.values[0],
        lambda x: _model.predict(
            pd.DataFrame(
                x,
                columns=X_train_df.columns
            )
        )
    )

    print("prediction:", prediction)

    print("Explanation for sample:")

    explanation_list_reg = exp.as_list()

    explanation_df = pd.DataFrame(
        explanation_list_reg,
        columns=["Feature", "Contribution"]
    )

    return explanation_df






input_values = [1000, 10,     # returned_cheque_count
    1.5,    # returned_cheque_value_rate_pct
    1.0,    # returned_cheque_count_rate_pct
    5000,   # returned_cheque_value_jod
    100000, # ef_digital_payment_value_jod
    50000,  # value_money_transfers_jod
    25000,  # on_us_money_transfer_value_jod
    30000,  # value_purchases_jod
    20000,  # off_us_money_transfer_value_jod
    100,    # off_us_money_transfer_count
    50,     # customer_legal_entity_count
    500,    # customer_individual_count
    200,    # tx_cash_out_count
    300     # tx_cash_in_count
]

explanation = explain_using_lime(input_values)

print("\nLIME Explanation:")
print(explanation)


joblib.dump(
    {
        "explainer": explainer,
        "X_train": X_train_df,
        "feature_names": FEATURE_NAMES,
    },
    "lime_regression_explainer.pkl"
)
print("LIME explainer saved successfully.")