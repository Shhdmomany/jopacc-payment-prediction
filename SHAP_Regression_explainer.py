import joblib
import shap
import numpy as np
import pandas as pd

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


data = joblib.load("preprocessed_regression.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

X_train = pd.DataFrame(X_train, columns=FEATURE_NAMES)
X_test = pd.DataFrame(X_test, columns=FEATURE_NAMES)

print("Data loaded successfully.")
 
model = joblib.load("regression_model.pkl")

print("Model loaded successfully.")

explainer = shap.Explainer(model, X_train)
 
shap_values = explainer(X_test)

print("SHAP values calculated.")

sample_index = 0

sample = X_test.iloc[sample_index:sample_index + 1]

print("True value:", y_test.iloc[sample_index])
print("Predicted value:", model.predict(sample)[0])

shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index].values,
    sample,
    matplotlib=True
)


joblib.dump(
    {
        "explainer": explainer,
        "X_train": X_train
    },
    "shap_regression_explainer.pkl"
)

print("Saved to shap_regression_explainer.pkl")

 

shap_df = pd.DataFrame({
    "Feature": sample.columns,
    "Feature Value": sample.values[0],
    "SHAP Value": shap_values[sample_index].values
})

 

shap_df = shap_df.sort_values(
    by="SHAP Value",
    ascending=False
)

print("\nSHAP Feature Contributions:")
print(shap_df)