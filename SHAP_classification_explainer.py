import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

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

 

data = joblib.load("preprocessed_classification.pkl")

X_train = pd.DataFrame(data["X_train"], columns=FEATURE_NAMES)
X_test = pd.DataFrame(data["X_test"], columns=FEATURE_NAMES)
y_train = data["y_train"]
y_test = data["y_test"]

print("Data loaded successfully.")
 

model = joblib.load("classification_model.pkl")

print("Model loaded successfully.")
 

explainer = shap.Explainer(model, X_train)

print("SHAP explainer created.")
shap_values = explainer(X_test)

print("SHAP values calculated.")

 
sample_index = 13

sample = X_test.iloc[[sample_index]]

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("\nTrue class:", y_test.iloc[sample_index])
print("Predicted class:", prediction)
print("Prediction probabilities:", probabilities)

 
try:
    predicted_class_index = list(model.classes_).index(prediction)

    shap.force_plot(
        shap_values.base_values[sample_index, predicted_class_index],
        shap_values.values[sample_index, :, predicted_class_index],
        sample.iloc[0],
        matplotlib=True,
        show=False
    )

    plt.show()

except Exception as e:
    print("Force plot could not be generated:")
    print(e)

sample_values = shap_values.values[sample_index]

if sample_values.ndim == 2:
    sample_values = sample_values[:, predicted_class_index]

shap_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Feature Value": sample.iloc[0].values,
    "SHAP Value": sample_values
})

shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()

shap_df = shap_df.sort_values(
    by="Absolute SHAP",
    ascending=False
)

print("\nTop Feature Contributions")
print(shap_df)

joblib.dump(
    {
        "explainer": explainer,
        "X_train": X_train,
        "feature_names": FEATURE_NAMES,
    },
    "shap_classification_explainer.pkl"
)

print("Saved successfully.")
