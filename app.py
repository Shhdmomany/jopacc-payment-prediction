from flask import Flask, render_template, request
from model_classification import predict_, collect_data, explain_shap, explain_lime
from model_regression import predict_regression
from gemini_api import explain_with_gemini
from anomaly_detection import validate_input
from fuzzy import evaluate_growth
from gemini_api import explain_fuzzy_growth
import joblib
from RL import recommend_action
Q = joblib.load("q_learning_policy.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Collect form data
        transactions_total  = request.form["transactions_total"]
        returned_cheque_value_rate_pct = request.form["returned_cheque_value_rate_pct"]
        returned_cheque_value_jod = request.form["returned_cheque_value_jod"]
        returned_cheque_count = request.form["returned_cheque_count"]
        returned_cheque_count_rate_pct = request.form["returned_cheque_count_rate_pct"]
        off_us_money_transfer_count = request.form["off_us_money_transfer_count"]
        customer_legal_entity_count = request.form["customer_legal_entity_count"]
        customer_individual_count = request.form["customer_individual_count"]
        ef_cash_payment_count = request.form["ef_cash_payment_count"]
        ef_digital_payment_count = request.form["ef_digital_payment_count"]
        users_total = request.form["users_total"]
        tx_cash_in_count = request.form["tx_cash_in_count"]
        tx_cash_out_count = request.form["tx_cash_out_count"]
        tx_purchases_count = request.form["tx_purchases_count"]
        tx_money_transfers_count = request.form["tx_money_transfers_count"]


        data = collect_data(transactions_total,
                            returned_cheque_value_rate_pct 
                            ,returned_cheque_value_jod,
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
                            tx_money_transfers_count)
 
        try:
            anomaly_result = validate_input(data)
        except Exception as e:
          return render_template("predict.html",anomaly_gate_error=str(e))

        if anomaly_result["is_anomaly"]:
            return render_template("predict.html", anomaly_result=anomaly_result,prediction_skipped=True)

        
        prediction_classification_final = predict_(data)
        prediction_reg_final = predict_regression(data)

         
        shap_results = explain_shap(data)

        
        lime_results = explain_lime(data)

        
        try:
            gemini_text = explain_with_gemini(prediction_classification_final,prediction_reg_final, shap_results, lime_results)
        except Exception as e:
            gemini_text = f"Gemini API error: {e}"

        return render_template(
            "predict.html",
            prediction_classification_final=prediction_classification_final,
            prediction_reg_final=prediction_reg_final,
            shap_results=shap_results,
            lime_results=lime_results,
            gemini_text=gemini_text,
            anomaly_result=anomaly_result,
        )

    return render_template("predict.html")





@app.route("/fuzzy", methods=["GET", "POST"])
def fuzzy():

    centroid_score = None
    mom_score = None
    label = None
    rules = None
    explanation = None


    if request.method == "POST":

        transactions = float(request.form["transactions"])
        wallets = float(request.form["wallets"])
        cash_in_value = float(request.form["cash_in_value"])
        cashout = float(request.form["cashout"])

        centroid_score, mom_score, label, rules = evaluate_growth(
            transactions,
            wallets,
            cash_in_value,
            cashout
        )
        explanation = explain_fuzzy_growth(transactions,wallets,cash_in_value,cashout,centroid_score,mom_score,label,rules) 


    return render_template(
        "fuzzy.html",
        centroid_score=centroid_score,
        mom_score=mom_score,
        label=label,
        rules=rules,
        explanation=explanation
    )

@app.route("/reinforcement", methods=["GET", "POST"])
def reinforcement():

    if request.method == "POST":

        system = request.form["system"]
        transactions_total = float(request.form["transactions_total"])
        feature_value = float(request.form["feature_value"])
        budget = float(request.form["budget"])

        recommendations = recommend_action(
            system,
            transactions_total,
            feature_value,
            budget
        )

        return render_template(
            "reinforcement.html",
            recommendations=recommendations
        )

    return render_template("reinforcement.html")


if __name__ == "__main__":
    app.run(debug=True)
