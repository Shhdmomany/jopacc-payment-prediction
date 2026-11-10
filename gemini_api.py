from google import genai

GEMINI_API_KEY = "YOUR_API_KEY"

client = genai.Client(api_key="")


def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    return response.text


def explain_with_gemini(classification_prediction,
                         regression_prediction,
                         shap_table,
                         lime_table):

    prompt = f"""
You are an Explainable AI assistant for the JoPACC Payment System Prediction Portal.

Your task is to explain the machine learning predictions using ONLY the information provided below.
Do NOT invent information or mention features that are not listed.

--------------------------------------------------
MODEL OUTPUT
--------------------------------------------------

Activity Level Prediction:
{classification_prediction}

Predicted Transaction Value (JOD):
{regression_prediction}

--------------------------------------------------
SHAP FEATURE IMPORTANCE
--------------------------------------------------

{shap_table}

SHAP values indicate how each feature influenced the prediction.
• Positive SHAP values increase the prediction.
• Negative SHAP values decrease the prediction.

--------------------------------------------------
LIME FEATURE CONTRIBUTIONS
--------------------------------------------------

{lime_table}

LIME explains which features were most influential for this individual prediction.

--------------------------------------------------
INSTRUCTIONS
--------------------------------------------------

Write a clear explanation in 4 short paragraphs.

Paragraph 1:
Explain what the predicted activity level means and summarize the predicted transaction value.

Paragraph 2:
Describe the most important factors according to SHAP.
Mention which features increased the prediction and which reduced it.

Paragraph 3:
Explain how LIME supports or differs from the SHAP explanation.
If both methods identify similar important features, mention that they agree.
If they differ, briefly explain that each method analyzes feature importance differently.

Paragraph 4:
Provide a short overall interpretation of the prediction for a business user.

Rules:
- Use simple professional English.
- Do not mention technical formulas.
- Do not use bullet points.
- Do not repeat feature names unnecessarily.
- Keep the explanation between 180 and 250 words.
- Base every statement only on the provided SHAP and LIME results.
"""

    return ask_gemini(prompt)




def explain_fuzzy_growth(
    transactions,
    wallets,
    cash_in,
    cashout,
    centroid_score,
    mom_score,
    label,
    rules
):
    """
    Explain the fuzzy logic evaluation using Gemini.
    """

    rule_descriptions = {
        "Rule 1": "High transactions, high wallet users, high cash-in value, and low cash-out indicate Sustainable Growth.",
        "Rule 2": "High transactions together with high cash-out indicate Unbalanced Growth.",
        "Rule 3": "Medium transactions, medium wallet users, and medium cash-in indicate Healthy Growth.",
        "Rule 4": "High cash-in value with low cash-out indicates Healthy Growth.",
        "Rule 5": "High transactions, high wallet users, and medium cash-out indicate Healthy Growth.",
        "Rule 6": "High wallet users but low cash-in value indicate Unbalanced Growth.",
        "Rule 7": "Low transactions together with high cash-out indicate Fragile Growth."
    }

    activated_rules = "\n".join(
        f"{rule}: {rule_descriptions.get(rule, '')}"
        for rule in rules
    )

    prompt = f"""
You are an Explainable AI assistant specializing in fuzzy logic systems.

Your task is to explain the results of a fuzzy logic model that evaluates the growth quality of a digital payment system.

==============================
INPUT VALUES
==============================

Transactions Total:
{transactions}

Wallet Individual Count:
{wallets}

Cash-in Value (JOD):
{cash_in}

Cash Out Count:
{cashout}

==============================
FUZZY LOGIC RESULTS
==============================

Centroid Score:
{centroid_score}

Mean of Maximum (MOM):
{mom_score}

Growth Quality:
{label}

Activated Rules:

{activated_rules}

==============================
INSTRUCTIONS
==============================

Write a professional explanation in simple English.

Include:

1. Explain what the predicted Growth Quality means.

2. Explain that fuzzy logic evaluates variables using degrees of membership instead of strict yes/no boundaries.

3. Explain why the activated rules contributed to the final decision.

4. Explain what the Centroid score represents.

5. Explain what the MOM score represents.

6. End with a short recommendation describing how the payment system could improve or maintain its growth quality.

Rules:

- Do not mention programming code.
- Do not mention Python.
- Do not invent new rules.
- Only use the activated rules provided.
- Write in 4–6 short paragraphs.
- Keep the explanation between 180 and 250 words.
"""

    return ask_gemini(prompt)