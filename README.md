# jopacc-payment-prediction
# Explainable Payment-System Analytics

A Flask web application for analysing Jordan's payment systems using the JoPACC dataset (Nov 2022 – Jun 2026). It combines predictive models, anomaly detection, explainable AI and decision support in one interface.

## Features

- **Prediction & Explanation** – classification (activity level) and regression models, explained with SHAP and LIME, plus a natural-language explanation generated with the Gemini API.
- **Anomaly gate** – inputs that fall outside the learned normal range are flagged and stopped before any prediction is made.
- **Fuzzy Logic** – evaluates the quality of a payment system's growth (centroid and MOM scores, activated rules).
- **Reinforcement Learning** – a trained Q-learning policy recommends interventions under a budget. Outcomes are simulated, since the dataset has no real intervention records.

## Project structure

```
app.py                      Flask routes
model_classification.py     Classification model + SHAP/LIME
model_regression.py         Regression model
anomaly_detection.py        Anomaly gate
fuzzy.py                    Fuzzy logic system
RL.py                       Q-learning recommendations
gemini_api.py               Gemini explanations
templates/                  HTML templates (base.html + pages)
static/style.css            Styling
*.ipynb                     Analysis and training notebooks
*.pkl                       Trained models and artefacts
```

## Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

### Gemini API key

AI explanations need a Gemini API key. Add your own key in `gemini_api.py`. Never commit a real key to the repository.

## Tech stack

Python, Flask, scikit-learn, SHAP, LIME, Google Gemini API, HTML/CSS.

## Notes

- Reinforcement learning results are simulated and do not represent real historical outcomes.
- Built as a final project for the Applied Analytical Models course.
