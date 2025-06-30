from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join("model", "lightgbm_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    confidence = None

    if request.method == "POST":
        try:
            account_balance = float(request.form.get("account_balance"))
            age = int(request.form.get("age"))
            account_type = request.form.get("account_type")
            loan_amount = float(request.form.get("loan_amount"))
            interest_rate = float(request.form.get("interest_rate"))
            transaction_amount = float(request.form.get("transaction_amount"))
            loan_term = int(request.form.get("loan_term"))

            # Create DataFrame
            input_data = pd.DataFrame([{
                'Account Balance': account_balance,
                'Age': age,
                'Account Type': account_type,
                'Loan Amount': loan_amount,
                'Interest Rate': interest_rate,
                'Transaction Amount': transaction_amount,
                'Loan Term': loan_term
            }])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            prediction_result = "Approved" if prediction == 1 else "Rejected"
            confidence = f"{(probability if prediction == 1 else 1 - probability) * 100:.2f}%"

        except Exception as e:
            prediction_result = f"Error: {e}"
            confidence = "N/A"

    return render_template("index.html", prediction=prediction_result, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)
