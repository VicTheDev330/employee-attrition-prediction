
from flask import Flask, request, jsonify, render_template_string
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
MODEL_PATH = Path(__file__).parent / "employee_attrition_app.pkl"
model = joblib.load(MODEL_PATH)

# Frontend interface
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Employee Attrition Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial;
            background: #101827;
            color: white;
            padding: 30px 15px;
        }
        .container {
            max-width: 550px;
            margin: auto;
            background: #1e293b;
            padding: 28px;
            border-radius: 14px;
        }
        h1 { color: #60a5fa; }
        label { display: block; margin-top: 18px; }
        input, select {
            width: 100%;
            padding: 12px;
            margin-top: 7px;
            box-sizing: border-box;
            border-radius: 7px;
        }
        button {
            margin-top: 25px;
            width: 100%;
            padding: 14px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 7px;
            cursor: pointer;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 8px;
            background: #334155;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Employee Attrition Prediction</h1>
    <p>Predict whether an employee may leave the company.</p>

    <form id="predictionForm">
        <label>Employee Age</label>
        <input type="number" id="age" min="18" max="65"
               value="30" required>

        <label>Monthly Income</label>
        <input type="number" id="income" min="1000"
               max="20000" value="5000" required>

        <label>Job Satisfaction</label>
        <select id="satisfaction">
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
        </select>

        <label>OverTime</label>
        <select id="overtime">
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select>

        <button type="submit">Predict Attrition</button>
    </form>

    <div id="result" hidden></div>
</div>

<script>
document.getElementById("predictionForm")
.addEventListener("submit", async function(event) {
    event.preventDefault();

    const result = document.getElementById("result");
    result.hidden = false;
    result.textContent = "Predicting...";

    const data = {
        age: Number(document.getElementById("age").value),
        income: Number(document.getElementById("income").value),
        satisfaction: Number(document.getElementById("satisfaction").value),
        overtime: document.getElementById("overtime").value
    };

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.error || "Prediction failed");
        }

        result.textContent =
            output.prediction + " | Attrition probability: " +
            output.probability + "%";

    } catch (error) {
        result.textContent = "Error: " + error.message;
    }
});
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)


# Python prediction API
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON input"}), 400

        age = int(data["age"])
        income = int(data["income"])
        satisfaction = int(data["satisfaction"])
        overtime = data["overtime"]

        if not 18 <= age <= 65:
            raise ValueError("Age must be between 18 and 65.")

        if not 1000 <= income <= 20000:
            raise ValueError("Income must be between 1000 and 20000.")

        if satisfaction not in [1, 2, 3, 4]:
            raise ValueError("Job satisfaction must be 1, 2, 3 or 4.")

        if overtime not in ["Yes", "No"]:
            raise ValueError("OverTime must be Yes or No.")

        input_data = pd.DataFrame([{
            "Age": age,
            "MonthlyIncome": income,
            "JobSatisfaction": satisfaction,
            "OverTime": overtime
        }])

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1]) * 100

        return jsonify({
            "prediction": (
                "Employee may leave the company"
                if prediction == 1
                else "Employee may stay in the company"
            ),
            "probability": round(probability, 2)
        })

    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)
