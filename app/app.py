from flask import Flask, request, jsonify
import joblib
from pathlib import Path
import pandas as pd

app = Flask(__name__)
def load_model():
    model_path = Path(__file__).resolve().parents[1] / "model" / "churn_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)

def extract_data(data):
    df = pd.DataFrame({
        "gender": [data.get('gender')],
        "SeniorCitizen": [int(data.get('SeniorCitizen', 0))],
        "Partner": [data.get('Partner')],
        "Dependents": [data.get('Dependents')],
        "tenure": [data.get('tenure')],
        "PhoneService": [data.get('PhoneService')],
        "MultipleLines": [data.get('MultipleLines')],
        "InternetService": [data.get('InternetService')],
        "OnlineSecurity": [data.get('OnlineSecurity')],
        "OnlineBackup": [data.get('OnlineBackup')],
        "DeviceProtection": [data.get('DeviceProtection')],
        "TechSupport": [data.get('TechSupport')],
        "StreamingTV": [data.get('StreamingTV')],
        "StreamingMovies": [data.get('StreamingMovies')],
        "Contract": [data.get('Contract')],
        "PaperlessBilling": [data.get('PaperlessBilling')],
        "PaymentMethod": [data.get('PaymentMethod')],
        "MonthlyCharges": [data.get('MonthlyCharges')],
        "TotalCharges": [data.get('TotalCharges')]
    })
    return df

# The whole web page lives here as a string
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Client Churn Prediction</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #60a5fa;
            --indigo: #4f46e5;
            --bg-1: #0f172a;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --card-bg: #ffffff;
            --input-bg: #f8fafc;
            --radius: 16px;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 20px 40px -10px rgba(15, 23, 42, 0.08);
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(160deg, #eef2ff 0%, #f0f9ff 40%, #f8fafc 100%);
            min-height: 100vh;
            padding: 48px 20px 64px;
            color: var(--text);
            -webkit-font-smoothing: antialiased;
        }

        /* ---------- Decorative background blobs ---------- */
        body::before,
        body::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(90px);
            z-index: -1;
            opacity: 0.5;
            pointer-events: none;
        }

        body::before {
            width: 480px;
            height: 480px;
            background: radial-gradient(circle, #c7d2fe, transparent 70%);
            top: -120px;
            right: -120px;
        }

        body::after {
            width: 420px;
            height: 420px;
            background: radial-gradient(circle, #bae6fd, transparent 70%);
            bottom: -100px;
            left: -100px;
        }

        .container {
            max-width: 960px;
            margin: auto;
        }

        /* ---------- Header ---------- */
        .header {
            text-align: center;
            margin-bottom: 40px;
            animation: fadeUp 0.7s cubic-bezier(0.4, 0, 0.2, 1) both;
        }

        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #e0e7ff;
            color: var(--indigo);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 7px 16px;
            border-radius: 100px;
            margin-bottom: 18px;
        }

        .header-badge::before {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--indigo);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.4); }
            50% { box-shadow: 0 0 0 6px rgba(79, 70, 229, 0); }
        }

        .header h1 {
            font-size: clamp(30px, 5vw, 44px);
            font-weight: 800;
            background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 45%, #4f46e5 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }

        .header p {
            color: var(--text-muted);
            font-size: 16px;
            max-width: 480px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---------- Card ---------- */
        .card {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 40px;
            box-shadow: var(--shadow);
            border: 1px solid rgba(226, 232, 240, 0.8);
            backdrop-filter: blur(10px);
            animation: fadeUp 0.7s cubic-bezier(0.4, 0, 0.2, 1) 0.15s both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(24px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* ---------- Grouped sections ---------- */
        .form-group-block {
            margin-bottom: 36px;
        }

        .form-group-block:last-of-type {
            margin-bottom: 0;
        }

        .group-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--indigo);
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }

        .group-title .icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 8px;
            background: #eef2ff;
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px 24px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 7px;
            color: #334155;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        label .required {
            color: #ef4444;
        }

        input,
        select {
            width: 100%;
            padding: 12px 14px;
            border: 1.5px solid var(--border);
            border-radius: 10px;
            font-size: 15px;
            font-family: inherit;
            background-color: var(--input-bg);
            color: var(--text);
            transition: var(--transition);
            appearance: none;
        }

        select {
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%2364748b' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
            padding-right: 40px;
            cursor: pointer;
        }

        input:focus,
        select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
        }

        input:hover,
        select:hover {
            border-color: #94a3b8;
        }

        input::placeholder {
            color: #94a3b8;
        }

        /* ---------- Submit button ---------- */
        .button-container {
            margin-top: 36px;
        }

        button {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--indigo) 100%);
            color: white;
            font-size: 16px;
            font-weight: 700;
            font-family: inherit;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }

        button::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, transparent 40%, rgba(255,255,255,0.2) 50%, transparent 60%);
            transform: translateX(-100%);
            transition: transform 0.6s ease;
        }

        button:hover::after {
            transform: translateX(100%);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.35);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            opacity: 0.75;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        button .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2.5px solid rgba(255, 255, 255, 0.4);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
            vertical-align: -3px;
            margin-right: 8px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* ---------- Result box ---------- */
        #result {
            display: none;
            margin-top: 24px;
            padding: 20px 22px;
            border-radius: 14px;
            text-align: center;
            font-size: 17px;
            font-weight: 600;
            line-height: 1.5;
            animation: slideDown 0.4s cubic-bezier(0.4, 0, 0.2, 1) both;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        #result.churn {
            background-color: #fef2f2;
            color: #b91c1c;
            border: 1.5px solid #fecaca;
        }

        #result.no-churn {
            background-color: #f0fdf4;
            color: #166534;
            border: 1.5px solid #bbf7d0;
        }

        #result.error {
            background-color: #fff7ed;
            color: #c2410c;
            border: 1.5px solid #fed7aa;
        }

        #result .result-icon {
            display: block;
            font-size: 26px;
            margin-bottom: 6px;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            margin-top: 32px;
            color: #94a3b8;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .footer .dot {
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: #cbd5e1;
        }

        /* ---------- Responsive ---------- */
        @media (max-width: 700px) {
            body { padding: 28px 14px 48px; }
            .form-grid { grid-template-columns: 1fr; }
            .card { padding: 26px 20px; border-radius: 18px; }
        }
    </style>

<base target="_blank">
</head>


<body>

<div class="container">

    <div class="header">
        <div class="header-badge">Machine Learning Model</div>
        <h1>Client Churn Prediction</h1>
        <p>Enter the customer's information to predict whether they are likely to churn.</p>
    </div>


    <div class="card">

        <form id="predictionForm">

            <!-- Personal Information -->
            <div class="form-group-block">
                <div class="group-title">
                    <span class="icon">👤</span>
                    Personal Information
                </div>

                <div class="form-grid">

                    <div class="form-group">
                        <label for="gender">Gender</label>
                        <select id="gender">
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="SeniorCitizen">Senior Citizen</label>
                        <select id="SeniorCitizen">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Partner">Partner</label>
                        <select id="Partner">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Dependents">Dependents</label>
                        <select id="Dependents">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="tenure">Tenure (Months) <span class="required">*</span></label>
                        <input type="number" id="tenure" min="0" placeholder="Example: 24" required>
                    </div>

                </div>
            </div>


            <!-- Services -->
            <div class="form-group-block">
                <div class="group-title">
                    <span class="icon">📡</span>
                    Services
                </div>

                <div class="form-grid">

                    <div class="form-group">
                        <label for="PhoneService">Phone Service</label>
                        <select id="PhoneService">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="MultipleLines">Multiple Lines</label>
                        <select id="MultipleLines">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No phone service">No Phone Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="InternetService">Internet Service</label>
                        <select id="InternetService">
                            <option value="DSL">DSL</option>
                            <option value="Fiber optic">Fiber Optic</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="OnlineSecurity">Online Security</label>
                        <select id="OnlineSecurity">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="OnlineBackup">Online Backup</label>
                        <select id="OnlineBackup">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="DeviceProtection">Device Protection</label>
                        <select id="DeviceProtection">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="TechSupport">Tech Support</label>
                        <select id="TechSupport">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="StreamingTV">Streaming TV</label>
                        <select id="StreamingTV">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="StreamingMovies">Streaming Movies</label>
                        <select id="StreamingMovies">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="No internet service">No Internet Service</option>
                        </select>
                    </div>

                </div>
            </div>


            <!-- Contract & Billing -->
            <div class="form-group-block">
                <div class="group-title">
                    <span class="icon">💳</span>
                    Contract &amp; Billing
                </div>

                <div class="form-grid">

                    <div class="form-group">
                        <label for="Contract">Contract</label>
                        <select id="Contract">
                            <option value="Month-to-month">Month-to-Month</option>
                            <option value="One year">One Year</option>
                            <option value="Two year">Two Year</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="PaperlessBilling">Paperless Billing</label>
                        <select id="PaperlessBilling">
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="PaymentMethod">Payment Method</label>
                        <select id="PaymentMethod">
                            <option value="Electronic check">Electronic Check</option>
                            <option value="Mailed check">Mailed Check</option>
                            <option value="Bank transfer (automatic)">Bank Transfer</option>
                            <option value="Credit card (automatic)">Credit Card</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="MonthlyCharges">Monthly Charges <span class="required">*</span></label>
                        <input type="number" step="0.01" min="0" id="MonthlyCharges" placeholder="Example: 79.50" required>
                    </div>

                    <div class="form-group">
                        <label for="TotalCharges">Total Charges <span class="required">*</span></label>
                        <input type="number" step="0.01" min="0" id="TotalCharges" placeholder="Example: 1450.75" required>
                    </div>

                </div>
            </div>


            <div class="button-container">
                <button type="submit" id="predictButton">Predict Customer Churn</button>
            </div>

        </form>


        <div id="result"></div>

    </div>


    <div class="footer">
        Machine Learning <span class="dot"></span> Customer Churn Prediction System
    </div>

</div>


<script>

    document
        .getElementById("predictionForm")
        .addEventListener(
            "submit",
            async function (event) {

                event.preventDefault();

                const button =
                    document.getElementById("predictButton");

                const resultBox =
                    document.getElementById("result");

                const data = {
                    gender: document.getElementById("gender").value,
                    SeniorCitizen: document.getElementById("SeniorCitizen").value,
                    Partner: document.getElementById("Partner").value,
                    Dependents: document.getElementById("Dependents").value,
                    tenure: parseInt(document.getElementById("tenure").value),
                    PhoneService: document.getElementById("PhoneService").value,
                    MultipleLines: document.getElementById("MultipleLines").value,
                    InternetService: document.getElementById("InternetService").value,
                    OnlineSecurity: document.getElementById("OnlineSecurity").value,
                    OnlineBackup: document.getElementById("OnlineBackup").value,
                    DeviceProtection: document.getElementById("DeviceProtection").value,
                    TechSupport: document.getElementById("TechSupport").value,
                    StreamingTV: document.getElementById("StreamingTV").value,
                    StreamingMovies: document.getElementById("StreamingMovies").value,
                    Contract: document.getElementById("Contract").value,
                    PaperlessBilling: document.getElementById("PaperlessBilling").value,
                    PaymentMethod: document.getElementById("PaymentMethod").value,
                    MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
                    TotalCharges: parseFloat(document.getElementById("TotalCharges").value)
                };

                try {

                    button.disabled = true;
                    button.innerHTML = '<span class="spinner"></span>Predicting...';

                    resultBox.style.display = "none";
                    resultBox.className = "";


                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(data)
                    });

                    if (!response.ok) {
                        throw new Error("Prediction request failed");
                    }

                    const result = await response.json();

                    const predictionText = String(result.prediction);

                    let icon, boxClass;
                    if (/yes|churn/i.test(predictionText) && !/no churn/i.test(predictionText)) {
                        icon = "⚠️";
                        boxClass = "Yes";
                    } else {
                        icon = "✅";
                        boxClass = "No";
                    }

                    resultBox.innerHTML =
                        '<span class="result-icon">' + icon + '</span>' +
                        'Prediction: ' + predictionText;

                    resultBox.classList.add(boxClass);
                    resultBox.style.display = "block";

                }
                catch (error) {

                    resultBox.innerHTML =
                        '<span class="result-icon">❌</span>' +
                        'Error: Could not get prediction.';

                    resultBox.classList.add("error");
                    resultBox.style.display = "block";

                    console.error(error);

                }
                finally {

                    button.disabled = false;
                    button.innerText = "Predict Customer Churn";

                }

            }
        );

</script>

</body>
</html>"""

@app.post('/predict')
def predict():
    data = request.get_json()
    df = extract_data(data)
    model = load_model()
    pred = model.predict(df)
    if pred.item() == 1:
        return jsonify({'prediction': 'Yes'})
    else:
        return jsonify({'prediction': 'No'})

@app.get('/')
def index():
    return INDEX_HTML

if __name__ == '__main__':
    app.run(debug=True)
