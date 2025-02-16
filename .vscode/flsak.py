import flask
import pandas as pd
import joblib
import logging
import threading

# ✅ Initialize Flask App
app = flask.Flask(__name__)

# ✅ Set up logging
logging.basicConfig(filename="api.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ✅ Load the trained model
model = joblib.load("Random Forest_model\model.pkl")  # Ensure this file exists

# ✅ Fraud Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = flask.request.get_json()  # Get JSON input
        df = pd.DataFrame([data])  # Convert input to DataFrame
        prediction = model.predict(df)[0]  # Predict fraud or not
        logging.info(f"Request: {data} | Prediction: {prediction}")
        return flask.jsonify({"fraud_prediction": int(prediction)})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return flask.jsonify({"error": "Prediction failed!"}), 400

# ✅ Fraud Summary Endpoint
@app.route("/fraud_summary", methods=["GET"])
def fraud_summary():
    try:
        df = pd.read_csv("fraud_data.csv")  # Ensure this file exists
        total_transactions = len(df)
        fraud_cases = df["class"].sum()
        fraud_percentage = round((fraud_cases / total_transactions) * 100, 2)

        return flask.jsonify({
            "total_transactions": total_transactions,
            "fraud_cases": fraud_cases,
            "fraud_percentage": fraud_percentage
        })
    except Exception as e:
        logging.error(f"Error fetching fraud summary: {str(e)}")
        return flask.jsonify({"error": "Failed to retrieve fraud data"}), 500

# ✅ Run Flask in a separate thread so Jupyter remains interactive
def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()
