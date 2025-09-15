from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
log_model = joblib.load("Models/logistic_model.pkl")

EXPECTED_FEATURES = [
    "Book_length_min",
    "Book_length_char",
    "Avg_rating",
    "Rating_count",
    "Review_score",
    "Price",
    "Discount",
    "Total_minutes_listened",
    "Completion",
    "Support_requests",
    "Support_request"
]
@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("➡️ /predict endpoint hit", flush=True)

        # Parse JSON
        data = request.get_json(silent=True)
        print("📩 Parsed JSON:", data, flush=True)

        if data is None:
            return jsonify({"error": "No valid JSON received"}), 400

        # Check for missing/extra features
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        extra = [f for f in data if f not in EXPECTED_FEATURES]

        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        if extra:
            return jsonify({"error": f"Unexpected features: {extra}"}), 400

        # Prepare dataframe
        df = pd.DataFrame([[data[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)

        # Make prediction
        prediction = log_model.predict(df)[0]
        print("🤖 Prediction result:", prediction, flush=True)

        return jsonify({
            "LogisticRegression_Prediction": int(prediction)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

