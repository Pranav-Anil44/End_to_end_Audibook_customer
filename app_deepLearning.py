from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
from tensorflow.keras.models import load_model


# Load trained deep learning model
dl_model = load_model("Models/deep_learning_model.h5")

# Must match exactly what you used during training
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

@app.route("/")
def home():
    return "✅ Deep Learning Audiobook Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        print("📩 Incoming JSON:", data, flush=True)

        if data is None:
            return jsonify({"error": "No valid JSON received"}), 400

        # Validate features
        missing = [f for f in EXPECTED_FEATURES if f not in data]
        extra = [f for f in data if f not in EXPECTED_FEATURES]

        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400
        if extra:
            return jsonify({"error": f"Unexpected features: {extra}"}), 400

        # Convert to DataFrame/array
        df = pd.DataFrame([[data[f] for f in EXPECTED_FEATURES]], columns=EXPECTED_FEATURES)

        # Prediction (deep learning model outputs probability)
        prob = dl_model.predict(df)[0][0]
        prediction = int(prob > 0.5)

        print(f"🤖 Prediction: {prediction}, Probability: {prob:.4f}", flush=True)

        return jsonify({
            "DeepLearning_Prediction": prediction,
            "Probability": float(prob)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ app_deeplearning.py starting...")
    app.run(debug=True)