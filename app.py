from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", 'rb'))
scaler = pickle.load(open("model/scalerr.pkl", 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    features = [
        float(request.form["alcohol"]),
        float(request.form["malic_acid"]),
        float(request.form["ash"]),
        float(request.form["alcalinity"]),
        float(request.form["magnesium"]),
        float(request.form["total_phenols"]),
        float(request.form["flavanoids"]),
        float(request.form["nonflavanoid_phenols"]),
        float(request.form["proanthocyanins"]),
        float(request.form["color_intensity"]),
        float(request.form["hue"]),
        float(request.form["od280"]),
        float(request.form["proline"])
    ]

    features = np.array([features])  # 2D array, shape (1, 13)

    # Use transform, NOT fit_transform
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    wine_classes = {
        0: "Class 0",
        1: "Class 1",
        2: "Class 2"
    }

    result = wine_classes[prediction]

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)