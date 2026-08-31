from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)

model=pickle.load(open("model/model.pkl",'rb'))

scaler=pickle.load(open("model/scaler.pkl",'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def predict():
    # Get values from HTML form
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

    # Convert to 2D array
    features = [features]

    # Scale input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0]

    # Wine class names
    wine_classes = {
        0: "Class 0",
        1: "Class 1",
        2: "Class 2"
    }

    result = wine_classes[prediction]

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)

