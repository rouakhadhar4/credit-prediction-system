import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
 
model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
 
 
@app.route("/")
def home():
    return render_template("index.html")
 
 
@app.route("/predict", methods=["POST"])
def predict():
    form_values = request.form.to_dict()
 
    # Vérifie qu'aucun champ n'est vide (formulaire non rempli / champ manquant)
    if not form_values or any(value.strip() == "" for value in form_values.values()):
        return render_template(
            "index.html",
            error_text="Veuillez remplir tous les champs du formulaire."
        )
 
    # Vérifie que toutes les valeurs sont bien des nombres valides
    try:
        features = [int(value) for value in form_values.values()]
    except ValueError:
        return render_template(
            "index.html",
            error_text="Veuillez saisir des valeurs valides dans le formulaire."
        )
 
    prediction = model.predict([np.array(features)])
 
    result = int(prediction[0])
 
    if result == 1:
        prediction_text = "Crédit accepté"
        prediction_status = "accepted"
    else:
        prediction_text = "Crédit refusé"
        prediction_status = "rejected"
 
    return render_template(
        "index.html",
        prediction_text=prediction_text,
        prediction_status=prediction_status
    )
 
 
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.get_json()
 
    features = np.array([list(data.values())])
 
    prediction = model.predict(features)
 
    result = int(prediction[0])
 
    prediction_text = "Crédit accepté" if result == 1 else "Crédit refusé"
 
    return jsonify({"prediction": prediction_text})
 
 
if __name__ == "__main__":
    app.run(debug=True)
