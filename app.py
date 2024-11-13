import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
#model = pickle.load(open("model.pkl", "rb"))
model = pickle.load(open("my_model_data.pickle", "rb"))



@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods = ["POST"])
def predict():
      
    float_features = [float(x) for x in list(request.form.values())[1:]]
    features = [np.array(float_features)]
    y_pred = model.predict(features)
    round_pred=round(y_pred[0])
    status=''
    if round_pred == 0:
        status = 'Extremely Weak'
    elif round_pred == 1:
        status =  'Weak'
    elif round_pred == 2:
        status = 'Normal'
    elif round_pred == 3:
        status = 'Overweight'
    elif round_pred == 4:
        status = 'Obese'
    elif round_pred == 5:
        status = 'Extremely Obese'
    return render_template("index.html", prediction_text = "Hi {}".format(list(request.form.values())[0])  + ", your predicted BMI is {}. ".format(round(y_pred[0])) + " You are {}".format(status) + ".")

if __name__ == "__main__":
    flask_app.run(debug=True)
