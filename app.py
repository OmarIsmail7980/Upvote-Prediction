from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Tag = int(request.form['Tag'])
    Reputation = int(request.form['Reputation'])
    Answers = int(request.form['Answers'])
    Views = int(request.form['Views'])
     
    arr = np.array([[Tag,Reputation,Answers,Views]])
    
    prediction = model.predict(arr)
    
    prediction = np.round(prediction[0], 0)
    
    return render_template("index.html", prediction_text=
                           'Number of Upvotes you will get is {}'.format(prediction))
    
    
if __name__ == "__main__":
    app.run(debug=True)

