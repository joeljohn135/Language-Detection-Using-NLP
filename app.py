import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import string
import re
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    translate_table = dict((ord(char),None)for char in string.punctuation)
    int_features = [x for x in request.form.values()]
    text =' '.join([str(item)for item in int_features])
    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"\d+","",text)
    text = text.translate(translate_table)
    pred = model.predict([text])
    prob = model.predict_proba([text])
    output = pred
    output = ' '.join(output)

    return render_template('index.html', prediction_text="The language is : "+output)


if __name__ == "__main__":
    app.run(debug=True)