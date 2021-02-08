#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('lrmodel.pkl', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    result = model.predict(final_features)
    if int(result) == 1:
        prediction = "Sorry, you are diabetic"
    else:
        prediction = "You are safe"   # making prediction


    return render_template('index.html', prediction_text=prediction) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)

