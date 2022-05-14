import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, render_template, request
app=Flask(__name__)
@app.route('/')
def index():
    return flask.render_template('index.html')
def HumPredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,4)
    loaded_model = pickle.load(open("HUM_forecast.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
def TempPredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = pickle.load(open("TC_forecast.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
@app.route("/predict",methods = ["POST"])
def result():
    if request.method == "POST":
        i1 = request.form.get('i1')
        X_test=np.asarray(int(i1))
        X_test=X_test.reshape((-1,1))
        poly = PolynomialFeatures(degree=3)
        Xh = poly.fit_transform(X_test)
        poly = PolynomialFeatures(degree=4)
        Xt = poly.fit_transform(X_test)
        print(Xt)
        print(Xh)
    result = HumPredictor(Xh)
    result1 = TempPredictor(Xt)
    prediction = [result,result1]
    return render_template("predict.html",input_value = i1, Humidity=prediction[0],Temperature=prediction[1])
if __name__ == "__main__":
    app.run(debug=True)