# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 23:16:17 2022

@author: amr_a
"""

from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('breast.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    arr = np.array([[data1, data2, data3, data4, data5]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)