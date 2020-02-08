from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

app = Flask(__name__)

# @app.route('/test')
# def test():
#     return "Flask is set up :)"

# Load the model here so that we dont have to load the model again and again.
# Thus we can imporve the performance of the app significantly

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            Age = float(request.form['Age'])
            Weight = float(request.form['Weight'])
            Height = float(request.form['Height'])
            BMI = Weight / ((Height)**2)
            pregnencies = int(request.form['pregnencies'])
            Glucose = int(request.form['Glucose'])
            Insulin = int(request.form['Insulin'])
            DPF = float(request.form['DPF'])
            BP = float(request.form['BP'])
            skinThickness = int(request.form['skinThickness'])

            pred_args = [pregnencies,Glucose,BP,skinThickness,Insulin,BMI,DPF,Age]

            pred_args_arr = np.array(pred_args)

            pred_args_arr = pred_args_arr.reshape(1,-1)

            Stdscalar = StandardScaler()

            pred_args_arr = Stdscalar.fit_transform(pred_args_arr)

            # ML_model_read = open("LR_model.pkl", "rb")
            # ML_model_read = open("XGBoost_model.pkl", "rb")
            #ML_model_read = open("Final_xgModel.pkl", "rb")
            ML_model_read = open("RF_Model.pkl", "rb")

            ML_model = joblib.load(ML_model_read)

            Model_prediction = ML_model.predict(pred_args_arr)

            Model_prediction = float(Model_prediction)
            
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction= Model_prediction)



if __name__ == '__main__':
    app.run(host= '127.0.0.5')