from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import PredictPipline,CustomData
import pandas as pd 
import os 
import sys 
import numpy as np 
#csv_data  = pd.read_csv(os.path.join("notebook/data","Clean Used Bike Price Data.csv"))

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        csv_data  = pd.read_csv(os.path.join("artifcats","raw.csv"))
        return render_template("form.html",bike_name=csv_data["bike_name"].unique(),city=csv_data["city"].unique())

    else:
        data = CustomData(
            bike_name = request.form.get("bike_name")
            , city = request.form.get("city")
            , kms_driven =float(request.form.get("kms_driven"))
            , owner = int(request.form.get("owner"))
            , age = float(request.form.get("age"))
            , power = float(request.form.get("power"))
            , brand = int(request.form.get("brand"))
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.Predict(final_data)

        result = round(pred[0],2)

        return render_template("form.html",final_result = "Your Bike Price Is: {}".format(result))

        
if __name__=="__main__":
    app.run(host="0.0.0.0")
