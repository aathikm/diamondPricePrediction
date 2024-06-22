from flask import Flask, redirect, request, render_template

from src.pipeline.prediction_pipeline import PredictionPipeline, GetCustomData
from src.logFile.loggingInfo import logging

app = Flask(__name__)
host = "127.0.0.1"
port = 5000

@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict_datapoint():
    
    if request.method == "GET":
        return render_template("form.html")
    else:
        customData_pipe = GetCustomData(
            carat=float(request.form.get("carat")),
            depth=float(request.form.get("depth")),
            table=float(request.form.get("table")),
            x=float(request.form.get("x")),
            y=float(request.form.get("y")),
            z=float(request.form.get("z")),
            cut=request.form.get("cut"),
            color=request.form.get("color"),
            clarity=request.form.get("clarity")            
        )
        data = customData_pipe.get_data()
        
        logging.info(f"the data is: {data}")
        predict_pipe = PredictionPipeline()
        pred = predict_pipe.predict(data)
        print(pred)
        
        result = round(pred[0], 2)
       
        return render_template("prediction.html", result_val=result)

if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)