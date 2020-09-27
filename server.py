from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd
import datetime
import json
from io import StringIO

modelPath = "output_model.h5"
goodModelPath = "modelBestAttemptStockData.h5"
goodScaler = "scalerGood.save"
scaler = joblib.load("scaler.save") 
dataDir = "data/"
stockDF = pd.read_csv(dataDir+"spData.csv",usecols=["Date","Close"]).set_index("Date")
newCases = pd.read_csv(dataDir+"new_cases.csv",usecols=["date","United States"]).set_index("date")
newDeaths = pd.read_csv(dataDir+"new_deaths.csv",usecols=["date","United States"]).set_index("date")
totalCases = pd.read_csv(dataDir+"total_cases.csv",usecols=["date","United States"]).set_index("date")
totalDeaths = pd.read_csv(dataDir+"total_deaths.csv",usecols=["date","United States"]).set_index("date")

def incrementDateStr(dateStr, numDays):
    dateParts = [int(part) for part in dateStr.split('-')]
    date = datetime.datetime(dateParts[0], dateParts[1], dateParts[2])
    nextDay = date + datetime.timedelta(days = numDays)
    nextDayStr = str(nextDay.year) + '-' + str(nextDay.month).zfill(2) + '-' + str(nextDay.day).zfill(2)
    return nextDayStr

def prevStockMarketDay(date):
    stockDates = list(stockDF.index)
    earliestDay = stockDates[0]
    day = incrementDateStr(date, -1)
    while not day in stockDates:
        day = incrementDateStr(day, -1)
        if day < earliestDay:
            return -1
    return day

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    # Add these headers to all responses
    def end_headers(self):
        self.send_header("Access-Control-Allow-Headers", 
                         "Origin, X-Requested-With, Content-Type, Accept")
        self.send_header("Access-Control-Allow-Origin", "*")
        SimpleXMLRPCRequestHandler.end_headers(self)

# Create server
with SimpleXMLRPCServer(('localhost', 9090),
                        requestHandler=RequestHandler) as server:
    server.register_introspection_functions()

    # load model from .h5 file
    model = tf.keras.models.load_model(modelPath)
    goodModel = tf.keras.models.load_model(goodModelPath)
    goodScaler = joblib.load("scalerGood.save") 
    # Register a function under a different name
    def predict(date):
        prevDay = incrementDateStr(date,-1)
        obj = np.array([[newCases['United States'][prevDay], newDeaths['United States'][prevDay], totalCases['United States'][prevDay], totalDeaths['United States'][prevDay]]])
        obj = scaler.transform(obj.reshape(-1,4))
        prediction = model.predict(obj)
        print(prediction[0][0].item())
        return prediction[0][0].item()
        
    predict("2020-9-26")
    server.register_function(predict, 'predict')

    def predictCurrentGood(date):
        prevDay = incrementDateStr(date, -1)
        prevStockDay = prevStockMarketDay(date)
        if prevStockDay == -1:
            continue
        obj = np.array([[stockDF['Close'][prevStockDay], newCases['United States'][prevDay], newDeaths['United States'][prevDay], totalCases['United States'][prevDay], totalDeaths['United States'][prevDay]]])
        obj = goodScaler.transform(obj.reshape(-1,4))
        prediction = goodModel.predict(obj)
        print(prediction[0][0].item())
        return prediction[0][0].item()
        
    predict("2020-9-26")
    server.register_function(predictCurrentGood, 'predictCurrentGood')

    def pred(data):
        obj = np.array([data])
        obj = scaler.transform(obj.reshape(-1,4))
        return model.predict(obj)

    def predictMany():
        x = []
        yPreds = []
        yTrues = []
        print("predicting many")
        
        for date, price in stockDF['Close'].iteritems():
            x.append(date)
            yTrues.append(price)
            prevDay = incrementDateStr(date, -1)
            data = [newCases['United States'][prevDay], newDeaths['United States'][prevDay], totalCases['United States'][prevDay], totalDeaths['United States'][prevDay]]
            prediction = pred(data)
            yPreds.append(prediction[0][0].item())
            #print(f"{date}: {data}. Prediction: {prediction[0][0]}, Actual: {price}")
        
        data = [x, yPreds, yTrues]
        print(json.dumps(data))
        
        
        print(data)

        return data
    server.register_function(predictMany, 'predictMany')

    def predGood(data):
        obj = np.array([data])
        obj = goodScaler.transform(obj.reshape(-1,5))
        return goodModel.predict(obj)

    def predictGood():
        x = []
        yPreds = []
        yTrues = []
        print("predicting many")

        for date, price in stockDF['Close'].iteritems():
            prevDay = incrementDateStr(date, -1)
            prevStockDay = prevStockMarketDay(date)
            if prevStockDay == -1:
                continue
            x.append(date)
            yTrues.append(price)        
            data = [stockDF['Close'][prevStockDay], newCases['United States'][prevDay], newDeaths['United States'][prevDay], totalCases['United States'][prevDay], totalDeaths['United States'][prevDay]]
            prediction = predGood(data)
            yPreds.append(prediction[0][0].item())
            print(f"{date}: {data}. Prediction: {prediction[0][0]}, Actual: {price}")
            print(data)

        data = [x, yPreds, yTrues]
        return data
    server.register_function(predictGood, 'predictGood')
    

    # Run the server's main loop
    server.serve_forever()