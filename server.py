from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import tensorflow as tf
from tensorflow.contrib import predictor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd
import datetime

modelPath = "output_model.h5"
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
with SimpleXMLRPCServer(('localhost', 8000),
                        requestHandler=RequestHandler) as server:
    server.register_introspection_functions()

    # load model from .h5 file
    model = tf.keras.models.load_model(modelPath)
    # Register a function under a different name
    def predict(date):
        prevDay = incrementDateStr(date,-1)
        obj = np.array([[newCases['United States'][prevDay], newDeaths['United States'][prevDay], totalCases['United States'][prevDay], totalDeaths['United States'][prevDay]]])
        obj = scaler.transform(obj.reshape(-1,4))
        prediction = model.predict(obj)
        print(prediction)
        return prediction[0][0]

    predict("2020-9-26")
    server.register_function(predict, 'predict')

    

    # Run the server's main loop
    server.serve_forever()