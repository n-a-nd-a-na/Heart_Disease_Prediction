from fastapi import FastAPI
import skops.io as sio
import joblib
import os
import pandas as pd

from Backend.schemas import PatientData

app = FastAPI()

model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"model","model.skops")

model = sio.load(model_path)

scaler = joblib.load('model/scaler.pkl')

@app.get("/")
def home():
    return {'message':'Heart Disease Prediction API is running'}


@app.post('/predict')
def predict(data:PatientData):
    input_data = pd.DataFrame([{
        'Age' : data.Age,
        'Sex' : 1 if data.Sex == "M" else 0,
        'ChestPainType' : data.ChestPainType,
        'RestingBP' : data.RestingBP,
        'Cholesterol' : data.Cholesterol,
        'FastingBS' : data.FastingBS,
        'RestingECG' : data.RestingECG,
        'MaxHR' : data.MaxHR,
        'ExerciseAngina' : 1 if data.ExerciseAngina == 'Y' else 0,
        'Oldpeak' : data.Oldpeak,
        'ST_Slope' : data.ST_Slope
        }])

    input_data = pd.get_dummies(input_data,columns=['ChestPainType','RestingECG','ST_Slope'],drop_first=True,dtype = int)

    expected_columns = ['Age','Sex','RestingBP','Cholesterol','FastingBS','MaxHR','ExerciseAngina','Oldpeak','ChestPainType_ATA',
                        'ChestPainType_NAP','ChestPainType_TA','RestingECG_Normal','RestingECG_ST','ST_Slope_Flat','ST_Slope_Up']

    input_data = input_data.reindex(columns = expected_columns,fill_value = 0)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability =model.predict_proba(input_scaled)[0][1]

    risk_percentage = round(probability * 100, 2)

    return {'prediction' : int(prediction),
            'risk_percentage' : risk_percentage}
