from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from bike_availability import logger
from bike_availability.predict import PredictPipeline, PredictionData
from bike_availability.config.config import conf
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory='templates')

predictor = PredictPipeline(conf)

@app.get("/")
async def root():
    welcome = {
    "message": "Welcome to the Bike Availability API",
    "endpoints": [
        "/predict",
        "/docs"]
    }
    return welcome


@app.post("/predict")
async def predict(request: PredictionData):
    try:
        df = pd.DataFrame([request.dict()])
        pred = predictor.predict(df)
        pred = float(pred)  
        response = {'percentage_docks_available': pred}
        return response, 200
    except Exception as e:
        logger.info(f"Error occurred during prediction: {str(e)}")
        return {'error': 'An error occurred during prediction'}, 500