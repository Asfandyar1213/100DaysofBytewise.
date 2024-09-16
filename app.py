from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the models
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

dl_model = load_model('dl_model.h5')

# Load the scaler
scaler = StandardScaler()
scaler.fit_transform(np.zeros((1, 4)))  # Dummy fit to initialize scaler

app = FastAPI()

class PredictionRequest(BaseModel):
    data: list

@app.post("/predict_rf")
async def predict_rf(request: PredictionRequest):
    data = request.data
    data_scaled = scaler.transform([data])
    prediction = rf_model.predict(data_scaled)
    return {"prediction": prediction[0]}

@app.post("/predict_dl")
async def predict_dl(request: PredictionRequest):
    data = request.data
    data_scaled = scaler.transform([data])
    prediction = dl_model.predict(data_scaled)
    return {"prediction": int(np.argmax(prediction))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
