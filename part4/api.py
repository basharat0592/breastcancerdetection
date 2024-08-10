from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define input data model (use first two features for simplicity)
class InputData(BaseModel):
    mean_radius: float
    mean_texture: float
    # Add more features if desired

# Load the model
model = joblib.load('model.joblib')

# Create FastAPI app
app = FastAPI()

@app.post('/predict')
def predict(data: InputData):
    # Prepare input data
    input_data = np.array([[data.mean_radius, data.mean_texture]])

    # Make prediction
    prediction = model.predict(input_data)

    # Return prediction
    return {'prediction': int(prediction[0])}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
