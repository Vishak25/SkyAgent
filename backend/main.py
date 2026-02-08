
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from model import STGNN
from data import AviationGraphHandler

app = FastAPI(title="SkyCast ST-GNN Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify the Angular app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
graph_handler = AviationGraphHandler()
# Initialize model (Input dim=4 for node features, hidden=16, output=1 scalar delay)
model = STGNN(in_channels=4, hidden_channels=16, out_channels=1)

@app.on_event("startup")
def load_model():
    # In a real scenario, we would load state_dict here
    # model.load_state_dict(torch.load("stgnn_weights.pth"))
    model.eval()
    print("ST-GNN Model Loaded.")

@app.get("/")
def root():
    return {"message": "SkyCast ST-GNN Backend API is running.", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health_check():
    return {"status": "active", "version": "1.0.0"}

@app.get("/predict/{flight_number}")
def predict_flight_delay(flight_number: str):
    """
    Endpoints that triggers the ST-GNN inference for a specific flight.
    """
    try:
        result = graph_handler.get_prediction_for_flight(flight_number, model)
        return result
    except Exception as e:
        print(f"Error predicting for {flight_number}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
