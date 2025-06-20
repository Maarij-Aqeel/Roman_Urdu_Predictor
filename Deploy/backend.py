from fastapi import FastAPI,Query
from model_main import Predictive_Keyboard
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn


device = torch.device('cpu')

Keyboard = Predictive_Keyboard(device)
Keyboard.load_model('./models/Roman_urdu_predictor_Final.pth')

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"message": "Incorrect Usage"}

@app.get("/predictions/")
def give_predictions(text: str = Query(..., min_length=1), top_k: int = 5):
    """
    Get predicted next words given input text
    """
    
    predictions = Keyboard.predict_next_word(text, top_k=top_k)
    return {"input": text, "predictions": [{"word": w, "probability": round(p, 4)} for w, p in predictions]}
