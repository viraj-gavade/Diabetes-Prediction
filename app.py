from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path
import pandas as pd
import json
import numpy as np
from src.components.pipline.prediction_pipeline import CustomData, PredictionPipeline

# Create FastAPI app
app = FastAPI(title="Diabetes Prediction")

# Add middleware to handle CSRF if needed
from starlette.middleware.sessions import SessionMiddleware
app.add_middleware(SessionMiddleware, secret_key="diabetes_prediction_secret")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Routes
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "page_title": "Home - Diabetes Prediction"})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "page_title": "Predict Diabetes"})

@app.post("/predict", response_class=HTMLResponse)
async def predict_diabetes(
    request: Request,
    pregnancies: int = Form(...),
    glucose: int = Form(...),
    bloodpressure: int = Form(...),
    skinthickness: int = Form(...),
    insulin: int = Form(...),
    bmi: float = Form(...),
    diabetespedigreefunction: float = Form(...),
    age: int = Form(...),
):
    try:
        # Create CustomData object with form values
        data = CustomData(
            Pregnancies=pregnancies,
            Glucose=glucose,
            BloodPressure=bloodpressure,
            SkinThickness=skinthickness,
            Insulin=insulin,
            BMI=bmi,
            DiabetesPedigreeFunction=diabetespedigreefunction,
            Age=age
        )
        
        # Convert to DataFrame
        df = data.get_data_as_dataframe()
        
        # Make prediction
        pipeline = PredictionPipeline()
        prediction = pipeline.predict_result(df)[0]
        
        # Determine prediction result and message
        result = "Positive" if prediction == 1 else "Negative"
        message = "You might be at risk for diabetes. Please consult a healthcare professional." if prediction == 1 else "No diabetes risk detected based on the provided information."
        
        # Return prediction result to template
        return templates.TemplateResponse(
            "predict.html", 
            {
                "request": request, 
                "page_title": "Predict Diabetes",
                "prediction_made": True,
                "result": result,
                "message": message,
                "form_data": {
                    "pregnancies": pregnancies,
                    "glucose": glucose,
                    "bloodpressure": bloodpressure,
                    "skinthickness": skinthickness,
                    "insulin": insulin,
                    "bmi": bmi,
                    "diabetespedigreefunction": diabetespedigreefunction,
                    "age": age
                }
            }
        )
    except Exception as e:
        # Return error message
        return templates.TemplateResponse(
            "predict.html", 
            {
                "request": request, 
                "page_title": "Predict Diabetes",
                "error": f"Error making prediction: {str(e)}"
            }
        )

if __name__ == "__main__":
    # Create directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(exist_ok=True)
    Path("static/js").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)