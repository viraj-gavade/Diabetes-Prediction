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

# Test endpoint
@app.post("/test-form")
async def test_form(name: str = Form(...)):
    return {"received": name}

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    return templates.TemplateResponse("test.html", {"request": request})

# Routes
@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "page_title": "Home - Diabetes Prediction"})


    return templates.TemplateResponse("visualize.html", {"request": request, "page_title": "Data Visualization"})

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

@app.get("/api/visualization-data")
async def get_visualization_data():
    # Read the CSV file
    df = pd.read_csv('artifacts/raw.csv')
    
    # Basic statistics
    basic_stats = {
        'total_records': len(df),
        'diabetic_count': int(df['Outcome'].sum()),
        'non_diabetic_count': int(len(df) - df['Outcome'].sum())
    }
    
    # Age distribution
    age_bins = [20, 30, 40, 50, 60, 70, 80]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
    df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    age_distribution = df['age_group'].value_counts().sort_index().to_dict()
    
    # Feature correlations with outcome
    correlations = {}
    for col in df.columns:
        if col not in ['Outcome', 'age_group']:
            correlations[col] = float(df[col].corr(df['Outcome']))
    
    # Feature distributions by outcome
    feature_distributions = {}
    for feature in ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']:
        diabetic = df[df['Outcome'] == 1][feature].tolist()
        non_diabetic = df[df['Outcome'] == 0][feature].tolist()
        feature_distributions[feature] = {
            'diabetic': diabetic,
            'non_diabetic': non_diabetic
        }
    
    # Pair-wise correlations
    corr_matrix = df.drop(columns=['age_group'] if 'age_group' in df.columns else []).corr().round(2)
    corr_data = corr_matrix.to_dict()
    
    # BMI categories
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['bmi_category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)
    bmi_outcome = pd.crosstab(df['bmi_category'], df['Outcome']).reset_index()
    bmi_outcome_data = {
        'categories': bmi_outcome['bmi_category'].tolist(),
        'diabetic': bmi_outcome[1].tolist(),
        'non_diabetic': bmi_outcome[0].tolist()
    }
    
    # Return all visualization data
    return {
        'basic_stats': basic_stats,
        'age_distribution': age_distribution,
        'correlations': correlations,
        'feature_distributions': feature_distributions,
        'correlation_matrix': corr_data,
        'bmi_outcome': bmi_outcome_data
    }

if __name__ == "__main__":
    # Create directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    Path("static/css").mkdir(exist_ok=True)
    Path("static/js").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)