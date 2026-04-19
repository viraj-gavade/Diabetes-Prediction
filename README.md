# Diabetes Prediction

A FastAPI-based machine learning web application that predicts diabetes risk from clinical input features using a trained model built on the Pima Indians Diabetes dataset.

## Features

- Web UI for entering health metrics and getting instant predictions
- End-to-end ML pipeline for ingestion, transformation, training, and inference
- Serialized model and preprocessor artifacts for production inference
- Modular Python package structure with custom logging and exception handling
- Basic API test coverage using FastAPI `TestClient`

## Tech Stack

- **Backend/API:** FastAPI, Uvicorn, Gunicorn
- **ML/Data:** scikit-learn, CatBoost, XGBoost, pandas, numpy
- **Templating/UI:** Jinja2 templates, Tailwind CSS (CDN)
- **Serialization:** dill

## Project Structure

```text
Diabetes-Prediction/
├── app.py                              # FastAPI app and prediction routes
├── wsgi.py                             # WSGI/ASGI entry point helper
├── test_app.py                         # Basic endpoint test
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── Procfile                            # Process command for deployment
├── render.yaml                         # Render deployment configuration
├── artifacts/                          # Trained model + processed files
│   ├── model.pkl
│   ├── processor.pkl
│   ├── raw.csv
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py           # Ingestion + pipeline trigger
│   │   ├── data_transformation.py      # Feature preprocessing
│   │   ├── model_trainer.py            # Model training and selection
│   │   └── pipline/
│   │       └── prediction_pipeline.py  # Runtime prediction pipeline
│   ├── utils.py                        # Utility methods (save/load/eval)
│   ├── logger.py                       # Logging setup
│   └── exception_handler.py            # Custom exception wrapper
├── templates/                          # Jinja2 HTML pages
├── static/                             # CSS/assets
└── Notebook/                           # Dataset + experimentation notebooks
```

## Input Features

The prediction form expects the following features:

1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

Target column used during training: **Outcome**.

## Local Setup

### 1) Clone and enter project

```bash
git clone <your-repo-url>
cd Diabetes-Prediction
```

### 2) Create virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Run the web app

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://127.0.0.1:8000`

## Model Training Pipeline

To run ingestion, transformation, and training from source data:

```bash
python src/components/data_ingestion.py
```

This pipeline reads `Notebook/diabetes.csv` and updates artifacts in `artifacts/`.

## Running Tests

```bash
python -m pytest -q
```

Current tests validate the `/predict` endpoint response status.

## API Routes

- `GET /` → Home page
- `GET /predict` → Prediction form
- `POST /predict` → Runs inference and returns result page

## Deployment

This repository includes deployment files for common hosts:

- **Procfile** for process-based platforms
- **render.yaml** for Render web service configuration

## Disclaimer

This project is for educational and screening purposes only and is **not** a medical diagnosis tool. Always consult a qualified healthcare professional for clinical decisions.
