from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    """Test the predict endpoint with a POST request"""
    form_data = {
        "pregnancies": 1,
        "glucose": 120,
        "bloodpressure": 70,
        "skinthickness": 20,
        "insulin": 80,
        "bmi": 25.5,
        "diabetespedigreefunction": 0.5,
        "age": 30
    }
    
    response = client.post("/predict", data=form_data)
    print(f"Status Code: {response.status_code}")
    print(f"Content: {response.content[:500]}")  # Print first 500 chars of response
    assert response.status_code == 200

if __name__ == "__main__":
    test_predict_endpoint()
