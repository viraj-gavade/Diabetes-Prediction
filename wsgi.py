"""
WSGI entry point for compatibility
"""
from app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wsgi:app", host="0.0.0.0", port=8000)
