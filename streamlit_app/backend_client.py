import requests

API_BASE_URL = "http://localhost:8000"  # or your deployment URL

def get_insights(chat_data):
    try:
        response = requests.post(f"{API_BASE_URL}/insights", json=chat_data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
