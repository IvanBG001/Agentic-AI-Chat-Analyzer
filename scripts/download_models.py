import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.model_handler import ModelHandler

def main():
    handler = ModelHandler()
    handler.download_all()

if __name__ == "__main__":
    main()
