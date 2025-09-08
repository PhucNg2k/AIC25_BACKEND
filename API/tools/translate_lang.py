import os
import sys
from dotenv import load_dotenv

from google import genai
from google.genai import types  

# Ensure project root is importable when running: python search_api.py
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.dirname(TOOLS_DIR)
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

load_dotenv()


# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def main():
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko."),
        contents="Hello there"
    )      
    
    print(response.text)
    
    return


if __name__ == "__main__":
    main()

