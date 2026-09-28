import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

models_to_test = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

config = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())]
)

for model_name in models_to_test:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Search the web: who is the current CEO of Amazon?",
            config=config,
        )
        print(model_name, "-> WORKS:", (response.text or "")[:100])
    except Exception as e:
        print(model_name, "-> FAILED:", str(e)[:300])