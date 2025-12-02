from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from PIL import Image
import re
import json
import io
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS (required for Flutter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API
api = os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api)

# OCR prompt
prompt = (
    "Transcribe the nutrition facts table in this image and output the data "
    "as a single JSON object. Use keys like 'serving-size', 'energy-kcal', "
    "'fat', 'carbohydrates', 'proteins', 'saturated-fat', 'trans-fat', 'sugars', "
    "'added-sugars', 'sodium', 'salt', and 'fiber'. "
    "If any values are missing, set them to 0. "
    "Normalize mg into grams (e.g., 500mg → 0.5g). "
)

def parse_json_from_model(raw):
    cleaned = raw.strip().replace("```json", "").replace("```", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Try strict JSON first
    try:
        return json.loads(cleaned)
    except:
        # Fallback parser
        data = {}
        body = cleaned.replace("{", "").replace("}", "")
        for pair in body.split(","):
            if ":" not in pair:
                continue
            key, value = pair.split(":", 1)
            key = key.replace('"', "").strip()
            try:
                value = float(value.replace('"', "").strip())
            except:
                value = 0
            data[key] = value
        return data

@app.get("/")
def home():
    return {"message": "OCR FastAPI is running on Deta!"}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))

        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image],
        )

        raw_json = result.text
        parsed = parse_json_from_model(raw_json)

        # Extract serving size
        serving = parsed.get("serving-size", "0g")
        match = re.findall(r"([\d\.]+)", str(serving))
        divider = float(match[0]) if match else 1

        # Normalize nutrients to per-1g
        normalized = {}
        for k, v in parsed.items():
            if k == "serving-size":
                continue
            try:
                num = float(v)
            except:
                num = 0
            normalized[k + "_1g"] = round(num / divider, 3)

        return JSONResponse(normalized)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
