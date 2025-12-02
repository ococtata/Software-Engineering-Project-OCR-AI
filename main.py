from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from PIL import Image
import io
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("GENAI_API_KEY not found in environment variables")

client = genai.Client(api_key=api_key)

prompt = (
    "Transcribe the nutrition facts table in this image and output the data "
    "as a single **JSON object**. Use keys like 'serving-size', 'energy-kcal', 'fat', 'carbohydrates', 'proteins', 'saturated-fat', 'trans-fat', 'sugars', 'added-sugars', 'sodium', 'salt', and 'fiber'. "
    "Do not include any text outside of the JSON object."
    "If the values are not in the image, fill the values with 0"
    "Pay attention to the unit, normalize the unit so it's stated in g (gram) instead of mg (miligram)"
)

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        print(f"Received file: {file.filename}")
        
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes))
        print(f"Image loaded: {image.size}, format: {image.format}")

        print("Calling Gemini API...")
        result = client.models.generate_content(
            model='gemini-2.0-flash-exp', 
            contents=[prompt, image]
        )

        data = result.text
        print(f"Raw Gemini response: {data}")

        cleanedData = data.strip().replace('```json', '').replace('```', '').replace('\n','').strip()
        print(f"Cleaned step 1: {cleanedData}")

        cleanedData = re.sub(r'\s+', ' ', cleanedData).strip()
        print(f"Cleaned step 2: {cleanedData}")

        tempDict = {}

        for i in cleanedData.split(','):
            if ':' not in i:
                continue
            x = i.split(':')
            try:
                col = (x[0].split('"'))[1]
            except:
                col = x[0].replace('{', '').replace('"', '').strip()
            num = x[1]

            tempDict[col] = num
            print(f"Parsed: {col} = {num}")

        print(f"tempDict: {tempDict}")

        divider = None
        for i in tempDict:
            if i == 'serving-size':
                divider_raw = tempDict[i].split('"')
                divider_str = divider_raw[1].split('g')
                divider = int(divider_str[0].strip())
                print(f"Found serving-size divider: {divider}g")
                break

        if divider is None or divider == 0:
            print("Warning: No valid serving-size found, defaulting to 1")
            divider = 1

        resDict = {}

        for i in tempDict:
            if i == 'serving-size':
                continue
            else:
                try:
                    value = float(tempDict[i].strip())
                    resDict[i+'_1g'] = round(value/divider, 3)
                except ValueError:
                    try:
                        value_str = tempDict[i].split("}")
                        value = float(value_str[0].strip())
                        resDict[i+'_1g'] = round(value/divider, 3)
                    except:
                        print(f"Warning: Could not parse value for {i}, setting to 0")
                        resDict[i+'_1g'] = 0

        print(f"Final result: {resDict}")
        return JSONResponse(resDict)

    except Exception as e:
        print(f"Error in OCR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {
        "status": "running",
        "endpoints": {
            "/": "Health check",
            "/ocr": "POST - Extract nutrition data from image (returns per 1g values)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)