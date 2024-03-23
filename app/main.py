from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import base64
import requests
import io
import openai

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/describe-image", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')  # Encode image to base64

    api_key = "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"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",  # This is hypothetical as GPT-4 wasn't available at the time of my knowledge update.
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the number in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    answer = response_data['choices'][0]['message']['content']

    # Return the result to the template
    return templates.TemplateResponse("form.html", {"request": request, "response": answer})
