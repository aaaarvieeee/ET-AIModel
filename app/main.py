from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import base64
import requests
import os

app = FastAPI()

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/describe-image", response_class=HTMLResponse)
async def process_image(request: Request, house_number_image: UploadFile = File(...), gesture_image: UploadFile = File(...)):

    house_number_image_bytes = await house_number_image.read()
    base64_house_number_image = base64.b64encode(house_number_image_bytes).decode('utf-8')
    
    # Read and encode the second image
    gesture_image_bytes = await gesture_image.read()
    base64_gesture_image = base64.b64encode(gesture_image_bytes).decode('utf-8')

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "tell me what the number is in the first image. the second image should be a person holding their hands up. tell me if that is true"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_house_number_image}"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_gesture_image}"
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response_data = response.json()
    print(response_data)
    answer = response_data['choices'][0]['message']['content']

    # Return the result to the template
    return templates.TemplateResponse("form.html", {"request": request, "response": answer})
