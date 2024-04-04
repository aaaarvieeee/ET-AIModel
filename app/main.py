from fastapi import FastAPI, File, UploadFile, Request, Form
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
async def process_image(request: Request, 
                        user_number: str = Form(...), 
                        user_action: str = Form(...),
                        house_number_image: UploadFile = File(...), 
                        gesture_image: UploadFile = File(...)):

    house_number_image_bytes = await house_number_image.read()
    base64_house_number_image = base64.b64encode(house_number_image_bytes).decode('utf-8')
    
    gesture_image_bytes = await gesture_image.read()
    base64_gesture_image = base64.b64encode(gesture_image_bytes).decode('utf-8')

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # Include the user-specified action in the prompt
    prompt_text = f"The user believes the number in the first image is {user_number}. " \
                  "Tell me what the number is in the first image. " \
                  f"The user describes the action in the second image as '{user_action}'. " \
                  "Tell me if the image matches this description, you can be lenient, no need to be strict. as long as it somewhat matches the description, it's fine. " \
                  "you don't need to describe it, just tell me if it somewhat matches the instruction. also mention, if both are valid, say the package will be delivered." \
                  "if both are invalid, say the package will not be delivered, and vice versa. if the house number is wrong but the gesture is right, don't deliver" \
                  "theoretically, you are a drone with a camera just validating its surroundings to ensure correct delivery location and" \
                  "the customer is legit."

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
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
    answer = response_data['choices'][0]['message']['content']

    # Return the result to the template
    return templates.TemplateResponse("form.html", {"request": request, "response": answer})