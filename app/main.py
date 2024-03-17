from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from app.middleware.processPrompt import promptResponse

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

# app = FastAPI()

# # Setup templates directory
# templates = Jinja2Templates(directory="app/templates")

# @app.get("/", response_class=HTMLResponse)
# async def read_form(request: Request):
#     return templates.TemplateResponse("form.html", {"request": request})

# @app.post("/", response_class=HTMLResponse)
# async def process_form(request: Request, prompt: str = Form(...)):
#     # Here you can process the prompt and get the response

#     newResponse = promptResponse(prompt)
#     return templates.TemplateResponse("form.html", {"request": request, "response": newResponse})





url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
