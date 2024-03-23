from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

app = FastAPI()


model_id = "vikhyatk/moondream2"
revision = "2024-03-06"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/describe-image", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...)):
    # Read the image file into PIL Image from the uploaded file in-memory
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Assuming you have methods to process the image and generate a description
    # Replace 'model.encode_image' and 'model.answer_question' with the actual methods you need to use
    # For example, you might use the tokenizer to encode the image and text input, 
    # then use the model to generate a response
    # The code below is just a placeholder and likely needs to be replaced with your actual model's API

    enc_image = model.encode_image(image)  # This is likely not the correct method call    
    answer = model.answer_question(enc_image, "What are the numbers", tokenizer)

    
    # Return the result to the template
    return templates.TemplateResponse("form.html", {"request": request, "response": answer})