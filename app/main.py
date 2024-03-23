from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from PIL import Image
import io

app = FastAPI()

# Load the model and processor
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cpu().eval()


# Setup templates directory
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# @app.post("/", response_class=HTMLResponse)
# async def process_form(request: Request, prompt: str = Form(...)):
#     # Here you can process the prompt and get the response

#     newResponse = imageResponse(prompt)
#     return templates.TemplateResponse("form.html", {"request": request, "response": newResponse})

@app.post("/", response_class=HTMLResponse)
async def process_image(file: UploadFile = File(...)):
    # Read the image file into PIL Image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Prepare the conversation structure
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>Describe each stage of this image.",
            "images": [image]  # Directly use the PIL image
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]

    # Load images and prepare for inputs (adjusted to directly use the PIL image)
    pil_images = [image]  # Directly use the loaded PIL image
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # Run the model to get the response
    outputs = vl_gpt.generate(
        input_ids=prepare_inputs.input_ids,
        attention_mask=prepare_inputs.attention_mask,
        max_length=512,
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        use_cache=True
    )

    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return templates.TemplateResponse("form.html", {"response": answer})
