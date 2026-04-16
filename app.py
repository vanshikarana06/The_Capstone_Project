from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
import cv2
from google import genai
from google.genai import errors
from dotenv import load_dotenv
from hasher import generate_hash
from metadata import get_metadata

load_dotenv()

app = FastAPI()

# 1. SETUP & MOUNTING
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. MODELS & GLOBALS
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = tf.keras.models.load_model('best_dual_input_model_final.h5')

class ChatRequest(BaseModel):
    message: str

last_analysis = {
    "result": "No image analyzed yet",
    "confidence": 0,
    "metadata": {}
}

# 3. HELPER FUNCTIONS (Placed ABOVE the routes)

def run_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    resaved_path = 'temp_resaved.jpg'
    original.save(resaved_path, 'JPEG', quality=quality)
    resaved = Image.open(resaved_path)
    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image.resize((128, 128))

LEGAL_DATABASE = {
    "TAMPERED": {
        "title": "Advisory: Potential Digital Forgery Detected",
        "sections": [
            {"act": "IT Act, 2000", "sec": "Section 65", "desc": "Tampering with computer source documents."},
            {"act": "BNS, 2023", "sec": "Section 336", "desc": "Forgery of electronic records."}
        ],
        "implication": "High risk of document invalidity under Section 65B of the Indian Evidence Act."
    },
    "AUTHENTIC": {
        "title": "Integrity Verified",
        "sections": [],
        "implication": "Image shows no significant ELA noise variance."
    }
}

def get_legal_advice(result):
    return LEGAL_DATABASE.get(result)

def generate_gradcam(model, rgb_tensor, ela_tensor, original_path):
    last_conv_layer_name = "conv2d_3" 
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model([rgb_tensor, ela_tensor])
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    img = cv2.imread(original_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    heatmap_filename = os.path.join(UPLOAD_DIR, f"heatmap_{os.path.basename(original_path)}")
    cv2.imwrite(heatmap_filename, superimposed_img)
    return heatmap_filename

# 4. ROUTES

@app.get("/")
async def home(request: Request):
    # Notice 'request' is passed first, then the template name, then the context.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": None,
            "confidence": None,
            "image_path": None,
            "heatmap_path": None,
            "hash": "No analysis performed",
            "metadata": {
                "size_kb": "0",
                "software": "N/A",
                "camera": "N/A"
            },
            "legal_info": None
        }
    )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. System Instruction - New SDK style
        # Some versions of the new SDK prefer this in the 'config'
        config = {
            "system_instruction": "You are a Forensic Image Specialist. Use the analysis context provided to help the user."
        }
        
        # 2. Context Data - Ensuring it's a string
        context_data = f"Analysis Context: {str(last_analysis)}"
        user_message = request.message

        # 3. The Call - Note the contents structure [context, message]
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[context_data, user_message],
            config=config
        )
        
        return {"analysis_reply": response.text}

    except Exception as e:
        # THIS IS THE CRITICAL PART: Check your terminal for this output!
        print("\n" + "="*50)
        print("🔴 GEMINI API ERROR DETECTED")
        print(f"Error Type: {type(e).__name__}")
        print(f"Details: {str(e)}")
        print("="*50 + "\n")
        
        if "429" in str(e):
            return {"analysis_reply": "⚠️ Rate limit hit. Please wait 1 minute."}
        
        return {"error": f"API Connection Issue: {type(e).__name__}"}
        
        
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Save file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Pre-process
    file_hash = generate_hash(file_path)
    file_meta = get_metadata(file_path) 
    rgb_img = np.array(Image.open(file_path).convert('RGB').resize((128, 128))) / 255.0
    ela_img = np.array(run_ela(file_path)) / 255.0
    
    rgb_tensor = np.expand_dims(rgb_img, 0)
    ela_tensor = np.expand_dims(ela_img, 0)

    # Prediction (Named inputs as required by your model)
    # ✅ Use a dictionary so the model knows which tensor is which

    prediction = model.predict({"raw_input": rgb_tensor, "ela_input": ela_tensor})
    is_tampered = prediction[0][0] > 0.5
    result = "TAMPERED" if is_tampered else "AUTHENTIC"
    confidence = round(float((prediction[0][0] if is_tampered else 1 - prediction[0][0]) * 100), 2)
    
    full_heatmap_path = generate_gradcam(model, rgb_tensor, ela_tensor, file_path)
    
    # Update global state
    global last_analysis
    last_analysis = {"result": result, "confidence": confidence, "metadata": file_meta}

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": result,
            "confidence": confidence,
            "image_path": f"uploads/{file.filename}",
            "heatmap_path": f"uploads/{os.path.basename(full_heatmap_path)}",
            "legal_info": get_legal_advice(result),
            "hash": file_hash,       
            "metadata": file_meta    
        }
    )

    