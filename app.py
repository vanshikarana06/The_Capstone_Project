from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
import os
import sqlite3
import time
from google import genai
from dotenv import load_dotenv

from hasher import generate_hash
from metadata import get_metadata

load_dotenv()

app = FastAPI()

# ── Static files & templates ────────────
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ── ML Model ─────────────────────────────
model = tf.keras.models.load_model('best_dual_input_model_final.h5')

# ── Gemini Client ─────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

last_analysis = {
    "result": "No image analyzed yet",
    "confidence": 0,
    "metadata": {}
}

# ── Legal Database ────────────────────────
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

# ── Database ─────────────────────────────
def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            filename   TEXT,
            hash       TEXT,
            result     TEXT,
            confidence REAL,
            timestamp  TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result(filename, hash_value, result, confidence):
    conn = sqlite3.connect('database.db')
    conn.execute(
        "INSERT INTO results (filename, hash, result, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (filename, hash_value, result, confidence, time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

# ── Helper Functions ──────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    heatmap_filename = os.path.join(UPLOAD_DIR, f"heatmap_{os.path.basename(original_path)}")
    cv2.imwrite(heatmap_filename, superimposed_img)
    return f"/static/uploads/heatmap_{os.path.basename(original_path)}"

# ── Routes ───────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/user", response_class=HTMLResponse)
async def user_view(request: Request):
    return templates.TemplateResponse(request=request, name="user_index.html")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global last_analysis

    if not allowed_file(file.filename):
        return JSONResponse(content={"error": "Only JPG and PNG allowed"}, status_code=400)

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Hash + Metadata
    file_hash = generate_hash(file_path)
    file_meta = get_metadata(file_path)

    # ML preprocessing
    rgb_img = np.array(Image.open(file_path).convert('RGB').resize((128, 128))) / 255.0
    ela_img = np.array(run_ela(file_path)) / 255.0
    rgb_tensor = np.expand_dims(rgb_img, 0)
    ela_tensor = np.expand_dims(ela_img, 0)

    # Prediction
    prediction = model.predict({"raw_input": rgb_tensor, "ela_input": ela_tensor})
    is_tampered = prediction[0][0] > 0.5
    result = "TAMPERED" if is_tampered else "AUTHENTIC"
    confidence = round(float((prediction[0][0] if is_tampered else 1 - prediction[0][0]) * 100), 2)

    # Grad-CAM Heatmap
    heatmap_url = generate_gradcam(model, rgb_tensor, ela_tensor, file_path)

    # Legal info
    legal_info = LEGAL_DATABASE.get(result)

    # Update global for chat context
    last_analysis = {"result": result, "confidence": confidence, "metadata": file_meta}

    # Save to SQLite
    save_result(file.filename, file_hash, result, confidence)

    return JSONResponse(content={
        "filename":    file.filename,
        "hash":        file_hash,
        "metadata":    file_meta,
        "result":      result,
        "confidence":  confidence,
        "image_url":   f"/static/uploads/{file.filename}",
        "heatmap_url": heatmap_url,
        "legal_info":  legal_info
    })

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        config = {
            "system_instruction": "You are a Forensic Image Specialist. Use the analysis context provided to help the user."
        }
        context_data = f"Analysis Context: {str(last_analysis)}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[context_data, request.message],
            config=config
        )
        return {"analysis_reply": response.text}
    except Exception as e:
        if "429" in str(e):
            return {"analysis_reply": "⚠️ Rate limit hit. Please wait 1 minute."}
        return {"error": f"API Connection Issue: {type(e).__name__}"}

# ── Startup ──────────────────────────────
@app.on_event("startup")
async def startup():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    init_db()