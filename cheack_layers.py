# import tensorflow as tf

# # Load the model
# model = tf.keras.models.load_model('best_dual_input_model_final.h5')

# # Print the architecture
# model.summary()

# # Specifically find the last Conv2D layer names
# print("\n🔍 CONV LAYERS FOUND:")
# for layer in model.layers:
#     if "conv2d" in layer.name:
#         print(f"Layer Name: {layer.name} | Output Shape: {layer.output_shape}")






import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("--- 🔍 Requesting ALL Models from API ---")
try:
    # Get the raw list
    model_list = list(client.models.list())
    
    if not model_list:
        print("⚠️ The API returned an empty list.")
    else:
        for m in model_list:
            # Print the name and all available attributes to see the structure
            print(f"📦 Model Name: {m.name}")
            # This helps us see what the actual attribute names are
            print(f"   Attributes: {m.__dict__.keys()}") 
            
except Exception as e:
    print(f"❌ API Call Failed: {e}")