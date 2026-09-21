import gradio as gr
import pytesseract
import cv2
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

# Canonical fuel types to target
FUEL_TYPES = [
    "Petrol", "Diesel", "Unleaded", "Premium Unleaded", 
    "Super Unleaded", "E10", "E5", "B7", "LPG", "Gasoline"
]

def preprocess_fuel_image(pil_img):
    """Enhance image contrast specifically for fuel pump screens and receipts."""
    open_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    
    # Upscale resolution
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Otsu Binarization
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def post_process_fuel_types(raw_text):
    """Detect fuel types from raw OCR output using fuzzy matching."""
    lines = raw_text.split('\n')
    detected_fuels = set()

    for line in lines:
        words = line.strip().split()
        for word in words:
            if len(word) < 2:
                continue
            # Match OCR word against known fuel dictionary
            match, score, _ = process.extractOne(word, FUEL_TYPES, scorer=fuzz.WRatio)
            if score >= 75:  # Match confidence threshold
                detected_fuels.add(match)

    if detected_fuels:
        return "Detected Fuel Types:\n" + "\n".join(f"- {f}" for f in detected_fuels)
    return "No standard fuel types confidently detected."

def extract_text_and_numbers(image, mode):
    if image is None:
        return "Please upload an image."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    processed_img = preprocess_fuel_image(image)
    
    if mode == "Numbers Only":
        config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.,$-/ \n'
        raw_text = pytesseract.image_to_string(processed_img, config=config)
        return raw_text.strip() if raw_text.strip() else "No numbers detected."
    
    else:
        config = r'--oem 3 --psm 11'
        raw_text = pytesseract.image_to_string(processed_img, config=config)
        
        # Fallback to raw image if preprocessing returns nothing
        if not raw_text.strip():
            raw_text = pytesseract.image_to_string(image, config=config)
            
        # Return only the fuzzy matched fuel types
        return post_process_fuel_types(raw_text)

# Gradio Interface
with gr.Blocks(title="Fuel Recognition Engine") as demo:
    gr.Markdown("## Fuel Type Extractor")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Receipt or Pump Screen")
            mode_input = gr.Radio(
                choices=["Fuel Type Detection", "Numbers Only"], 
                value="Fuel Type Detection", 
                label="Extraction Mode"
            )
            submit_btn = gr.Button("Identify Fuel", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="Detected Result", lines=6)
            
    submit_btn.click(
        fn=extract_text_and_numbers,
        inputs=[img_input, mode_input],
        outputs=text_output
    )

demo.launch()