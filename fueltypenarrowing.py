import gradio as gr
import pytesseract
import cv2
import numpy as np
from PIL import Image
from rapidfuzz import process, fuzz

# Comprehensive dictionary of fuel types and alternatives
FUEL_TYPES = [
    # Standard Fuels
    "Petrol", "Diesel", "Unleaded", #"Regular Unleaded", "Regular Diesel", "City Diesel"
    
    #  Gases
    "Autogas",
    
    # Additives & Alternative Fuels
    #"AdBlue", "HVO", "Biodiesel", "E85", "E10", "E5", "B7", "B10",
    
    # Hydrogen & EV
    #"Hydrogen", "H2", "EV Charge", "Electric",
    
    # Premium / Brand Variants
    "Super Unleaded", "Premium Unleaded", #"Ultimate", "V-Power", 
    #"Synergy", "Momentum", "Supreme", "MilesPlus", "CleanFuel"
]

def preprocess_fuel_image(pil_img):
    """Enhance image contrast for better fuel text recognition."""
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

def extract_positioned_fuels(image):
    """Detect fuel types and preserve their spatial order (top-to-bottom)."""
    # Extract bounding box data from Tesseract
    data = pytesseract.image_to_data(image, config=r'--oem 3 --psm 11', output_type=pytesseract.Output.DICT)
    
    n_boxes = len(data['text'])
    word_matches = []

    # Iterate through recognized words and match fuel types
    for i in range(n_boxes):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if len(word) < 2 or conf < 30:
            continue
            
        match, score, _ = process.extractOne(word, FUEL_TYPES, scorer=fuzz.WRatio)
        
        if score >= 75:
            top = data['top'][i]
            left = data['left'][i]
            word_matches.append({
                'match': match,
                'top': top,
                'left': left
            })

    if not word_matches:
        return "No standard fuel types detected."

    # Sort vertically (top-to-bottom), then horizontally
    word_matches.sort(key=lambda item: (item['top'] // 20, item['left']))

    # Deduplicate while preserving sequence order
    ordered_fuels = []
    for item in word_matches:
        fuel_name = item['match']
        if fuel_name not in ordered_fuels:
            ordered_fuels.append(fuel_name)

    return "Detected Fuel Types (in image order):\n" + "\n".join(f"{idx + 1}. {fuel}" for idx, fuel in enumerate(ordered_fuels))

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
        # Run positional detection on preprocessed image first
        result = extract_positioned_fuels(processed_img)
        
        # Fallback to unprocessed raw image if thresholding removed faint details
        if "No standard fuel types" in result:
            result = extract_positioned_fuels(image)
            
        return result

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
            submit_btn = gr.Button("Identify Fuels", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(label="Detected Result", lines=8)
            
    submit_btn.click(
        fn=extract_text_and_numbers,
        inputs=[img_input, mode_input],
        outputs=text_output
    )

demo.launch()