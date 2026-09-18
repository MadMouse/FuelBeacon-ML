import cv2
import numpy as np
import pytesseract
import re

def normalize_ocr_number(text: str) -> str:
    # Common OCR character-to-digit swaps
    ocr_map = str.maketrans({
        'O': '0', 'o': '0', 'D': '0',
        'l': '1', '|': '1', '!': '1',
        'Z': '2', 'z': '2',
        'E': '3',
        'A': '4',
        'S': '5', 's': '5', '$': '5',
        'G': '6', 'b': '6',
        'T': '7',
        'B': '8',
        'g': '9', 'q': '9',
        ',': '.', ' ': ''  # Fix comma decimals and rogue spaces
    })
    return text.translate(ocr_map)



# 1. Load image
img = cv2.imread('/Users/aimerydrewery/Desktop/Work/StartUp/PetrolPrices/petrolPriceCrop6.png')

# 2. Preprocess image for LED/7-segment displays
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smooth noise while keeping edges crisp
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Contrast stretching (helps with bright LED digits on dark backgrounds)
norm_img = cv2.normalize(blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Thresholding (Otsu binarization to turn digits black/white)
_, thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological Closing: Dilate slightly to bridge gaps in 7-segment display gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 3. Configure Tesseract specifically for numeric gas prices
# --psm 6: Assume a single uniform block of text
# tessedit_char_whitelist: Restrict search strictly to digits and decimal point
custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789.'

# 4. Extract Text
raw_text = pytesseract.image_to_string(cleaned, config=custom_config)

prices2 = normalize_ocr_number(pytesseract.image_to_string(cleaned))

allextract = pytesseract.image_to_string(cleaned)
 
print("Raw Extract Prices:", allextract)
print("Extracted and Cleaned Fuel Prices", prices2)

cv2.imshow( 'window' ,cleaned)
cv2.waitKey(0)
cv2.destroyAllWindows()
