import cv2 
import re
import pytesseract
from pytesseract import Output
import save
from save import safe_path
import matplotlib
from matplotlib import pyplot as plt

img = cv2.imread('/Users/aimerydrewery/Desktop/Work/StartUp/PetrolPrices/petrolPriceCrop.png')

def convert_grayscale(img):
    img = cv2.cvtColor(img,
                       cv2.COLOR_BGR2GRAY)
    return img

def blur(img, param):
    img - cv2.medianBlur(img, param)
    return img

def threshold(img):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

h, w, c = img.shape

boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():
    
    b = b.split(' ')
    x1 = int(b[1])
    y1 = int(h - int(b[2]))
    x2 = int(b[3])
    y2 = int(h - int(b[4]))
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

cv2.imshow("img", img)
cv2.waitKey(0)

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d.keys())

n_boxes = len(d['test'])

for i in range (n_boxes):
    if int(float(c['conf'][i])) > 60:
        (x, y, w, h) = (d["left"][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle (img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)

