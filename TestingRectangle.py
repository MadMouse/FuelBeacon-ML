from PIL import Image, ImageOps, ImageEnhance
import sys, subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])

import pytesseract

#import textblob
#from textblob import TextBlob


#ORIGINAL IMAGE
#importing the image under the name imageIn
imageIn = Image.open("./RectangleTest.png")

#extracting and printing the text from the original image
#imageExtractedText = pytesseract.image_to_string(imageIn)
#print('Original Image Analysis' , imageExtractedText)

#GRAYSCALED IMAGE
#making the image into grayscale
imageGray = ImageOps.grayscale(imageIn)

#saving the grayscale image to the file name imageSave1.png
imageGray.save("./TestingRectangle/imageSave.png")

#extracting the printing the text from the gray image
imageExtractedText1 = pytesseract.image_to_string(imageGray)
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'
text = pytesseract.image_to_string(imageGray, config=custom_config)

print('Gray Image Analysis' , imageExtractedText1)
print('WordOnly =', text)