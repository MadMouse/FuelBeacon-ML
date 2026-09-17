from PIL import Image, ImageOps, ImageEnhance
import sys, subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "pytesseract"])

import pytesseract

#import textblob
#from textblob import TextBlob


#ORIGINAL IMAGE
#importing the image under the name imageIn
imageIn = Image.open("./PetrolPrices/petrolprice3.png")

#extracting and printing the text from the original image
#imageExtractedText = pytesseract.image_to_string(imageIn)
#print('Original Image Analysis' , imageExtractedText)

#GRAYSCALED IMAGE
#making the image into grayscale
imageGray = ImageOps.grayscale(imageIn)

#saving the grayscale image to the file name imageSave1.png
imageGray.save("./SavedOutcomesPPReading/imageSave32.png")

#extracting the printing the text from the gray image
imageExtractedText1 = pytesseract.image_to_string(imageGray)
custom_config = r'-c tessedit_char_whitelist=0123456789 --psm 6'
text = pytesseract.image_to_string(imageGray, config=custom_config)

print('Gray Image Analysis' , imageExtractedText1)
print('NumOnly =', text)

#GRAY, RESIZED, BRIGHTENED AND CONTRASTED IMAGE
#making the image into grayscale, resizing with a scalefactor of 2, increasing brighness by 50% and contrast by 20%
imageGray = ImageOps.grayscale(imageIn)

scalingFactor1 = 2
imageGrayResized = imageGray.resize(
    (imageGray.width * scalingFactor1 , imageGray.height * scalingFactor1),
    resample=Image.LANCZOS
)

brightnessFactor = 1.5
enhancerBrightness = ImageEnhance.Brightness(imageGrayResized)
imageGrayResizedBrightened = enhancerBrightness.enhance(brightnessFactor)

contrastFactor = 1.2
enhancerContrast = ImageEnhance.Contrast(imageGrayResizedBrightened)
imageGrayResizedBrightenedContrasted = enhancerContrast.enhance(contrastFactor)

imageGrayResizedBrightenedContrasted.save("./SavedOutcomesPPReading/imageSave7.png")


imageExtractedText2 = pytesseract.image_to_string(imageGrayResizedBrightenedContrasted)
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz --psm 6'
text2 = pytesseract.image_to_string(imageGrayResizedBrightenedContrasted, config=custom_config)
print('Gray, Contrast, Brightened, Scaled Image Analysis' , imageExtractedText2)
print('NumOnly2 =', text2)