import google.generativeai as genai
from PIL import Image
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
 
load_dotenv()
Gemini_API = os.getenv('GEMINI_API_KEY')
genai.configure(api_key = Gemini_API)

# 1. Define the exact output structure using Pydantic
class FuelItem(BaseModel):
    fuel_name: str = Field(description="Name or type of fuel, e.g., Unleaded 95, Diesel, E10")
    price_per_unit: float = Field(description="Numerical unit price per liter or gallon")

class FuelReceiptData(BaseModel):
    station_name: str = Field(description="Name of the fuel station or brand")
    fuels_detected: list[FuelItem]

# 2. Initialize Gemini with the schema enforced
model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": FuelReceiptData,
    }
)

# 3. Process the image
image = Image.open("PetrolPrices/petrolprice5.png")
prompt = "Analyze this image and extract fuel names along with their unit prices and return in json."

response = model.generate_content([image, prompt])

# Output will be clean JSON adhering to your schema
print(response.text)