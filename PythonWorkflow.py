#!/usr/bin/env python3
"""
process_photo.py
Extracts metadata from a local image and calls Google's Gemini 3.6 Flash model
with Google Search grounding to verify facts found in the image and its metadata.
"""
import array
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, List

from google import genai
from google.genai import types
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os
 
load_dotenv()

class Coordinates(BaseModel):
    latitude: Optional[float] = Field(description="Latitude coordinate of the location")
    longitude: Optional[float] = Field(description="Longitude coordinate of the location")

class Location(BaseModel):
    name: str = Field(description="Name of the fuel station or venue")
    address: str = Field(description="Full physical address")
    coordinates: Optional[Coordinates] = Field(description="Geographic coordinates")

class Transaction(BaseModel):
    cost_gbp: float = Field(description="Total cost of the transaction in GBP")
    fuel_dispensed_litres: float = Field(description="Volume of fuel dispensed in litres")
    fuel_type_dispensed: str = Field(description="Fuel type selected for this transaction")
    #price_per_litre_gbp: float = Field(description="Price per litre in GBP for the dispensed fuel")

class PriceDetail(BaseModel):
    fuel_type: str = Field(description="Fuel Type")
    price_per_litre_pence: float = Field(description="Price per litre in pence")
    #price_per_litre_gbp: float = Field(description="Price per litre in GBP")

class FuelPrices(BaseModel):
    prices: List[PriceDetail]
    #diesel: Optional[PriceDetail] = Field(default=None, description="Price details for Diesel")
    #super_fuel_premium_unleaded_97: Optional[PriceDetail] = Field(
    #    default=None, description="Price details for Super / Premium Unleaded fuel")
    #unleaded: Optional[PriceDetail] = Field(default=None, description="Price details for standard Unleaded fuel")

class FuelTransactionSchema(BaseModel):
    location: Location
    date_time: str = Field(description="Timestamp in format YYYY-MM-DD HH:MM:SS (tz)")
    transaction: Transaction
    fuel_prices: FuelPrices

def extract_metadata(image_path: str) -> Dict[str, Any]:
    """Extracts standard EXIF metadata and GPS data from an image file."""
    metadata = {}
    try:
        with Image.open(image_path) as img:
            metadata["format"] = img.format
            metadata["mode"] = img.mode
            metadata["size"] = img.size

            exif_data = img._getexif()
            if not exif_data:
                metadata["exif"] = "No EXIF metadata found."
                return metadata

            parsed_exif = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)

                # Decode GPS Info tags if present
                if tag_name == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id, gps_val in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag_name] = str(gps_val)
                    parsed_exif["GPSInfo"] = gps_data
                else:
                    # Convert non-serializable values to string
                    if isinstance(value, (bytes, tuple)):
                        value = str(value)
                    parsed_exif[tag_name] = value

            metadata["exif"] = parsed_exif

    except Exception as e:
        metadata["error"] = f"Failed to extract metadata: {e}"

    return metadata


def verify_image_with_gemini(image_path: str, metadata: Dict[str, Any]) -> str:
    """Passes the image and extracted metadata to Gemini 3.6 Flash using Google Search grounding."""
    # Ensure API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Error: GEMINI_API_KEY environment variable is not set.")

    # Initialize Gemini client
    client = genai.Client()

    # Load image for multimodal input
    pil_image = Image.open(image_path)

    # Convert metadata dict to clean JSON string
    metadata_json = json.dumps(metadata, indent=2, default=str)

    prompt = f"""
    Using the metadata from the provided image find location and date/time, give name of location using Web Search. If this is a fuel station, extract fuel type and price and return the result in json, and if possible return cost and amount of fuel dispensed
    Extracted Image Metadata:
    ```json
    {metadata_json}
    ```
    """

    #print("Calling Gemini 3.6 Flash with Google Search verification...\n")

    # Call Gemini 3.6 Flash with Google Search enabled
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[pil_image, prompt],
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],  # Enable web search grounding
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=FuelTransactionSchema
        ),
    )

    return response.text


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from an image and verify facts using Gemini 3.6 Flash."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        sys.exit(f"Error: File '{args.image_path}' not found.")

    print(f"Processing image: {args.image_path}")
    #print("-" * 50)

    # 1. Extract EXIF & metadata
    metadata = extract_metadata(args.image_path)
    #print("Extracted Metadata:")
    #print(json.dumps(metadata, indent=2, default=str))
    #print("-" * 50)

    # 2. Verify with Gemini 3.6 Flash
    verification_report = verify_image_with_gemini(args.image_path, metadata)

    #print("\nVerification Results:\n")
    print(verification_report)


if __name__ == "__main__":
    main()