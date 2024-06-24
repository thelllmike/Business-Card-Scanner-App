from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import io
import cv2
import numpy as np
import spacy
import logging
import re
import phonenumbers
from geopy.geocoders import Nominatim

app = FastAPI()

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Configure the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # MacOS/Linux

# Load the spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# Initialize geolocator
geolocator = Nominatim(user_agent="address_extractor")

@app.post("/extract-details/")
async def extract_details(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        preprocessed_image = preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed_image)
        details = parse_details(text)
        return JSONResponse(content=details)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_image(image):
    try:
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return Image.fromarray(thresh)
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        raise

def parse_details(text):
    try:
        doc = nlp(text)
        doc = enhance_ner(doc)
        name = extract_name(doc)
        company = extract_company(doc)
        address = extract_address(text)
        phones = extract_phone(text)
        email = extract_email(text)
        website = extract_website(text)
        return {
            "name": name,
            "company": company,
            "address": address,
            "phones": phones,
            "email": email,
            "website": website
        }
    except Exception as e:
        logging.error(f"Error in parse_details: {e}")
        raise

def enhance_ner(doc):
    # Additional custom rules or improvements can be added here
    return doc

def extract_name(doc):
    try:
        # Assuming the first detected PERSON entity is the name
        person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        return person_entities[0] if person_entities else "Name not found"
    except Exception as e:
        logging.error(f"Error in extract_name: {e}")
        return "Name not found"

def extract_company(doc):
    try:
        # Assuming the first detected ORG entity is the company
        org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        return org_entities[0] if org_entities else "Company not found"
    except Exception as e:
        logging.error(f"Error in extract_company: {e}")
        return "Company not found"

def extract_phone(text):
    try:
        phone_numbers = []
        for match in phonenumbers.PhoneNumberMatcher(text, "US"):  # Specify region if known
            phone_numbers.append(phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL))
        return phone_numbers if phone_numbers else ["Not found"]
    except Exception as e:
        logging.error(f"Error in extract_phone: {e}")
        raise

def extract_email(text):
    try:
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
        email_matches = re.findall(email_regex, text, re.IGNORECASE)
        return email_matches[0] if email_matches else "Not found"
    except Exception as e:
        logging.error(f"Error in extract_email: {e}")
        raise

def extract_website(text):
    try:
        website_regex = r'\b(?:http[s]?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7}\b'
        website_matches = re.findall(website_regex, text)
        return website_matches[0] if website_matches else "Not found"
    except Exception as e:
        logging.error(f"Error in extract_website: {e}")
        raise

def extract_address(text):
    try:
        address_entities = [ent.text for ent in nlp(text).ents if ent.label_ in ["GPE", "LOC", "FAC", "ADDRESS"]]
        if address_entities:
            full_address = ", ".join(address_entities)
            location = geolocator.geocode(full_address)
            if location:
                return location.address
            else:
                return full_address
        return "Address not found"
    except Exception as e:
        logging.error(f"Error in extract_address: {e}")
        return "Address not found"
