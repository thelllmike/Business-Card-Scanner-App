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

app = FastAPI()

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Configure the Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # MacOS/Linux

# Load the spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

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
        name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "Name not found")
        company = next((ent.text for ent in doc.ents if ent.label_ == "ORG"), "Company not found")
        address = next((ent.text for ent in doc.ents if ent.label_ == "GPE"), "Address not found")
        phone = extract_phone(text)
        email = extract_email(text)
        return {
            "name": name,
            "company": company,
            "address": address,
            "phone": phone,
            "email": email
        }
    except Exception as e:
        logging.error(f"Error in parse_details: {e}")
        raise

def enhance_ner(doc):
    # Additional custom rules or improvements can be added here
    return doc

def extract_phone(text):
    try:
        phone_regex = r'(\+?\d{1,3}?[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9})'
        phone_matches = re.findall(phone_regex, text)
        return phone_matches[0] if phone_matches else "Not found"
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
