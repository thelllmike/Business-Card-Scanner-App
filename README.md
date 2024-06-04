# Business Card Scanner App

## Overview
The Business Card Scanner App is designed to help users digitize business cards by capturing their images and extracting key information such as names, phone numbers, and email addresses. This solution leverages Optical Character Recognition (OCR) and Named Entity Recognition (NER) to ensure accurate data extraction.

## Technologies Used
- **FastAPI**: Used to create RESTful endpoints for uploading images and retrieving extracted data.
- **Tesseract OCR**: An OCR engine used to convert images to text.
- **spaCy**: A powerful library for advanced Natural Language Processing (NLP), used here for Named Entity Recognition to identify names from text.
- **OpenCV**: Employed for image preprocessing to enhance OCR accuracy.
- **Pillow (PIL Fork)**: Used for basic image handling operations like opening and converting images.
- **Python**: The primary programming language used.
- **Pytesseract**: A Python wrapper that provides a way to interact with Tesseract OCR.

## Project Structure

/backend
/app
- main.py # Entry point for the FastAPI app
- dependencies.py # Optional: For managing dependencies
/test
- test_main.py # Optional: For endpoint tests
requirements.txt # Python dependencies


## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/business-card-scanner.git
   cd business-card-scanner


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


Install the required packages

pip install -r requirements.txt

Configure Tesseract Path (Adjust according to your Tesseract installation)

Windows 

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

Linux/Mac

pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # or wherever tesseract is installed



