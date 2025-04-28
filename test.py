import cv2
import pytesseract
from PIL import Image
import os
from inspect import getsourcefile
from os.path import abspath

# Default path for tesseract-ocr (windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return enhanced

def run_ocr(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    # image location is test.py directory
    image_path = os.path.dirname(os.path.abspath(__file__)) + "\\" + "test_image.png"

    if not os.path.exists(image_path):
        print(f"Imagepath '{image_path}' not found.")
        return

    # Preprocessing
    preprocessed_img = preprocess_image(image_path)

    # OCR
    text = run_ocr(preprocessed_img)

    # Res
    print("===== OCR Output =====")
    print(text)

if __name__ == "__main__":
    main()
