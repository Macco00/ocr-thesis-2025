import cv2
import pytesseract
from PIL import Image
import os

# Default path for Tesseract OCR (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === OCR MOTOR FLAGS ===
USE_TESSERACT = True
USE_GOOGLE = False
USE_AMAZON = False

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return enhanced

def run_tesseract(image):
    return pytesseract.image_to_string(image)

# Placeholder — replace with real API later
def run_google(image):
    return "[Google OCR not implemented]"

def run_amazon(image):
    return "[Amazon OCR not implemented]"

def save_text(result, engine, processed, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{engine}_{'preprocessed' if processed else 'original'}.txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)

def main():
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hello.png")
    
    if not os.path.exists(image_path):
        print(f"Imagepath '{image_path}' not found.")
        return

    print("[INFO] Reading image...")
    normal_image = cv2.imread(image_path)
    preprocessed_img = preprocess_image(image_path)

    # === OCR: Tesseract ===
    if USE_TESSERACT:
        print("[INFO] Running Tesseract...")
        tesseract_normal = run_tesseract(normal_image)
        tesseract_preprocessed = run_tesseract(preprocessed_img)
        save_text(tesseract_normal, "tesseract", processed=False)
        save_text(tesseract_preprocessed, "tesseract", processed=True)

    # === OCR: Google Vision ===
    if USE_GOOGLE:
        print("[INFO] Running Google OCR...")
        google_normal = run_google(normal_image)
        google_preprocessed = run_google(preprocessed_img)
        save_text(google_normal, "google", processed=False)
        save_text(google_preprocessed, "google", processed=True)

    # === OCR: AWS Textract ===
    if USE_AMAZON:
        print("[INFO] Running Amazon Textract...")
        amazon_normal = run_amazon(normal_image)
        amazon_preprocessed = run_amazon(preprocessed_img)
        save_text(amazon_normal, "amazon", processed=False)
        save_text(amazon_preprocessed, "amazon", processed=True)

    print("[INFO] OCR results saved to /results")

if __name__ == "__main__":
    main()
