import os
import cv2
import pytesseract
import difflib
import boto3
from google.cloud import vision

# === CONFIGURATION ===
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_key.json"

# === FLAGS ===
USE_TESSERACT = True
USE_GOOGLE = True
USE_AMAZON = True
SHOW_PREPROC_IMAGE = False

RESULTS_DIR = "results"
IMAGE_DIR = "picture"
TEXT_DIR = "text"
CSV_LOG = os.path.join(RESULTS_DIR, "metrics_log.csv")


def preprocess_image(image_path, use_clahe=True, use_threshold=True, use_morph=True, scale=1.2):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if use_threshold:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5
        )

    if use_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return gray





def run_tesseract(image):
    return pytesseract.image_to_string(image)


def run_google(image):
    client = vision.ImageAnnotatorClient()
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return "[Image encoding failed]"
    image_bytes = encoded_image.tobytes()
    google_image = vision.Image(content=image_bytes)
    response = client.text_detection(image=google_image)

    if response.error.message:
        return f"[Google OCR error: {response.error.message}]"

    return response.text_annotations[0].description if response.text_annotations else "[No text detected]"


def run_amazon(image):
    client = boto3.client('textract', region_name='us-east-1')
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return "[Image encoding failed]"
    img_bytes = encoded_image.tobytes()

    try:
        response = client.detect_document_text(Document={'Bytes': img_bytes})
        lines = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
        return '\n'.join(lines)
    except Exception as e:
        return f"[Amazon OCR error: {str(e)}]"


def compare_text(output, truth):
    clean = lambda t: t.replace('\n', '').replace('\r', '').replace(' ', '').strip().lower()
    sm = difflib.SequenceMatcher(None, clean(output), clean(truth))
    cer = 1 - sm.ratio()
    return round(cer * 100, 2)


def save_text(text, engine, base_name, processed):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = "preproc" if processed else "original"
    filename = f"{base_name}_{engine}_{suffix}.txt"
    with open(os.path.join(RESULTS_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text)


def log_to_csv(engine, base_name, cer, processed):
    import csv
    file_exists = os.path.isfile(CSV_LOG)
    with open(CSV_LOG, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Image", "Engine", "Preprocessed", "CER (%)"])
        writer.writerow([base_name, engine, processed, cer])


def process_image(base_name):
    image_path = os.path.join(IMAGE_DIR, f"{base_name}.jpg")
    text_path = os.path.join(TEXT_DIR, f"{base_name}.txt")

    if not os.path.exists(image_path) or not os.path.exists(text_path):
        print(f"[WARNING] Missing file for '{base_name}'")
        return

    normal = cv2.imread(image_path)
    #preproc = preprocess_image(image_path, use_clahe=False, use_threshold=False, use_morph=False, scale=1.0)
    #preproc = preprocess_image(image_path, use_clahe=True, use_threshold=False, use_morph=False)
    #preproc = preprocess_image(image_path, use_clahe=True, use_threshold=True, use_morph=False)
    #preproc = preprocess_image(image_path, use_clahe=True, use_threshold=True, use_morph=True, scale=2.0)
    preproc = preprocess_image(image_path, use_clahe=True, use_threshold=True, use_morph=True, scale=1.5)






    # Spara förbehandlad bild
    preproc_image_path = os.path.join(RESULTS_DIR, f"{base_name}_preproc.jpg")
    cv2.imwrite(preproc_image_path, preproc)

    with open(text_path, 'r', encoding='utf-8') as f:
        ground_truth = f.read()

    engines = {
        "Tesseract": run_tesseract if USE_TESSERACT else None,
        "Google": run_google if USE_GOOGLE else None,
        "Amazon": run_amazon if USE_AMAZON else None,
    }

    for engine_name, runner in engines.items():
        if not runner:
            continue

        for processed, img in [(False, normal), (True, preproc)]:
            output = runner(img)
            save_text(output, engine_name.lower(), base_name, processed)
            cer = compare_text(output, ground_truth)
            log_to_csv(engine_name, base_name, cer, processed)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CSV_LOG):
        os.remove(CSV_LOG)

    for file in os.listdir(IMAGE_DIR):
        if file.endswith(".jpg"):
            base = os.path.splitext(file)[0]
            print(f"[INFO] Processing '{base}'...")
            process_image(base)

    print("[INFO] Done. See results in:", CSV_LOG)


if __name__ == "__main__":
    main()
