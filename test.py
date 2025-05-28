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
USE_GOOGLE = False
USE_AMAZON = False
SHOW_PREPROC_IMAGE = False

RESULTS_DIR = "results"
IMAGE_DIR = "picture"
TEXT_DIR = "text"
CSV_LOG = os.path.join(RESULTS_DIR, "metrics_log.csv")


def preprocess_image(image_path, use_clahe=True, use_threshold=True, use_morph=True, scale=1.2,
                     clip_limit=1.5, tile_grid=8, block_size=33, C=2, morph_kernel=1):
    gray = load_as_grayscale(image_path)

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        gray = clahe.apply(gray)

    if use_threshold:
        if block_size % 2 == 0:
            block_size += 1
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )

    if use_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
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

def load_as_grayscale(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"[ERROR] Failed to read image at {image_path}")
    
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img.copy()

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
    preproc = preprocess_image(image_path, use_clahe=False, use_threshold=True, use_morph=False, scale=2.0)






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

def interactive_preprocess_view(image_path):
    def nothing(x): pass

    cv2.namedWindow('Preprocessing Controls', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('Use CLAHE', 'Preprocessing Controls', 1, 1, nothing)
    cv2.createTrackbar('CLAHE Clip', 'Preprocessing Controls', 15, 40, nothing)
    cv2.createTrackbar('CLAHE Grid', 'Preprocessing Controls', 8, 16, nothing)

    cv2.createTrackbar('Use Threshold', 'Preprocessing Controls', 1, 1, nothing)
    cv2.createTrackbar('Block Size', 'Preprocessing Controls', 15, 50, nothing)
    cv2.createTrackbar('C', 'Preprocessing Controls', 5, 20, nothing)

    cv2.createTrackbar('Use Morph', 'Preprocessing Controls', 1, 1, nothing)
    cv2.createTrackbar('Kernel Size', 'Preprocessing Controls', 1, 10, nothing)

    cv2.createTrackbar('Scale x10', 'Preprocessing Controls', 10, 30, nothing)

    while True:
        use_clahe = cv2.getTrackbarPos('Use CLAHE', 'Preprocessing Controls') == 1
        clip = cv2.getTrackbarPos('CLAHE Clip', 'Preprocessing Controls') / 10.0
        grid = max(1, cv2.getTrackbarPos('CLAHE Grid', 'Preprocessing Controls'))

        use_thresh = cv2.getTrackbarPos('Use Threshold', 'Preprocessing Controls') == 1
        block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing Controls')
        if block_size % 2 == 0:
            block_size += 1
        C = cv2.getTrackbarPos('C', 'Preprocessing Controls')

        use_morph = cv2.getTrackbarPos('Use Morph', 'Preprocessing Controls') == 1
        kernel_size = max(1, cv2.getTrackbarPos('Kernel Size', 'Preprocessing Controls'))

        scale = cv2.getTrackbarPos('Scale x10', 'Preprocessing Controls') / 10.0

        processed = preprocess_image(
            image_path,
            use_clahe=use_clahe,
            use_threshold=use_thresh,
            use_morph=use_morph,
            scale=scale,
            clip_limit=clip,
            tile_grid=grid,
            block_size=block_size,
            C=C,
            morph_kernel=kernel_size
        )

        cv2.imshow('Preprocessed Image', processed)

        key = cv2.waitKey(100)
        if key == 27:
            break
        elif key == ord('s'):
            save_path = os.path.join(RESULTS_DIR, "interactive_saved.jpg")
            cv2.imwrite(save_path, processed)
            print(f"[INFO] Image saved to {save_path}")

    cv2.destroyAllWindows()



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
    #test_image = os.path.join(IMAGE_DIR, "image8.jpg")
    #interactive_preprocess_view(test_image)
    main()
