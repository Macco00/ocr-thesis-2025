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

PARAMS_LOG = os.path.join(RESULTS_DIR, "params_log.csv")

def log_best_params(base_name, C, block_size, cer):
    import csv
    file_exists = os.path.isfile(PARAMS_LOG)
    with open(PARAMS_LOG, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Image", "C", "Block Size", "CER (%)"])
        writer.writerow([base_name, C, block_size, cer])

def preprocess_image(image_path, scale=1.0,
                     use_clahe=False,
                     clahe_clip=1.5,
                     clahe_tile=(8, 8),
                     use_threshold=False,
                     block_size=15,
                     C=5,
                     use_morph=False):

    gray = load_as_grayscale(image_path)

    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
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

def find_best_clahe_only(image_path, truth_path, scale=1.0):
    best_cer = float('inf')
    best_output = ""
    best_params = {}

    ground_truth = open(truth_path, 'r', encoding='utf-8').read()
    gray = load_as_grayscale(image_path)

    for clip in [1.0, 1.5, 2.0, 3.0]:
        for tile in [(4, 4), (8, 8), (16, 16)]:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                processed = clahe.apply(gray)

                output = run_tesseract(processed)
                cer = compare_text(output, ground_truth)

                print(f"[CLAHE clip={clip}, tile={tile}] CER = {cer}%")

                if cer < best_cer:
                    best_cer = cer
                    best_output = output
                    best_params = {
                        "CLAHE": True,
                        "clipLimit": clip,
                        "tileGridSize": tile,
                        "CER": cer
                    }

            except Exception as e:
                print(f"[ERROR] CLAHE clip={clip}, tile={tile}: {e}")

    return best_output, best_params



def find_best_threshold_params(image_path, truth_path, scale=1.0):
    best_cer = float('inf')
    best_output = ""
    best_params = {}

    ground_truth = open(truth_path, 'r', encoding='utf-8').read()

    for C in range(1, 11):  # C = 1–10
        for block_size in range(7, 50, 2):  # Block size = 7–49, steg 2 = bara udda
            try:
                preproc = preprocess_image(
                    image_path,
                    use_clahe=False,
                    use_threshold=True,
                    use_morph=False,
                    scale=scale,
                    block_size=block_size,
                    C=C
                )

                output = run_tesseract(preproc)
                cer = compare_text(output, ground_truth)

                print(f"[C={C}, Block={block_size}] CER = {cer}%")

                if cer < best_cer:
                    best_cer = cer
                    best_output = output
                    best_params = {
                        "C": C,
                        "block_size": block_size,
                        "CER": cer
                    }

            except Exception as e:
                print(f"[ERROR] C={C}, Block={block_size}: {e}")

    print(f"\nBest result: CER={best_params['CER']}% with C={best_params['C']}, block_size={best_params['block_size']}")
    return best_output, best_params

def auto_preprocess_tuner(image_path, truth_path, scale=1.0):
    from copy import deepcopy

    print(f"[DEBUG] Starting tuning on: {image_path}")
    gray = load_as_grayscale(image_path)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    ground_truth = open(truth_path, 'r', encoding='utf-8').read()
    best_cer = float('inf')
    best_result = None
    best_params = {}

    # === Step 1: CLAHE only ===
    best_clahe_img = None
    best_clahe_params = {}
    print("[DEBUG] Trying CLAHE alone...")
    for clip in [1.0, 1.5, 2.0]:
        for tile in [(4, 4), (8, 8), (16, 16)]:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                processed = clahe.apply(gray)
                output = run_tesseract(processed)
                cer = compare_text(output, ground_truth)
                print(f"  CLAHE clip={clip}, tile={tile} → CER={cer}")

                if cer < best_cer:
                    best_cer = cer
                    best_clahe_img = processed.copy()
                    best_clahe_params = {"clip": clip, "tile": tile}
            except Exception as e:
                print(f"  [ERROR] CLAHE failed: {e}")
                continue

    # === Step 1.5: Apply threshold on best CLAHE image ===
    if best_clahe_img is not None:
        print("[DEBUG] Refining with threshold on best CLAHE result...")
        for C in range(1, 11):
            for block_size in range(7, 50, 2):
                try:
                    threshed = cv2.adaptiveThreshold(
                        best_clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, block_size, C
                    )
                    output = run_tesseract(threshed)
                    cer = compare_text(output, ground_truth)
                    print(f"    CLAHE→Thresh C={C}, block={block_size} → CER={cer}")
                    if cer < best_cer:
                        best_cer = cer
                        best_result = deepcopy(threshed)
                        best_params = {
                            "use_clahe": True,
                            "clahe_clip": best_clahe_params["clip"],
                            "clahe_tile": best_clahe_params["tile"],
                            "use_threshold": True,
                            "block_size": block_size,
                            "C": C,
                            "CER": cer
                        }
                except Exception as e:
                    print(f"    [ERROR] CLAHE→Thresh failed: {e}")
                    continue
    else:
        print("[WARN] No valid CLAHE output to continue with.")

    # === Step 2: Threshold only ===
    best_thresh_img = None
    best_thresh_params = {}
    print("[DEBUG] Trying Threshold alone...")
    for C in range(1, 11):
        for block_size in range(7, 50, 2):
            try:
                threshed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, C
                )
                output = run_tesseract(threshed)
                cer = compare_text(output, ground_truth)
                print(f"  Thresh C={C}, block={block_size} → CER={cer}")
                if cer < best_cer:
                    best_cer = cer
                    best_thresh_img = threshed.copy()
                    best_thresh_params = {"C": C, "block_size": block_size}
            except Exception as e:
                print(f"  [ERROR] Thresh failed: {e}")
                continue

    # === Step 2.5: Apply CLAHE on best thresholded image ===
    if best_thresh_img is not None:
        print("[DEBUG] Refining with CLAHE on best threshold result...")
        for clip in [1.0, 1.5, 2.0]:
            for tile in [(4, 4), (8, 8), (16, 16)]:
                try:
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                    processed = clahe.apply(best_thresh_img)
                    output = run_tesseract(processed)
                    cer = compare_text(output, ground_truth)
                    print(f"    Thresh→CLAHE clip={clip}, tile={tile} → CER={cer}")
                    if cer < best_cer:
                        best_cer = cer
                        best_result = deepcopy(processed)
                        best_params = {
                            "use_clahe": True,
                            "clahe_clip": clip,
                            "clahe_tile": tile,
                            "use_threshold": True,
                            "block_size": best_thresh_params["block_size"],
                            "C": best_thresh_params["C"],
                            "CER": cer
                        }
                except Exception as e:
                    print(f"    [ERROR] Thresh→CLAHE failed: {e}")
                    continue
    else:
        print("[WARN] No valid Threshold output to continue with.")

    # === Fallback ===
    if best_result is None:
        print("[WARN] No valid combination found. Using original image as fallback.")
        best_result = gray
        best_params = {
            "use_clahe": False,
            "use_threshold": False,
            "CER": 100.0
        }


    print(f"[BEST] CER={best_cer}% with params: {best_params}")
    return best_result, best_params




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
    print(f"[INFO] Using preprocessing on: {base_name}")

    image_path = os.path.join(IMAGE_DIR, f"{base_name}.jpg")
    text_path = os.path.join(TEXT_DIR, f"{base_name}.txt")

    if not os.path.exists(image_path) or not os.path.exists(text_path):
        print(f"[WARNING] Missing file for '{base_name}'")
        return

    normal = cv2.imread(image_path)
    preproc = auto_preprocess_image(image_path, scale=1.2)

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

def is_almost_binary(gray):
    # Histogram: hur mycket som är nästan svart eller vitt
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size
    dark = hist[:30].sum()
    bright = hist[225:].sum()
    binary_ratio = (dark + bright) / total_pixels
    print(f"[DEBUG] Binary ratio: {binary_ratio:.3f}")
    return binary_ratio > 0.99

def auto_preprocess_image(image_path, scale=1.0):
    gray = load_as_grayscale(image_path)

    # Skippa preprocessing om bilden redan är nästan binär
    resized_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if is_almost_binary(resized_gray):
        print("[INFO] Skipping preprocessing (binary-like image)")
        return gray

    # Hitta ground truth
    truth_path = os.path.join(TEXT_DIR, os.path.basename(image_path).replace('.jpg', '.txt'))
    if os.path.exists(truth_path):
        print("[INFO] Tuning CLAHE + threshold settings...")
        _, best_params = auto_preprocess_tuner(image_path, truth_path, scale=scale)

        # Logga bästa inställningar
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        log_best_params(
            base_name,
            best_params.get("C", "-"),
            best_params.get("block_size", "-"),
            best_params["CER"]
        )

        return preprocess_image(
            image_path,
            use_clahe=best_params.get("use_clahe", False),
            clahe_clip=best_params.get("clahe_clip", 1.5),
            clahe_tile=best_params.get("clahe_tile", (8, 8)),
            use_threshold=best_params.get("use_threshold", False),
            block_size=best_params.get("block_size", 15),
            C=best_params.get("C", 5),
            use_morph=False,
            scale=scale
        )

    # Fallback om ingen ground truth finns
    print("[INFO] Applying default CLAHE (no ground truth found)")
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    return clahe.apply(resized_gray)




def main1():
    interactive_preprocess_view("C:/Users/marcu/Desktop/GitHub/ocr-thesis-2025/picture/image15.jpg")
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

def analyze_preprocessing_results():
    import pandas as pd

    if not os.path.exists(CSV_LOG):
        print("[ERROR] CSV-logg saknas.")
        return

    df = pd.read_csv(CSV_LOG)
    if "Preprocessed" not in df.columns:
        print("[ERROR] Saknar kolumn 'Preprocessed'")
        return

    grouped = df[df["Engine"] == "Tesseract"].pivot(index="Image", columns="Preprocessed", values="CER (%)")
    grouped.columns = ["Original CER", "Preprocessed CER"]
    grouped["Difference"] = grouped["Original CER"] - grouped["Preprocessed CER"]
    grouped["Improved"] = grouped["Difference"] > 0

    print("\n=== Resultatöversikt ===")
    print(grouped)

    total = len(grouped)
    improved = grouped["Improved"].sum()
    worsened = total - improved
    avg_before = grouped["Original CER"].mean()
    avg_after = grouped["Preprocessed CER"].mean()
    avg_diff = grouped["Difference"].mean()

    print(f"\nTotalt antal bilder: {total}")
    print(f"Förbättrades: {improved} bilder")
    print(f"Försämrades: {worsened} bilder")
    print(f"Genomsnittlig CER före: {avg_before:.2f}%")
    print(f"Genomsnittlig CER efter: {avg_after:.2f}%")
    print(f"Genomsnittlig förbättring: {avg_diff:.2f} procentenheter")


if __name__ == "__main__":
    main()
    analyze_preprocessing_results()

