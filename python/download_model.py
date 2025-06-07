import os
import sys
import requests
import zipfile
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    print(f"Downloading {url} to {destination}")
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress bar"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        total = len(members)
        print(f"Extracting {zip_path} to {extract_to}")
        for i, member in enumerate(members):
            zip_ref.extract(member, extract_to)
            percent = int((i / total) * 100)
            sys.stdout.write(f"\rExtracting: {percent}% complete")
            sys.stdout.flush()
    print("\nExtraction complete!")

def main():
    MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    MODEL_FILENAME = os.path.basename(MODEL_URL)
    MODEL_DIR = os.path.join("models", "vosk")
    ZIP_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # Make sure directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")):
        print("Model already downloaded and extracted.")
        return
    
    # Download the file
    download_file(MODEL_URL, ZIP_PATH)
    
    # Extract the zip
    extract_zip(ZIP_PATH, MODEL_DIR)
    
    # Clean up
    os.remove(ZIP_PATH)
    print("Model downloaded and extracted successfully.")

if __name__ == "__main__":
    main() 