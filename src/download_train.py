import os
import sys
import time
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import certifi

BASE_URL = "https://raw.githubusercontent.com/aochinwen/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/Rail_Corrugation/Train"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Train")
LABELS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Train_Labels.csv")

# Create SSL context using certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

def download_single_file(filename, max_retries=3):
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    # Check if already downloaded and valid size (> 10 MB)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10 * 1024 * 1024:
        return filename, True, "Already exists"

    for attempt in range(1, max_retries + 1):
        try:
            temp_path = out_path + f".tmp_{attempt}"
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            )
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response, open(temp_path, 'wb') as out_file:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    
            if os.path.getsize(temp_path) > 10 * 1024 * 1024:
                os.replace(temp_path, out_path)
                return filename, True, f"Downloaded ({os.path.getsize(out_path)/(1024*1024):.1f} MB)"
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries:
                return filename, False, str(e)
            time.sleep(1.0 * attempt)
            
    return filename, False, "Failed after retries"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(LABELS_FILE)
    filenames = df['filename'].tolist()
    total = len(filenames)
    
    print(f"[*] Starting download of {total} training files to {OUTPUT_DIR} using certifi SSL...")
    start_time = time.time()
    
    # Prioritize fault files first so we have them immediately
    fault_files = df[df['label'] != 'Normal']['filename'].tolist()
    normal_files = df[df['label'] == 'Normal']['filename'].tolist()
    ordered_files = fault_files + normal_files
    
    completed = 0
    failed = []
    
    # Use 12 threads for faster download
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_single_file, fname): fname for fname in ordered_files}
        for future in as_completed(futures):
            fname, success, msg = future.result()
            completed += 1
            if not success:
                failed.append(fname)
                print(f"[-] [{completed}/{total}] ERROR {fname}: {msg}")
            else:
                if completed % 10 == 0 or completed == total or "Downloaded" in msg:
                    elapsed = time.time() - start_time
                    speed = completed / max(elapsed, 0.1)
                    print(f"[+] [{completed}/{total}] {fname}: {msg} | Elapsed: {elapsed:.1f}s ({speed:.2f} files/s)")
                    
    elapsed_total = time.time() - start_time
    print(f"\n[✓] Download finished in {elapsed_total:.1f}s. Successfully downloaded: {completed - len(failed)}/{total}.")
    if failed:
        print(f"[!] Warning: {len(failed)} files failed: {failed}")

if __name__ == "__main__":
    main()
