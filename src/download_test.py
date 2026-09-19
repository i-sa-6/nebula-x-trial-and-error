"""
Download script for all 68 Test CSV files from GitHub.
Strictly read-only downloading from public raw URL.
"""
import os
import sys
import time
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import certifi

from src.config import TEST_DIR

BASE_URL = "https://raw.githubusercontent.com/aochinwen/NebulaX-Hackathon-ProblemStatement/main/PS3/02_Datasets/Rail_Corrugation/Test"
ssl_context = ssl.create_default_context(cafile=certifi.where())

def download_single_test_file(file_num, max_retries=3):
    filename = f"Test{file_num}.csv"
    url = f"{BASE_URL}/{filename}"
    out_path = os.path.join(TEST_DIR, filename)
    
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

def download_all_test_files():
    os.makedirs(TEST_DIR, exist_ok=True)
    total_files = 68
    print(f"[*] Starting download of {total_files} test files to {TEST_DIR}...")
    start_time = time.time()
    
    completed = 0
    failed = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_test_file, i): f"Test{i}.csv" for i in range(1, total_files + 1)}
        for future in as_completed(futures):
            fname, success, msg = future.result()
            completed += 1
            if not success:
                failed.append(fname)
                print(f"[-] [{completed}/{total_files}] ERROR {fname}: {msg}")
            else:
                if completed % 10 == 0 or completed == total_files or "Downloaded" in msg:
                    elapsed = time.time() - start_time
                    rate = completed / max(elapsed, 0.1)
                    print(f"[+] [{completed}/{total_files}] {fname}: {msg} | Elapsed: {elapsed:.1f}s ({rate:.2f} files/s)")
                    
    elapsed_total = time.time() - start_time
    print(f"\n[✓] Test download finished in {elapsed_total:.1f}s. Successfully downloaded: {completed - len(failed)}/{total_files}.")
    if failed:
        print(f"[!] Warning: {len(failed)} files failed: {failed}")

if __name__ == "__main__":
    download_all_test_files()
