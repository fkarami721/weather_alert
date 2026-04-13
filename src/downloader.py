import os
import requests
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

# def fetch_grib_files(variable_code, base_url, target_dir, max_threads=10):
#     os.makedirs(target_dir, exist_ok=True)
#     # Your existing threaded download logic goes here.
#     # Yield or return the paths to the downloaded files.
#     pass


def download_HRDPS_data(data_dir, folder_name, variable_code, base_url):
    # 1. Define the folder name and create it if it doesn't exist
    target_dir = os.path.join(data_dir, folder_name, variable_code)
    os.makedirs(target_dir, exist_ok=True)

    string_list = [f"{i:03}" for i in range(0, 49)]

    # print("Starting download of GRIB2 files...")
    
    # Helper function for threaded downloading
    def download_file(session, url, filepath):
        # FAST FAIL: Check if file exists BEFORE downloading
        if os.path.exists(filepath):
            return f"Already exists: {os.path.basename(filepath)}"
            
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return f"Downloaded: {os.path.basename(filepath)}"
        except Exception as e:
            return f"Failed {url}: {e}"

    max_workers = os.cpu_count() or 1

    
    # Use a Session to reuse the underlying TCP connection to the server
    with requests.Session() as session:
        # Use ThreadPoolExecutor to download files in parallel (10 at a time)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in string_list:
                url = f"{base_url}{i}/{folder_name}_{variable_code}_RLatLon0.0225_PT{i}H.grib2"
                filename = url.split('/')[-1]
                filepath = os.path.join(target_dir, filename)
                
                # Submit the download task to the thread pool
                futures.append(executor.submit(download_file, session, url, filepath))
            
            # Optional: wait for all to finish and print progress
            for future in as_completed(futures):
                # You can uncomment the next line to see real-time download progress
                # print(future.result()) 
                pass
    return target_dir            