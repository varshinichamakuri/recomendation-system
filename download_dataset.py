import os
import urllib.request
import time

def download_file(url, dest_path, max_retries=5):
    dest_dir = os.path.dirname(dest_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {url} (Attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, 'wb') as out_file:
                content_length = response.getheader('Content-Length')
                expected_size = int(content_length) if content_length else None
                
                block_size = 1024 * 64
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                if expected_size and downloaded < expected_size:
                    raise Exception(f"Incomplete download: got {downloaded} out of {expected_size} bytes")
            
            print(f"Successfully downloaded to {dest_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url} on attempt {attempt}: {e}")
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(2)
    return False

def download_and_extract_movielens(dest_dir="data"):
    extracted_dir = os.path.join(dest_dir, "ml-100k")
    
    # Github Raw URL mirrors for MovieLens 100k
    ratings_url = "https://raw.githubusercontent.com/cfgranda/ps4ds/main/data/ml-100k/u.data"
    movies_url = "https://raw.githubusercontent.com/cfgranda/ps4ds/main/data/ml-100k/u.item"
    
    u_data_path = os.path.join(extracted_dir, "u.data")
    u_item_path = os.path.join(extracted_dir, "u.item")
    
    # Check if files already exist and are non-empty
    valid_ratings = os.path.exists(u_data_path) and os.path.getsize(u_data_path) > 0
    valid_movies = os.path.exists(u_item_path) and os.path.getsize(u_item_path) > 0
    if valid_ratings and valid_movies:
        print("Dataset already downloaded and exists.")
        return

    if os.path.exists(u_data_path) and os.path.getsize(u_data_path) == 0:
        print(f"Removing empty file: {u_data_path}")
        os.remove(u_data_path)
    if os.path.exists(u_item_path) and os.path.getsize(u_item_path) == 0:
        print(f"Removing empty file: {u_item_path}")
        os.remove(u_item_path)

    print("Fetching MovieLens 100k from GitHub mirror...")
    success_ratings = download_file(ratings_url, u_data_path)
    success_movies = download_file(movies_url, u_item_path)
    
    if success_ratings and success_movies:
        print("Successfully downloaded both dataset files!")
    else:
        print("Failed to download dataset files from mirror.")

if __name__ == "__main__":
    download_and_extract_movielens()
