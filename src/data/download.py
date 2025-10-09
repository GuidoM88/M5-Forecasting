import os
import zipfile
from pathlib import Path

def download_m5_dataset():
    """Download M5 dataset from Kaggle"""
    
    # Create data directory if not exists
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading M5 dataset from Kaggle...")
    os.system("kaggle competitions download -c m5-forecasting-accuracy -p data/raw")
    
    # Unzip files
    print("Extracting files...")
    zip_file = data_dir / "m5-forecasting-accuracy.zip"
    
    if zip_file.exists():
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file
        zip_file.unlink()
        print(f"Dataset downloaded successfully in {data_dir}")
    else:
        print("Error: zip file not found")

if __name__ == "__main__":
    download_m5_dataset()
