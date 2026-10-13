import kagglehub
from pathlib import Path
import os 
import shutil

# Download latest version
path = kagglehub.dataset_download("azaemon/preprocessed-ct-scans-for-covid19")

print("Path to dataset files:", path)

#Source
source = Path(path)

#Destination
DATA_DIR = Path("data")
os.makedirs(DATA_DIR, exist_ok=True)

#Copy dataset 
for item in source.iterdir():
    destination = DATA_DIR / item.name
    
    if item.is_dir():
        shutil.copytree(item,destination,dirs_exist_ok=True)
    else:
        shutil.copy2(item,destination)

print("Dataset copied successfully to: ", DATA_DIR)