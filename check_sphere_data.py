import numpy as np
import os

# path to the folder containing .npz files
folder_path = "./data_nspheres_1"

# loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".npz"):
        file_path = os.path.join(folder_path, filename)
        try:
            # load the .npz file
            data = np.load(file_path)
            
            if "spheres" in data:
                print(f"File: {filename}")
                print(data["spheres"])
                print("-" * 50)
            else:
                print(f"File: {filename} does not contain 'spheres'")
                
        except Exception as e:
            print(f"Error opening {filename}: {e}")
