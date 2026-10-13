import random
from PIL import Image
from pathlib import Path
import numpy as np  
import matplotlib.pyplot as plt 

data_path = Path(__file__).parent / "data"
images_path = data_path/"Original CT Scans"
print(images_path)
print(images_path.exists())

#Get all image paths 
images_path_list = list(images_path.rglob("*.jpg"))

#Pick a random image path 
random_image_path = random.choice(images_path_list)

#Get the image class from the path name
image_class = random_image_path.parent.stem

#Opening the image 
img = Image.open(random_image_path)

#Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
print(f"Image mode: {img.mode}")


####Visualizing using matplotlib#####
#Turning the image into a numpy array
image_as_array = np.asarray(img)

#Plot the image with matplotlib 
plt.figure(figsize=(10,10))
plt.imshow(image_as_array)
plt.title(f"Image class {image_class} | Image shape {image_as_array.shape} -> [Height,Width,Color_Channels](HWC)")
plt.axis(False)
plt.show()


####  Class Distribution  ####
# Count the number of images per class (each subfolder = one class)
class_counts = {}
for class_dir in images_path.iterdir():
    if class_dir.is_dir():
        count = len(list(class_dir.glob("*.jpg")))
        class_counts[class_dir.name] = count 

print("\n Class distribution:")
for class_name, count in class_counts.items():
    print(f" {class_name}: {count} images")

fig, ax = plt.subplots()
ax.pie(class_counts.values(),labels=class_counts.keys(), autopct='%1.1f%%',shadow=True, startangle=180)
ax.set_title("Class distribution")
plt.show()


