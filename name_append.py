import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to the semantic segmentation label image
image_path = "./labels/000030_10.png"

# Load the image and convert it to a NumPy array
label_img = Image.open(image_path)
label_array = np.array(label_img)
print(np.unique(label_array))

# Specify the ID you want to extract
target_id = 17  # Change this to your desired class ID

# Create binary mask
binary_mask = (label_array == target_id).astype(np.uint8)

# Display the binary mask
plt.figure(figsize=(6, 6))
plt.imshow(binary_mask, cmap='gray')
plt.title(f'Binary Mask for ID {target_id}')
plt.axis('off')
plt.show()

