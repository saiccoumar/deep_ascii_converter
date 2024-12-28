import os
from PIL import Image, ImageOps

# Define the directory containing PNG images
directory = "examples/oscii_ex_building"

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(".png"):  # Check for PNG files
        filepath = os.path.join(directory, filename)
        
        # Open the image
        with Image.open(filepath) as img:
            # Invert the image
            inverted_image = ImageOps.invert(img.convert("RGB"))
            
            # Save the inverted image back (optional: save with a new name)
            inverted_image.save(os.path.join(directory, f"inverted_{filename}"))

print("All PNG images have been inverted.")