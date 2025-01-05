import os
from PIL import Image, ImageOps

# for i in range(11,21):
#     # Define the directory containing PNG images
#     directory = f"examples/{i}"
#     inverted_dir = os.path.join(directory, "inverted")

#     # Create the 'inverted' directory if it doesn't exist
#     os.makedirs(inverted_dir, exist_ok=True)

#     # Iterate over all files in the directory
#     for filename in os.listdir(directory):
#         if filename.lower().endswith(".png"):  # Check for PNG files
#             filepath = os.path.join(directory, filename)
            
#             # Open the image
#             with Image.open(filepath) as img:
#                 # Invert the image
#                 inverted_image = ImageOps.invert(img.convert("RGB"))
                
#                 # Save the inverted image in the 'inverted' directory
#                 inverted_image.save(os.path.join(inverted_dir, f"inverted_{filename}"))

#     print("All PNG images have been inverted.")



# Define the directory containing PNG images
directory = f"examples/shapes_ex"
inverted_dir = os.path.join(directory, "inverted")

# Create the 'inverted' directory if it doesn't exist
os.makedirs(inverted_dir, exist_ok=True)

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.lower().endswith(".png"):  # Check for PNG files
        filepath = os.path.join(directory, filename)
        
        # Open the image
        with Image.open(filepath) as img:
            # Invert the image
            inverted_image = ImageOps.invert(img.convert("RGB"))
            
            # Save the inverted image in the 'inverted' directory
            inverted_image.save(os.path.join(inverted_dir, f"inverted_{filename}"))

print("All PNG images have been inverted.")