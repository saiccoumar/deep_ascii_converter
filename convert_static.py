# http://www.jave.de/image2ascii/algorithms.html
# https://aa-project.sourceforge.net/aalib/
# https://github.com/JEphron/TEXTFLIX/blob/master/main.py
# https://github.com/StefanPitur/Edge-detection---Canny-detector
from legacy_static import *
from model_architectures import *
from canny_edge_detector_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from torchvision.transforms import ToPILImage
from skimage.feature import hog
from sklearn import svm
import joblib
from scipy.spatial.distance import cosine



def apply_autoencoder(img_array, dimensions):
    """
    Apply autoencoder to an input image array by splitting it into tiles,
    processing each tile, and reconstructing the full image
    
    Args:
        img_array (numpy.ndarray): Input image as numpy array of any size
        dimensions (int): Either 10 or 64, determining which autoencoder to use
        
    Returns:
        numpy.ndarray: Decoded image as numpy array
    """
    start_time = time.time()
    # Input validation
    if dimensions not in [10, 64]:
        raise ValueError("dimensions must be either 10 or 64")
    
    # Calculate padding needed
    pad_height = dimensions - (img_array.shape[0] % dimensions) if img_array.shape[0] % dimensions != 0 else 0
    pad_width = dimensions - (img_array.shape[1] % dimensions) if img_array.shape[1] % dimensions != 0 else 0
    
    # Pad the image
    padded_img = np.pad(img_array, 
                       ((0, pad_height), (0, pad_width)), 
                       mode='constant', 
                       constant_values=0)
    
    # Calculate number of tiles in each dimension
    num_tiles_height = math.ceil(padded_img.shape[0] / dimensions)
    num_tiles_width = math.ceil(padded_img.shape[1] / dimensions)
    total_tiles = num_tiles_height * num_tiles_width
    
    # Create tensor to hold all tiles
    tiles = np.zeros((total_tiles, dimensions, dimensions))
    
    # Split image into tiles
    tile_idx = 0
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            h_start = i * dimensions
            h_end = (i + 1) * dimensions
            w_start = j * dimensions
            w_end = (j + 1) * dimensions
            tiles[tile_idx] = padded_img[h_start:h_end, w_start:w_end]
            tile_idx += 1
    
    # Convert tiles to tensor
    tiles_tensor = torch.from_numpy(tiles).float().unsqueeze(1)  # Add channel dimension
    
    # Initialize appropriate autoencoder
    if dimensions == 10:
        model = Autoencoder10()
    else:
        model = Autoencoder64()
    
    # Load model weights
    model_path = f"artifacts/autoencoder_ascii_{dimensions}_x_{dimensions}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Process tiles in batches
    with torch.no_grad():
        _, decoded_tiles = model(tiles_tensor)
    
    # Convert back to numpy and remove channel dimension
    decoded_tiles = decoded_tiles.squeeze(1).numpy()
    
    # Reconstruct the full image
    decoded_img = np.zeros(padded_img.shape)
    tile_idx = 0
    for i in range(num_tiles_height):
        for j in range(num_tiles_width):
            h_start = i * dimensions
            h_end = (i + 1) * dimensions
            w_start = j * dimensions
            w_end = (j + 1) * dimensions
            decoded_img[h_start:h_end, w_start:w_end] = decoded_tiles[tile_idx]
            tile_idx += 1
    
    # Remove padding to return to original dimensions
    if pad_height > 0 or pad_width > 0:
        decoded_img = decoded_img[:img_array.shape[0], :img_array.shape[1]]
    display_samples = False
    if display_samples:
        # # Display original and decoded images
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(decoded_img, cmap='gray')
        plt.title('Autoencoded Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    return decoded_img, elapsed_time


def convert_edge_aiss(fi,save, binarize=False, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, ae = False ,ded=False):
    start_time = time.time()
    factor = factor * inc / 10 
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    
    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))
    img_array_edges = np.array(final).astype(float)
    def log_polar_histogram(center, image, bins_r=5, bins_theta=12):
        """Create log-polar histogram for a given center point"""
        h, w = image.shape
        y, x = np.ogrid[:h, :w]
        radius = min(h,w) //2
        # Calculate r and theta for each pixel
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        theta = np.arctan2(y - center[1], x - center[0])
        
        # Create log-spaced radial bins
        r_bins = np.logspace(0.1, np.log2(radius), bins_r + 1, base=2)
        theta_bins = np.linspace(-np.pi, np.pi, bins_theta + 1)
        
        # Initialize histogram
        hist = np.zeros((bins_r, bins_theta))
        
        # Fill histogram bins
        for i in range(bins_r):
            for j in range(bins_theta):
                mask = (r >= r_bins[i]) & (r < r_bins[i + 1]) & \
                    (theta >= theta_bins[j]) & (theta < theta_bins[j + 1])
                hist[i, j] = np.sum(image[mask])
        # print(hist.flatten())
        return hist.flatten()

    def match_ascii_characters(grid_cell, ascii_histograms):
        """
        Match the log-polar histogram of a grid cell with the best ASCII character.
        """
        cell_hist = log_polar_histogram((grid_cell.shape[0] // 2, grid_cell.shape[1] // 2), grid_cell)
        # best_match, _ = min(ascii_histograms.items(), key=lambda item: np.linalg.norm(cell_hist - item[1]))
        # best_match,_ = min(ascii_histograms.items(), key=lambda item: euclidean(cell_hist, item[1]))
        # best_match, _ = max(ascii_histograms.items(), key=lambda item: 1 - cosine(cell_hist, item[1]))
        best_match, _ = min(ascii_histograms.items(), key=lambda item: np.mean(np.abs(cell_hist - item[1])))

        return best_match  # Return the ASCII character (key) with the best match

    def compute_ascii_histograms(char_images, bins_r=5, bins_theta=12):
        """
        Compute log-polar histograms for all ASCII character images.
        """
        histograms = {}
        for char, img in char_images.items():
            center = (img.shape[1] // 2, img.shape[0] // 2)
            hist = log_polar_histogram(center, img, bins_r=bins_r, bins_theta=bins_theta)
            histograms[char] = hist
        return histograms
    # 3: Divide into grid cells
    grid_size = inc,inc # Example grid size

    fin = ""
    end_time_1 = time.time()
    ascii_histograms = compute_ascii_histograms(generate_ascii_images("arial.ttf",image_size=inc))
    # print(ascii_histograms)

    for i in range(0, img_array_edges.shape[0], grid_size[1]):
        row = []
        for j in range(0, img_array_edges.shape[1], grid_size[0]):
            grid_cell = img_array_edges[i:i+grid_size[1], j:j+grid_size[0]]
            if grid_cell.size == 0:
                continue
            matched_char = match_ascii_characters(grid_cell, ascii_histograms)
            row.append(matched_char)
        fin += ("".join(row))
        fin += "\n"

    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")
    

def convert_edge_v3(fi,save, binarize=False, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, ae = False ,ded=False):
    start_time = time.time()
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)

    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=10)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")

   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    # Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            min = 0
            min_dist = np.linalg.norm(edge_vectors3[0]-subs[0].flatten())
            # if not (np.all(subs == 0)):
            for j in range(0,len(edge_vectors3)):
                new_mind_dist = np.linalg.norm(edge_vectors3[j]-subs[0].flatten())

                if (new_mind_dist<min_dist):
                    min = j
                    min_dist = new_mind_dist
            
            
            
            fin = fin + chr(min+32)
        fin = fin + '\n'
    

    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")


def convert_edge_v4(fi,save, binarize=False, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, ae = False,ded=False):
    start_time = time.time()
    factor = factor * 6.4
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    print(max_value)
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    # if binarize:
    #     f_vec = np.vectorize(f4)
    #     img_array_edges = f_vec(img_array_edges)

    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=64)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 64
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    #     Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            min = 0
            min_dist = np.linalg.norm(edge_vectors4[0]-subs[0].flatten())
            # if not (np.all(subs == 0)):
            for j in range(0,len(edge_vectors4)):
                new_mind_dist = np.linalg.norm(edge_vectors4[j]-subs[0].flatten())

                if (new_mind_dist<min_dist):
                    min = j
                    min_dist = new_mind_dist
            
            
            
            fin = fin + chr(min+32)
        fin = fin + '\n'
    

    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_knn(fi, save, binarize=False, inc=10, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable=True, ae=False, ded=False):
    start_time = time.time()
    factor = factor * inc / 10 

    # 0: Import KNN Model
    if hog_enable:
        loaded_clf = joblib.load(f'artifacts/knn_model_hog_{inc}_x_{inc}.pkl')
    else:
        loaded_clf = joblib.load(f'artifacts/knn_model_{inc}_x_{inc}.pkl')

    def extract_hog_features(image):
        hog_features = []

        features = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True)
        hog_features.append(features)
        return np.array(hog_features)
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    
    tiles = []
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    # Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            # print(subs.shape)
            
        
            if hog_enable:
                subs_hog = extract_hog_features(subs[0])[0]
                tiles.append(subs_hog)
            else: 
                tiles.append(subs.flatten().reshape(1,-1)[0])
    
                
        #     fin = fin + chr(y_pred+32)
        # fin = fin + '\n'
    print(np.array(tiles).shape)
    predictions = [int(elem) for elem in loaded_clf.predict(tiles)]
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(predictions[index] + 32)
            fin += char
            index += 1
        fin += '\n'
    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_svm(fi,save, binarize = False, inc = 10, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable = True, ae = False,ded=False):
    start_time = time.time()
    factor = factor * inc / 10 
    # 0: Import SVM Model
    if hog_enable:
        loaded_clf = joblib.load(f'artifacts/svm_model_hog_{inc}_x_{inc}.pkl')
    else:
        loaded_clf = joblib.load(f'artifacts/svm_model_{inc}_x_{inc}.pkl')
    def extract_hog_features(image):
        hog_features = []

        features = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True)
        hog_features.append(features)
        return np.array(hog_features)
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    
    tiles = []
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    # Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            # print(subs.shape)
            
        
            if hog_enable:
                subs_hog = extract_hog_features(subs[0])[0]
                tiles.append(subs_hog)
            else: 
                tiles.append(subs.flatten().reshape(1,-1)[0])
    
                
        #     fin = fin + chr(y_pred+32)
        # fin = fin + '\n'
    print(np.array(tiles).shape)
    predictions = [int(elem) for elem in loaded_clf.predict(tiles)]
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(predictions[index] + 32)
            fin += char
            index += 1
        fin += '\n'
    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_random_forest(fi,save, binarize=False, inc = 64,  factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable=True, ae = False,ded=False):
    start_time = time.time()
    factor = factor * inc / 10
    # 0: Import SVM Model
    if hog_enable:
        loaded_clf = joblib.load(f'artifacts/random_forest_model_hog_{inc}_x_{inc}.pkl')
    else:
        loaded_clf = joblib.load(f'artifacts/random_forest_model_{inc}_x_{inc}.pkl')
    def extract_hog_features(image):
        hog_features = []

        features = hog(image, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True)
        hog_features.append(features)
        return np.array(hog_features)
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)
    
    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    
    tiles = []
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    # Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            # print(subs.shape)
            
        
            if hog_enable:
                subs_hog = extract_hog_features(subs[0])[0]
                tiles.append(subs_hog)
            else: 
                tiles.append(subs.flatten().reshape(1,-1)[0])
    
                
        #     fin = fin + chr(y_pred+32)
        # fin = fin + '\n'
    print(np.array(tiles).shape)
    predictions = [int(elem) for elem in loaded_clf.predict(tiles)]
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(predictions[index] + 32)
            fin += char
            index += 1
        fin += '\n'
    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_nn(fi,save, binarize=False, inc = 64, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True,  ae = False,ded=False):
    start_time = time.time()
    factor = factor * inc / 10
    device = "cuda"
    # 0: Import SVM Model
    model = NeuralNetwork(inc * inc, 1024, 256, 64, 96)
    model.load_state_dict(torch.load(f'artifacts/nn_ascii_classifier_{inc}_x_{inc}.pth'))
    model.to(device)
    model.eval()


    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    
    tiles = []
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    # Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            # print(subs.shape)
            
        
            # print((subs.flatten().reshape(1,-1)[0]).shape)
            tiles.append(subs.flatten().reshape(1,-1)[0])
       
    
                
        #     fin = fin + chr(y_pred+32)
        # fin = fin + '\n'
    # print(np.array(tiles).shape)
    
    with torch.no_grad():
        outputs = model(torch.tensor(tiles).to(torch.float32).to(device))
        probabilities = F.softmax(outputs, dim=1)
        _, indices = torch.max(probabilities, 1)
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(indices[index] + 32)
            fin += char
            index += 1
        fin += '\n'
    
    # 5: Output to terminal
    print(fin)
    # 6: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} Fseconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")




def convert_edge_cnn(fi,save,binarize=False, inc = 64,factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, ae = False, ded=False):
    start_time = time.time()
    factor = factor * inc / 10
    # 0: Import Model
    device = "cuda"
    model = CNN(input_shape=(1, inc, inc), num_classes=96).to(device)
    model.load_state_dict(torch.load(f'artifacts/cnn_ascii_classifier_{inc}_x_{inc}.pth'))
    model.eval()
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width  = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    else:
        max_value = np.max(img_array_edges)  # Get the maximum value
        if max_value > 1:  # Avoid re-normalizing if already normalized
            img_array_edges /= max_value
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
    print('preprocessing done')

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    tiles = []
    
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            subs = img_array_edges[li:li + inc, i:i + inc]
            # subs = subs / 255.0  # Normalize
            tiles.append(subs)

    tiles = np.array(tiles)
    tiles_tensor = torch.tensor(tiles).unsqueeze(1).to(torch.float32).to(device)

    # 5: Perform batch prediction
    with torch.no_grad():
        outputs, _ = model(tiles_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, indices = torch.max(probabilities, 1)
    
    # 6: Convert predictions to ASCII characters and build the output string
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(indices[index].item() + 32)
            fin += char
            index += 1
        fin += '\n'

    
    # 7: Output to terminal
    print(fin)
    # 8: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_resnet(fi,save, binarize=False,inc = 64, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True,  ae = False, ded=False):
    start_time = time.time()
    factor = factor * inc / 10
    # 0: Import Model
    device = "cuda"
    resnet18_model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=96, grayscale=True).to(device)
    resnet18_model.load_state_dict(torch.load(f'artifacts/resnet18_ascii_classifier_{inc}_x_{inc}.pth'))
    resnet18_model.eval()

    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    else:
        max_value = np.max(img_array_edges)  # Get the maximum value
        if max_value > 1:  # Avoid re-normalizing if already normalized
            img_array_edges /= max_value
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
    print('preprocessing done')

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    tiles = []
    
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            subs = img_array_edges[li:li + inc, i:i + inc]
            # subs = subs / 255.0  # Normalize
            tiles.append(subs)

    tiles = np.array(tiles)
    tiles_tensor = torch.tensor(tiles).unsqueeze(1).to(torch.float32).to(device)

    # 5: Perform batch prediction
    with torch.no_grad():
        outputs, _ = resnet18_model(tiles_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, indices = torch.max(probabilities, 1)
    
    # 6: Convert predictions to ASCII characters and build the output string
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(indices[index].item() + 32)
            fin += char
            index += 1
        fin += '\n'

    
    # 7: Output to terminal
    print(fin)
    # 8: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")

def convert_edge_mobile(fi,save, binarize=False, inc = 64, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, ae = False, ded=False):
    start_time = time.time()
    factor = factor * inc / 10
    # 0: Import Model
    device = "cuda"
    model = MobileNetV2(num_classes=96, grayscale=True).to(device)
    model.load_state_dict(torch.load(f'artifacts/mobilenetv2_ascii_classifier_{inc}_x_{inc}.pth'))
    model.eval()

    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if not ded:
        if accelerated:
            tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
            tensor = tensor * 255.0
            tensor = tensor.byte()
            to_pil = ToPILImage()
            pil_img = to_pil(tensor).convert("L")
            # print(tensor)
            # show(tensor)
        else:
            cv2_img = DetectEdges(fi, low_thresh, high_thresh)
            pil_img = Image.fromarray(cv2_img).convert("L")
    else: 
        pil_img = ImageOps.invert(Image.fromarray(og_img).convert("L"))
    

    # 2: Resize
    height, width = og_img.shape[:2]
    print(f"Width: {width}, Height: {height}")
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(float) 
    max_value = np.max(img_array_edges)  # Get the maximum value
    if max_value > 1:  # Avoid re-normalizing if already normalized
        img_array_edges /= max_value
    if binarize:
        f_vec = np.vectorize(f4)
        img_array_edges = f_vec(img_array_edges)
    if ae:
        img_array_edges, ae_time = apply_autoencoder(img_array_edges, dimensions=inc)
        print(f"AE Conversion Time: {ae_time:.4f} seconds")
    
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character

    tiles = []
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            subs = img_array_edges[li:li + inc, i:i + inc]
            # subs = subs / 255.0  # Normalize
            tiles.append(subs)
        
    tiles = np.array(tiles)
    tiles_tensor = torch.tensor(tiles).unsqueeze(1).to(torch.float32).to(device)

    # 5: Perform batch prediction
    with torch.no_grad():
        outputs = model(tiles_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, indices = torch.max(probabilities, 1)
    
    # 6: Convert predictions to ASCII characters and build the output string
    fin = ""
    index = 0
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            char = chr(indices[index].item() + 32)
            fin += char
            index += 1
        fin += '\n'
    

    
    # 7: Output to terminal
    print(fin)
    # 8: Save to output file
    if(save):
        with open(save,'w') as f:
            f.write(fin)
    end_time_2 = time.time()
    elapsed_time = end_time_2 - start_time
    image_processing_time = end_time_1 - start_time
    model_conversion_time = end_time_2 - end_time_1
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")
    

if __name__ == "__main__":
    # Argument Parser to take in command line arguments
    parser = argparse.ArgumentParser(prog='ASCII')

    parser.add_argument('--filename', help='name of file to process',required=True,type=str)
    parser.add_argument('--algorithm','-a', help='algorithm to run',type=str)
    parser.add_argument('-i', '--increment', help='Increment to segment the image by (10x10 tiles or 64x64 tiles)',type=int,default=10)
    parser.add_argument('-f', '--factor', help='factor to scale output ASCII by',type=float,default=2.5)
    parser.add_argument('-c', '--color', help='colored output if using greyscale algorithm',action="store_true")
    parser.add_argument('-s', '--save', help='save output',type=str)
    parser.add_argument('-cf', '--convolutionalfilter', help='filter to apply edge detection (sobel or laplace)',type=str,default='sobel')
    parser.add_argument('-low','--low_thresh', help='low threshold for edge detection', type=int, default=0)
    parser.add_argument('-high','--high_thresh', help='high threshold for edge detection', type=int, default=50)
    parser.add_argument('--accelerated', help='enable accelerated processing', action='store_true', default=False)
    parser.add_argument('--binarize', help='binarize image before converting to ascii', action='store_true', default=False)
    parser.add_argument('--discludeEdge', '-ded', help='disclude', action='store_true', default=False)
    parser.add_argument('--disable_hog', help='disable HOG feature processing', action='store_true', default=False)
    parser.add_argument('--autoencoder', '-ae', help='Apply Autoencoder filtering prior to inference', action='store_true', default=False)
    
    try:
        args = parser.parse_args()
        print(vars(args))
        fi = vars(args)['filename']
        algo = vars(args)['algorithm']
        factor = vars(args)['factor']
        color = vars(args)['color']
        inc = vars(args)['increment']
        save = vars(args)['save']
        cf = vars(args)['convolutionalfilter']
        low_thresh = int(vars(args)['low_thresh'])
        high_thresh = int(vars(args)['high_thresh'])
        accelerated = vars(args)['accelerated']
        bin = vars(args)['binarize']
        disable_hog = not vars(args)['disable_hog']
        ae = vars(args)['autoencoder']
        ded = vars(args)['discludeEdge']
        if(vars(args)['algorithm']=='grey'):
            convert_grey(Image.open(fi),save,color,factor)
        elif(vars(args)['algorithm']=='edge'):
            convert_edge(Image.open(fi),cf,save,factor)
        elif(vars(args)['algorithm']=='pitur'):
            convert_edge_cv2(fi,save,factor)
        elif(vars(args)['algorithm']=='aiss'):
            convert_edge_aiss(fi,save, bin, factor, low_thresh, high_thresh, accelerated, ae, ded)
        elif(vars(args)['algorithm']=='edgev3'):
            convert_edge_v3(fi,save, bin, factor, low_thresh, high_thresh, accelerated, ae, ded)
        elif(vars(args)['algorithm']=='edgev4'):
            convert_edge_v4(fi,save, bin,factor, low_thresh, high_thresh, accelerated, ae, ded)
        elif(vars(args)['algorithm']=='knn'):
            convert_edge_knn(fi,save,bin, inc= inc, factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, ae=ae, ded=ded)
        elif(vars(args)['algorithm']=='svm'):
            convert_edge_svm(fi,save,bin, inc= inc, factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, hog_enable = disable_hog, ae=ae, ded=ded)
        elif(vars(args)['algorithm']=='rforest'):
            convert_edge_random_forest(fi,save,bin,  inc= inc, factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, hog_enable = disable_hog, ae=ae, ded=ded)
        elif(vars(args)['algorithm']=='nn'):
            convert_edge_nn(fi,save, bin, inc= inc ,factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, ae=ae,ded=ded)
        elif(vars(args)['algorithm']=='cnn'):
            convert_edge_cnn(fi,save, bin, inc= inc, factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, ae=ae,ded=ded)
        elif(vars(args)['algorithm']=='resnet'):
            convert_edge_resnet(fi,save, inc= inc ,factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, ae=ae,ded=ded)
        elif(vars(args)['algorithm']=='mobile'):
            convert_edge_mobile(fi,save, bin, inc= inc, factor=factor, low_thresh=low_thresh, high_thresh=high_thresh, accelerated=accelerated, ae=ae,ded=ded)
        else:
            convert_blackwhite(Image.open(fi),save,factor)

    except argparse.ArgumentError as e:
        print(e.message)
    print()

print(rgb(255,255,255)+"Thank you for using the ascii_terminal!")


