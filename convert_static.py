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


def convert_edge_v3(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
    start_time = time.time()
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
   
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


def convert_edge_v4(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
    start_time = time.time()
    factor = factor * 6.4
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
  
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

def convert_edge_svm(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable = True):
    start_time = time.time()

    # 0: Import SVM Model
    if hog_enable:
        loaded_clf = joblib.load('artifacts/svm_model_hog.pkl')
    else:
        loaded_clf = joblib.load('artifacts/svm_model.pkl')
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
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
    
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

def convert_edge_random_forest(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable=True):
    start_time = time.time()

    # 0: Import SVM Model
    if hog_enable:
        loaded_clf = joblib.load('artifacts/random_forest_model_hog.pkl')
    else:
        loaded_clf = joblib.load('artifacts/random_forest_model.pkl')
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
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
    
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

def convert_edge_nn(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True, hog_enable=True):
    start_time = time.time()
    device = "cuda"
    # 0: Import SVM Model
    model = NeuralNetwork(225, 128, 64, 32, 96)
    model.load_state_dict(torch.load('artifacts/nn_ascii_classifier.pth'))
    model.to(device)
    model.eval()

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
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
   
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
    
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
            
        
       
            subs_hog = extract_hog_features(subs[0])[0]
            tiles.append(subs_hog)
       
    
                
        #     fin = fin + chr(y_pred+32)
        # fin = fin + '\n'
    print(np.array(tiles).shape)
    
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
    print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
    print(f"Image Processing Time: {image_processing_time:.4f} seconds")
    print(f"Model Conversion Time: {model_conversion_time:.4f} seconds")




def convert_edge_cnn(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
    start_time = time.time()
    # 0: Import Model
    device = "cuda"
    model = CNN(input_shape=(1, 10, 10), num_classes=96).to(device)
    model.load_state_dict(torch.load('artifacts/cnn_ascii_classifier.pth'))
    model.eval()
    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width  = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final) 
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
    tiles = []
    for li in range(0, math.floor(len(img_array_edges) / inc) * inc, inc):
        for i in range(0, math.floor(len(img_array_edges[li]) / inc) * inc, inc):
            subs = img_array_edges[li:li + inc, i:i + inc]
            # subs = subs / 255  # Normalize
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

def convert_edge_resnet(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
    start_time = time.time()
    # 0: Import Model
    device = "cuda"
    resnet18_model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=96, grayscale=True).to(device)
    resnet18_model.load_state_dict(torch.load('artifacts/resnet18_ascii_classifier.pth'))
    resnet18_model.eval()

    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final) 
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
    print('preprocessing done')

    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
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

def convert_edge_mobile(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
    start_time = time.time()
    # 0: Import Model
    device = "cuda"
    model = MobileNetV2(num_classes=96, grayscale=True).to(device)
    model.load_state_dict(torch.load('artifacts/mobilenetv2_ascii_classifier.pth'))
    model.eval()

    # 1: detect edges using Canny Edge Detection
    og_img = cv2.imread(fi)
    if og_img is None:
        print("Image doesn't exist in path.")
        return
    if accelerated:
        tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
        tensor = tensor * 255.0
        tensor = tensor.byte()
        to_pil = ToPILImage()
        pil_img = to_pil(tensor).convert("L")
        # print(tensor)
        # show(tensor)
    else:
        cv2_img = DetectEdges(fi)
        pil_img = Image.fromarray(cv2_img).convert("L")
    

    # 2: Resize
    height, width = og_img.shape[:2]
    print(f"Width: {width}, Height: {height}")
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = pil_img
    final = pil_img.resize((width,height))

    
    # 3: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final) 
    f_vec = np.vectorize(f4)
    img_array_edges = f_vec(img_array_edges)
    fin = ""
    end_time_1 = time.time()
    # 4: Take 10x10 tiles and replace with ASCII character
    inc = 10
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
    parser.add_argument('-f', '--factor', help='factor to scale output ASCII by',type=float,default=2.5)
    parser.add_argument('-c', '--color', help='colored output if using greyscale algorithm',action="store_true")
    parser.add_argument('-s', '--save', help='save output',type=str)
    parser.add_argument('-cf', '--convolutionalfilter', help='filter to apply edge detection (sobel or laplace)',type=str,default='sobel')
    parser.add_argument('-low','--low_thresh', help='low threshold for edge detection', type=int, default=0)
    parser.add_argument('-high','--high_thresh', help='high threshold for edge detection', type=int, default=50)
    parser.add_argument('--accelerated', help='enable accelerated processing', action='store_true', default=False)
    parser.add_argument('--disable_hog', help='eisable HOG feature processing', action='store_true', default=False)
    try:
        args = parser.parse_args()
        print(vars(args))
        fi = vars(args)['filename']
        algo = vars(args)['algorithm']
        factor = vars(args)['factor']
        color = vars(args)['color']
        save = vars(args)['save']
        cf = vars(args)['convolutionalfilter']
        low_thresh = int(vars(args)['low_thresh'])
        high_thresh = int(vars(args)['high_thresh'])
        accelerated = vars(args)['accelerated']
        disable_hog = not vars(args)['disable_hog']
    
        if(vars(args)['algorithm']=='grey'):
            convert_grey(Image.open(fi),save,color,factor)
        elif(vars(args)['algorithm']=='edge'):
            convert_edge(Image.open(fi),cf,save,factor)
        elif(vars(args)['algorithm']=='pitur'):
            convert_edge_cv2(fi,save,factor)
        elif(vars(args)['algorithm']=='edgev3'):
            convert_edge_v3(fi,save,factor, low_thresh, high_thresh, accelerated)
        elif(vars(args)['algorithm']=='edgev4'):
            convert_edge_v4(fi,save,factor, low_thresh, high_thresh, accelerated)
        elif(vars(args)['algorithm']=='svm'):
            convert_edge_svm(fi,save,factor, low_thresh, high_thresh, accelerated, disable_hog)
        elif(vars(args)['algorithm']=='rforest'):
            convert_edge_random_forest(fi,save,factor, low_thresh, high_thresh, accelerated, disable_hog)
        elif(vars(args)['algorithm']=='nn'):
            convert_edge_nn(fi,save,factor, low_thresh, high_thresh, accelerated)
        elif(vars(args)['algorithm']=='cnn'):
            convert_edge_cnn(fi,save,factor, low_thresh, high_thresh, accelerated)
        elif(vars(args)['algorithm']=='resnet'):
            convert_edge_resnet(fi,save,factor, low_thresh, high_thresh, accelerated)
        elif(vars(args)['algorithm']=='mobile'):
            convert_edge_mobile(fi,save,factor, low_thresh, high_thresh, accelerated)
        else:
            convert_blackwhite(Image.open(fi),save,factor)

    except argparse.ArgumentError as e:
        print(e.message)
    print()

print(rgb(255,255,255)+"Thank you for using the ascii_terminal!")


# def convert_edge_v3(fi,save, factor=2.5, low_thresh=0, high_thresh=50, accelerated=True):
#     start_time = time.time()
#     # 1: detect edges using Canny Edge Detection
#     og_img = cv2.imread(fi)
#     if og_img is None:
#         print("Image doesn't exist in path.")
#         return
    
#     if accelerated:
#         tensor = canny_edge_detection([og_img],low_threshold=low_thresh, high_threshold=high_thresh)[0]
#         tensor = tensor * 255.0
#         tensor = tensor.byte()
#         to_pil = ToPILImage()
#         pil_img = to_pil(tensor).convert("L")
#         # print(tensor)
#         # show(tensor)
#     else:
#         cv2_img = DetectEdges(fi)
#         pil_img = Image.fromarray(cv2_img).convert("L")
    

#     # 2: Resize
#     width, height = og_img.shape[:2]
#     scale = width / (factor*500)
#     width, height = width / scale, height / (scale*2)
#     width, height = int(width), int(height)

#     final = pil_img
#     final = pil_img.resize((width,height))
    
#     whole = True
    
#     # 3: Convert to 0s and 1s for bnw
#     img_array_edges = np.array(final).astype(int)
    
#     # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
#     if whole:
#         f_vec = np.vectorize(f4)
#     else:
#         f_vec = np.vectorize(f3)    
#     img_array_edges = f_vec(img_array_edges)
#     print(img_array_edges)
#     image = np.array(img_array_edges, dtype=np.uint8)
#     cv2.imshow('Image', image*255)
#     cv2.imshow('Image2', og_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     fin = ""

#     # SAVING TRAIN DATA
#     data = []
#     labels = []

    
#     if whole:
#         e = edge_vectors3
#     else:
#         e = edge_vectors
#     # 4: Take 10x10 tiles and replace with ASCII character
#     inc = 10
#     for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
#     # Odd bounds have data loss on last col/row
#         for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
#             # Subsetting the tile itself
#             subs = []
#             subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
#             subs = np.array(subs)
#             # checking the minimum distance between each character vector and tile vector
#             min = 0
#             min_dist = np.linalg.norm(e[0]-subs.flatten())
#             # if not (np.all(subs == 0)):
#             for j in range(0,len(e)):
#                 new_mind_dist = np.linalg.norm(e[j]-subs.flatten())

#                 if (new_mind_dist<min_dist):
#                     min = j
#                     min_dist = new_mind_dist
            
#             # print(subs.shape)
#             # print(edge_vectors3.shape)
#             data.append(subs[0])
#             labels.append(min)

#             if whole:
#                 fin = fin + chr(min+32)
#             else: 
#                 fin = fin + edge_conversions[min]
#         fin = fin + '\n'
    
#     data = np.array(data)
#     # print(data.shape)
#     labels = np.array(labels)
#     data = data / 255.0
#     data = data.reshape((len(data), 10, 10, 1))  # (num_samples, height, width, channels)
#     flattened_images = data.reshape(data.shape[0], -1)
#     reshaped_data = np.concatenate((labels.reshape(-1, 1).astype(np.int8), flattened_images.astype(np.float64)), axis=1)

#     df = pd.DataFrame(reshaped_data)
    
#     # 5: Output to terminal
#     print(fin)
#     # 6: Save to output file
#     if(save):
#         print("Saved!")
#         df = df.reset_index(drop=True)
#         df.to_csv('archive/ascii_character_classification_sanrio_data_v3.csv', mode='a', header=None, index=False)
#         fi2 = os.path.splitext(fi)[0].replace('images', 'text') + ".txt"
#         with open(fi2,'w') as f:
#             f.write(fin)

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds")
