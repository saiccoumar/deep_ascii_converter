import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def bilateral_filter(images, d, sigma_color, sigma_space):
    filtered_images = []
    for image in images:
        # Apply the bilateral filter
        filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        filtered_images.append(filtered_image)
    
    return filtered_images

def show(tensor):
    # Convert tensor to numpy array
    tensor = tensor.cpu().detach().numpy()
    
    # If the tensor is in (channels, height, width), transpose it to (height, width, channels)
    if tensor.shape[0] == 3:  # Assuming 3 channels for RGB
        tensor = np.transpose(tensor, (1, 2, 0))
    elif tensor.shape[0] == 1:  # Grayscale image
        tensor = tensor.squeeze(0)
    
    # Normalize to the range 0-255 if necessary
    epsilon = 1e-5
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min()+epsilon) * 255).astype(np.uint8)
    
    # Display the image using OpenCV
    if tensor.ndim == 2:  # Grayscale image
        cv2.imshow('Image', tensor)
    else:  # RGB image
        cv2.imshow('Image', cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Step 1: Grayscale Conversion using PyTorch operations
def rgb2gray(images,device):
    # width, height = 1000, 1000  
    # for i in range(len(images)):
    #     images[i] = cv2.resize(images[i], (width, height))    
    images = np.array(images)

    # Convert the NumPy array to a PyTorch tensor
    images = torch.tensor(images, dtype=torch.float32).to(device)

    # print("Step 1")
    images = images.permute(0, 3, 1, 2)  # Adjust permutation to (n, channels, height, width)
    gray_images = 0.299 * images[:, 0, :, :] + 0.587 * images[:, 1, :, :] + 0.114 * images[:, 2, :, :]
    return gray_images.unsqueeze(1)



def gaussian_blur(images, device, kernel_size=3, sigma=0.0):
    # print("Step 2")   
    # Create a one-dimensional Gaussian distribution
    gauss = torch.Tensor([np.exp(-(x - kernel_size//2)**2/float(2*sigma**2)) for x in range(kernel_size)]).to(device)
    
    # Normalize to ensure the sum of weights equals 1
    gauss /= gauss.sum()

    # Expand dimensions to create a 2D Gaussian kernel
    gauss_2d = torch.outer(gauss, gauss)
    # Generate Gaussian kernel
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)
    
    # Apply Gaussian blur
    blurred_images = F.conv2d(images, kernel, padding=kernel_size // 2).to(device)
    return blurred_images

def sobel_filters(images,device):
    # print("Step 3")

    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).to(device)

    G_x = F.conv2d(images, kernel_x, padding=1)
    G_y = F.conv2d(images, kernel_y, padding=1)

    convolved = torch.sqrt(G_x ** 2 + G_y ** 2)
    convolved = convolved * (255.0 / convolved.max())
    
    angles = torch.atan2(G_y, G_x) * (180.0 / torch.pi)
    angles[angles < 0] += 180
    convolved = convolved.to(torch.uint8)

    return convolved, angles

def non_maximum_suppression(gradients, angles,device):
    # print("Step 4")
    pad_gradients = F.pad(gradients, (1, 1, 1, 1), mode='constant', value=0).to(device)
    pad_angles = F.pad(angles, (1, 1, 1, 1), mode='circular').to(device)

    grad_x = pad_gradients[:, :, 1:-1, 2:] - pad_gradients[:, :, 1:-1, :-2]
    grad_y = pad_gradients[:, :, 2:, 1:-1] - pad_gradients[:, :, :-2, 1:-1]

    angle = pad_angles[:, :, 1:-1, 1:-1] % 180

    grad_x[angle >= 157.5] = 0
    grad_y[angle >= 157.5] = 0
    grad_x[angle < 22.5] = 0
    grad_y[angle < 22.5] = 0

    mask = (grad_x >= 0) & (grad_y >= 0) & (grad_x >= grad_y)
    suppressed = gradients.clone()
    suppressed[~mask] = 0

    return suppressed

def double_threshold_hysteresis(image_tensor, low_thresh, high_thresh,device):
    # print("Step 5")
    # Convert image tensor to float and normalize to range [0, 1]
    image_tensor = image_tensor.float().to(device)
    
    # Step 1: Apply high threshold
    strong_edges = (image_tensor >= high_thresh).float().to(device)
    
    # Step 2: Apply low threshold
    weak_edges = (image_tensor >= low_thresh).float() * (image_tensor < high_thresh).float().to(device)
    
    # Step 3: Perform connectivity analysis to strengthen weak edges
    kernel = torch.tensor([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]).float().view(1, 1, 3, 3).to(device)
    weak_edges_dilated = torch.nn.functional.conv2d(weak_edges, kernel, padding=1)
    weak_edges = (weak_edges_dilated > 0).float() * strong_edges
    
    return strong_edges, weak_edges

def canny_edge_detection(images, low_threshold=0, high_threshold=50, kernel_size=3, sigma=1.0, device="cpu"):
    i = 0
    # Step 1: Grayscale conversion
    
    gray_images = rgb2gray(images,device)
    # print(gray_images.shape)
    # for i in range(len(images)):
    #     show(gray_images[i])

    # Step 2: Gaussian Blur
    blurred_image = gaussian_blur(gray_images, device, kernel_size, sigma)
    # print(blurred_image.shape)
    # for i in range(len(images)):
    #     show(blurred_image[i])
    
    # Step 3: Gradient Calculation
    gradients, angles = sobel_filters(blurred_image,device)
    
    # Step 4: Non-Maximum Suppression
    non_max_suppressed = non_maximum_suppression(gradients, angles,device)
    # print(non_max_suppressed.shape)
    # for i in range(len(images)):
    #     show(non_max_suppressed[i])
  
    # Step 5: Double Thresholding and Edge Tracking by Hysteresis
    edges = double_threshold_hysteresis(non_max_suppressed, low_threshold, high_threshold,device)[0]
    # print(edges.shape)
    # for i in range(len(images)):
    #     show(edges[i])

    return edges



# image = cv2.imread('test/ascii.png')
# image2 = cv2.imread('test/kitty.webp')
# image3 = cv2.imread('test/kuromi.webp')

# width, height = 1000, 1000  
# image = cv2.resize(image, (width, height))
# image2 = cv2.resize(image2, (width, height))
# image3 = cv2.resize(image3, (width, height))

# images = np.array([image2])
# start_time = time.time()
# edges = canny_edge_detection(images)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed Conversion Time: {elapsed_time:.4f} seconds, Device: {device}")
# show(edges[0])

# edges[0]

