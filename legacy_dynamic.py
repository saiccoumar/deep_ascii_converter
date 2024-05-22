# http://www.jave.de/image2ascii/algorithms.html
# https://aa-project.sourceforge.net/aalib/
# https://github.com/JEphron/TEXTFLIX/blob/master/main.py
# https://github.com/StefanPitur/Edge-detection---Canny-detector

import argparse
import numpy as np
import math
import os
from PIL import Image, ImageEnhance, ImageFilter
from canny_edge_detection_pitur import *
import cv2

 
# Windows requirement
os.system("color")

# Pixel threshold function to convert greyscale to bnw
# for edge detection
def f3(x):
    if x<6.5:
        return 0
    return 1

  
# edge_conversions = [',','<','>','/','.','^','"',' ','V','|','`','_','\\']

# Loading the edge vectors saved from the file and the conversions to their respective character
edge_conversions = ['`','|',' ','/','.','\\',' ']
# allImg()
edge_vectors = np.loadtxt('edge_vectors.txt')


# Convert an image to ASCII using the black/white algorithm
def convert_blackwhite(img, factor=2.5):
# 1: Tuples that map all 16 possible combinations of 2x2 pixels to an ASCII character
    black_white_chars = {
    # Format of [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
        (False, False, False, False) : ' ', #0 
        (False, True, False, False) : '.', #1
        (True, False, False, False) : ',', #1
        (False, False, True, False) : '\'', #1
        (False, False, False, True) : '-', #1
        (True, True, False, False) : ':', #2 
        (False, False, True, True) : ';', #2 
        (True, False, True, False) : '|', #2
        (False, True, False, True) : '\\', #2
        (False, True, True, False) : '/', #2
        (True, False, False, True) : '[', #2
        (False, True, True, True) : ']', #3
        (True, True, False, True) : 'J', #3
        (True, False, True, True): 'I', #3 
        (True, True, True, False) : '0', #3
        (True, True, True, True) : '@' #4
    }
# 2: Resize image.
    width, height = img.size
    scale = width / (factor*100)
    width, height = width / scale, height / (scale*2)
    
    img = img.resize((int(width), int(height)), Image.ANTIALIAS)
# 3: Turn image into array of numbers and convert image to greyscale    
    img_array_grey = np.array(img.convert("1")) #Uses black/white 
# 4: Iterate through every 4 pixels and convert it into an ascii character and append ascii character to string
    fin = ""
    for li in range(0,math.floor(len(img_array_grey) / 2) * 2,2): #Floor function to use only even bounds. 
        #Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_grey[li]) / 2.) * 2,2):
            tu = (img_array_grey[li, i],img_array_grey[li+1, i],img_array_grey[li, i+1],img_array_grey[li+1, i+1])
            fin = fin + black_white_chars[tu]
        fin = fin + '\n'
# 5: Output to terminal
    return fin 

    


def convert_grey(img, color='white', factor=2.5):
# 1: initialize grey_ramp that chooses characters to replace
    grey_ramp = ' ....________,:;\'`^"l!i><~+_-?][}{1)*#(|/tfjrxnuvczmwqpdbkhaoIXYUJCLQ0OZMW&8%B@$'
 # Alternate grey_ramp
    # grey_ramp = '@MV%#:;*+=-:...___   '
    # grey_ramp = grey_ramp[::-1]

# 1: Resize the image
    width, height = img.size
    scale = width / (factor*50)
    width, height = width / scale, height / (scale*2)
    img = img.resize((int(width), int(height)), Image.ANTIALIAS)
# 3: Turn image into array of numbers and convert image to greyscale    
    img_array = np.array(img.convert("RGB"))
    img_array_grey = np.array(img.convert("L"))
# 4: Mapping greyscaled value to the grey_ramp
    def map(x):
        return (math.floor(x * len(grey_ramp) / 255) - 1)
# 5: Iterate through every value and print append corresponding character to string
    fin = ""
    for li in range(0,len(img_array_grey)):
        for i in range(0,len(img_array_grey[li])):
            # Conditional to either print in color or not 
            if(color):
                if (color=='static'):
                    fin = fin + rgb(img_array[li][i][0],img_array[li][i][1],img_array[li][i][2]) + grey_ramp[map(img_array_grey[li][i])]
                else:
                    fin = fin + rgb_string(color) + grey_ramp[map(img_array_grey[li][i])]
            else:
                fin = fin + grey_ramp[map(img_array_grey[li][i])]
            
        fin = fin + '\n'
# 6: Output to terminal 
    return fin
# 7: Save to output file
    

# Helper function to convert rgb values to an ANSI code 
def rgb_string(color):
    # Predefined ANSI color palette
    colors = {
        'black': 30,     # Black
        'red': 31,   # Red
        'green': 32,   # Green
        'yellow': 33, # Yellow
        'blue': 34,   # Blue
        'magenta': 35, # Magenta
        'cyan': 36, # Cyan
        'white': 37 # White
    }

    return f"\033[{colors[color]}m"
def rgb(r,g,b):
    # Predefined ANSI color palette
    rgb = (r,g,b)
    colors = {
        (0, 0, 0): 30,     # Black
        (255, 0, 0): 31,   # Red
        (0, 255, 0): 32,   # Green
        (255, 255, 0): 33, # Yellow
        (0, 0, 255): 34,   # Blue
        (255, 0, 255): 35, # Magenta
        (0, 255, 255): 36, # Cyan
        (255, 255, 255): 37 # White
    }

    closest_color = min(colors, key=lambda c: sum(abs(x - y) for x, y in zip(c, rgb)))
    return f"\033[{colors[closest_color]}m"

def convert_edge(img,cf, factor=2.5):
    # 1: Enhance image for better output using gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius = 5))

    # 2: Convert to Black and White
    img = img.convert("L")
    

    # 3: Apply filtering to get edges
    #Sobel filter is the most common, has the most negative space
    sobel = (1,0,-1,2,0,-2,1,0,-1)
    #The laplacian filter has less negative space 
    laplacian = (-1, -1, -1, -1, 8,-1, -1, -1, -1)
    #In-built PIL filter has the least negative space and same performance to laplacian

    find_edge = ImageFilter.FIND_EDGES
    if (cf=='laplace'):
        img_grey_edge = img.filter(ImageFilter.Kernel((3, 3), laplacian, 1, 0))
        # img_grey_edge = img.filter(find_edge)
    else:
        img_grey_edge = img.filter(ImageFilter.Kernel((3, 3), sobel, 1, 0))


    # 4: Resize
    width, height = img.size
    scale = width / (factor*500)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = img_grey_edge
    final = ImageEnhance.Brightness(final)
    final = img_grey_edge.resize((width,height))

    
    # 5: Convert to 0s and 1s for bnw
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f3)
    img_array_edges = f_vec(img_array_edges)


    
    fin = ""

    # 6: Take 10x10 tiles and replace with ASCII character
    inc = 10
    for li in range(0,math.floor(len(img_array_edges) / inc) * inc,inc): #Floor function to use only even bounds. 
    #     #Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_edges[li]) / inc) * inc,inc):
            # Subsetting the tile itself
            subs = []
            subs.append([xi[i:i+inc] for xi in img_array_edges[li:li+inc]])
            subs = np.array(subs)
            # checking the minimum distance between each character vector and tile vector
            min = 0
            min_dist = np.linalg.norm(edge_vectors[0]-subs.flatten())
            for j in range(0,len(edge_vectors)):
                new_mind_dist = np.linalg.norm(edge_vectors[j]-subs.flatten())

                if (new_mind_dist<min_dist):
                    min = j
                    min_dist = new_mind_dist
            # replace tile with new character in output string
            fin = fin + edge_conversions[min]
        fin = fin + '\n'
    

    
# 7: Output to terminal
    return fin

def convert_edge_bnw(img,cf, factor=2.5):
    # print(img)
    # 1: Enhance image for better output using gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius = 5))

    # 2: Convert to Black and White
    img = img.convert("L")
    

    # 3: Apply filtering to get edges
    #Sobel filter is the most common, has the most negative space
    sobel = (1,0,-1,2,0,-2,1,0,-1)
    #The laplacian filter has less negative space 
    laplacian = (-1, -1, -1, -1, 8,-1, -1, -1, -1)
    #In-built PIL filter has the least negative space and same performance to laplacian

    find_edge = ImageFilter.FIND_EDGES
    if (cf=='laplace'):
        img_grey_edge = img.filter(ImageFilter.Kernel((3, 3), laplacian, 1, 0))
        # img_grey_edge = img.filter(find_edge)
    else:
        img_grey_edge = img.filter(ImageFilter.Kernel((3, 3), sobel, 1, 0))
    # img_grey_edge.show()

    # 4: Resize
    width, height = img.size
    scale = width / (factor*100)
    width, height = width / scale, height / (scale*2)
    width, height = int(width), int(height)

    final = img_grey_edge
    final = img_grey_edge.resize((width,height))

 
   # 5: Tuples that map all 16 possible combinations of 2x2 pixels to an ASCII character
    # return convert_blackwhite(final, factor)
    black_white_chars = {
    # Format of [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]
        (False, False, False, False) : ' ', #0 
        (False, True, False, False) : '.', #1
        (True, False, False, False) : ',', #1
        (False, False, True, False) : '\'', #1
        (False, False, False, True) : '-', #1
        (True, True, False, False) : ':', #2 
        (False, False, True, True) : ';', #2 
        (True, False, True, False) : '|', #2
        (False, True, False, True) : '\\', #2
        (False, True, True, False) : '/', #2
        (True, False, False, True) : '[', #2
        (False, True, True, True) : ']', #3
        (True, True, False, True) : 'J', #3
        (True, False, True, True): 'I', #3 
        (True, True, True, False) : '0', #3
        (True, True, True, True) : '@' #4
    }

# 6: Iterate through every 4 pixels and convert it into an ascii character and append ascii character to string
    img_array_edges = np.array(final).astype(int)
    # Converting greyscale to black and white using threshold. greyscale>=50 = 1 (black). greyscale < 50 = 0 (white)
    f_vec = np.vectorize(f3)
    img_array_grey = f_vec(img_array_edges)

    fin = ""
    for li in range(0,math.floor(len(img_array_grey) / 2) * 2,2): #Floor function to use only even bounds. 
        #Odd bounds have data loss on last col/row
        for i in range(0,math.floor(len(img_array_grey[li]) / 2.) * 2,2):
            tu = (img_array_grey[li, i],img_array_grey[li+1, i],img_array_grey[li, i+1],img_array_grey[li+1, i+1])
            fin = fin + black_white_chars[tu]
        fin = fin + '\n'
# 7: Output to terminal
    return fin 
    
