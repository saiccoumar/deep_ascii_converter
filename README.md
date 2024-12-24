# ASCII Image Converter: Converting images to ASCII!

<p align="center">
<img width="100%" height="auto" src="https://github.com/user-attachments/assets/07a1ed55-7141-4997-9708-b069ccdc5cb7">
	<em>https://commons.wikimedia.org/wiki/File:SVG_ASCII_logo.svg</em>
</p>


by Sai Coumar
<br />
Sections: <br />
[Introduction](#introduction)<br />
[Usage](#usage)<br />
[The Algorithms Behind ASCII Conversion](#the-algorithms-behind-ascii-conversion)<br />
[Potential Improvements](#potential-improvements)<br />
[Citations](#citations-and-resources)<br />
[Conclusion](#conclusion)<br />
# Introduction
Welcome to my Deep ASCII Image Converter! This was a ~~quick littl~~ big project I made following my last iteration to utilize machine learning and deep learning for structure-based ASCII art synthesis. I'll cover the installation and usage, as well as a high level overview of some of the algorithms available.
# Usage
First make sure all the dependencies are met.
```bash
pip install -r requirements.txt
```
This software uses numpy, opencv, and Pillow, PyTorch and scikitlearn and is written entirely in native python. 
I was also using Python v3.12.7 for this project. 
<br />
## Image Conversion
In order to convert an image to ASCII use convert_static.py. convert_static.py requires a --filename argument with the path of the input file. <br /><br />
You can specify --algorithm or -a to specify the ASCII conversion algorithm. The options are 'edgev3', 'edgev4', 'knn', 'svm', 'rforest', 'nn', 'cnn', 'resnet', 'mobile', 'aiss','grey', 'edge', 'pitur', and 'bnw'. If no algorithm is specified it will use bnw by default. <br /><br />
You can use the -f or --factor arguments followed by an argument between 0 and 6 to specify the size of the output in the terminal. The default factor is 2.5. You'll likely need to resize your window anyway to see the entire image.<br /><br />
If -c or --color is specified while using the greyscale algorithm then the ascii text will be outputted in color.<br /><br />
If you use -s or --save followed by a file path you can save the output of the conversion. <br /><br />
If you use -cf followed by a convolutional filter (either sobel or laplace) while using the edge detection algorithm then you can pick which edge detection filter to use. 
<br />
ex.
``` bash
python convert_static.py --filename [INPUT FILENAME] --algorithm [ALGORITHM] --factor [size] -c --save [OUTPUT FILENAME] -cf [CONVOLUTIONAL FILTER]
```

⚠️ UNDER CONSTRUCTION ⚠️

## Video Conversion
Video conversion is a legacy feature but is still functional. See instructions in the previous iteration's repository.

# The Algorithms Behind ASCII Conversion

⚠️ UNDER CONSTRUCTION ⚠️

# Citations and Resources:
Below are resources that I consulted during this project <br />
1. [Stefan Pitur's Edge detection](https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py)
2. [JavE image2ascii](http://www.jave.de/image2ascii/algorithms.html)
3. [ANSI Codes Wiki](https://en.wikipedia.org/wiki/ANSI_escape_code)

# Conclusion:



⚠️ UNDER CONSTRUCTION ⚠️

