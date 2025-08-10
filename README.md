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

# Introduction
Welcome to my Deep ASCII Image Converter! This was a ~~quick littl~~ big project I made following my last iteration to utilize machine learning and deep learning for structure-based ASCII art synthesis. I'll cover the installation and usage. To read more about the algorithms, please read my paper here: [Evaluating Machine Learning Approaches for ASCII Art Generation](https://arxiv.org/abs/2503.14375v1).


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

## Video Conversion
Video conversion is a legacy feature but is still functional. See instructions in the previous iteration's repository.

# The Algorithms Behind Deep ASCII Conversion
Fundamentally, all of these ASCII methods work very similarly. They take an input image, extract the edges, divide the image into nxn "tiles", and then use some advanced character matching method to correspond that tile to the closest text character that matches. It then assembles the output as a string and returns it to the terminal. In general, it appears that classical machine learning algorithms, such as random forest and SVM, tend to perform much better than deep neural networks. This is likely attributed to a loss of data as information propogates through deep neural networks in a domain that uses low dimensionality input. 

<img width="1205" height="394" alt="image" src="https://github.com/user-attachments/assets/1f95692f-73ae-4dca-a24b-fe4f15454a55" />

# Citation:
If you'd like to cite this work please use
```
@misc{coumar2025evaluatingmachinelearningapproaches,
      title={Evaluating Machine Learning Approaches for ASCII Art Generation}, 
      author={Sai Coumar and Zachary Kingston},
      year={2025},
      eprint={2503.14375},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2503.14375}, 
}
```

Gallery:
<img width="548" height="704" alt="inverted_rforest_no_hog" src="https://github.com/user-attachments/assets/761d1398-6c3e-4f57-9a0b-8c7940601e66" />
<img width="1078" height="1016" alt="inverted_rforest_no_hog" src="https://github.com/user-attachments/assets/fcbbd40c-c4d0-47af-99fb-492f9c4f7e58" />
<img width="1360" height="583" alt="inverted_rforest" src="https://github.com/user-attachments/assets/b311875f-b275-4c0a-b4d1-49a8015667a2" />
<img width="999" height="648" alt="inverted_rforest_no_hog" src="https://github.com/user-attachments/assets/490cc04f-0996-48ab-b2ab-83049a3df782" />
<img width="661" height="772" alt="inverted_rforest" src="https://github.com/user-attachments/assets/f43e2d2f-4232-44d0-82c1-51305eb9b19f" />
<img width="1046" height="586" alt="inverted_rforest_no_hog" src="https://github.com/user-attachments/assets/3b17bf72-85b9-4900-93eb-2347c60c097f" />
<img width="1095" height="503" alt="inverted_rforest_no_hog" src="https://github.com/user-attachments/assets/d84ef10a-f26c-4e05-aa31-42bceeb96387" />



