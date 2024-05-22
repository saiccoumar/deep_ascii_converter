# ASCII Image Converter: Converting images to ASCII!

<p align="center">
<img width="100%" height="auto" src="https://github.com/saiccoumar/ascii_converter/assets/55699636/2be26524-80f9-4348-89e9-bcdf9b032321">
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
Welcome to my ASCII Image Converter! This was a quick little project I made to have some practice with image processing as well as implementing algorithms in python since I'm rusty. I'll cover the usage, how the algorithms I implemented work, as well as some potential improvements that can be made in the future. 
# Usage
First make sure all the dependencies are met.
```bash
pip install -r requirements.txt
```
This software uses mostly numpy, opencv, and Pillow, but I added matplotlib to the requirements.txt because I was using it for testing. 
I was also using Python v3.7.6 for this project. 
<br />
## Image Conversion
In order to convert an image to ASCII use convert_static.py. convert_static.py requires a --filename argument with the path of the input file. <br /><br />
You can specify --algorithm or -a to specify the ASCII conversion algorithm. The options are 'grey', 'edge', 'pitur', and 'bnw'. If no algorithm is specified it will use bnw by default. <br /><br />
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
In order to convert a video or camera input to ASCII use convert_dynamic.py. convert_dynamic.py requires a --media argument with the path of the input file; if the input is specified to cam then camera input will be used instead. <br /><br />
You can specify --algorithm or -a to specify the ASCII conversion algorithm. The options are 'grey', 'edge','edge-bnw, and 'bnw'. If no algorithm is specified it will use bnw by default.<br /><br />
You can use the -f or --factor arguments followed by an argument between 0 and 6 to specify the size of the output in the terminal. You'll likely need to resize your window anyway to see the entire image. The default factor is 2.5.<br /><br />
If -c or --color is specified while using the greyscale algorithm then the ascii text will be outputted in color. With convert_dynamic.py you need to specify the color of the text. You can either use 'static' to use the original colors or black, red, green, yellow, blue, magenta, cyan, or white. <br /><br />
If you use -s or --save followed by a file path you can save the output of the conversion. <br /><br />
If you use -cf followed by a convolutional filter (either sobel or laplace) while using the edge detection algorithm then you can pick which edge detection filter to use. <br />
ex.
``` bash
python convert_dynamic.py --media [INPUT] --algorithm [ALGORITHM] --factor [size] --color [COLOR] --save [OUTPUT FILENAME] -cf [CONVOLUTIONAL FILTER]
```

# The Algorithms Behind ASCII Conversion
For ASCII conversion I found 3 common algorithms. 
- Black/White algorithms
- Greyscale-mapping algorithms
- Edge Detection algorithms 
## Black/White Algorithms:
Black/White algorithms were by far the easiest to implement and had the best results for conversion. The intuition is that pixels that are dark are replaced with a character whereas pixels that are light are replaced with an empty character to create negative space. The contrast from negative space and filled in characters eventually creates shapes which you can perceive as ASCII art. While you can replace each pixel individually, you can significantly improve the resolution of your output by taking 2x2 tiles of pixels and matching the pixel density (by pattern) to an ascii character. For example if a 2x2 tile has 4 white pixels then we'd replace it with an empty character. If there is one pixel that is black we'd replace it with a '.'. If all 4 pixels are black we replace it with an @ symbol. 

<p align="center">
	<img width="75%" height="auto" src="https://github.com/saiccoumar/ascii_converter/assets/55699636/7d86b06f-1463-4102-acd5-e6add6a6ce29">
</p>
This tree represents all 16 possible combinations of 2x2 pixel tiles. Red represents a value of false, and blue represents a value of true. Each level is the next value in the tuple. <br />

As you can see from this tree, the pixel combinations with more pixels filled in typically have larger ASCII characters while the ones with fewer have characters with either smaller or no characters. 

My implementation:
1. Create a dictionary of tuples matching 2x2 pixel combinations (as tuples) to ASCII characters. 
2. Resize the input image using Pillow functions
3. Convert the Pillow image to a black and white image of Pillow type 1 image and then into a numpy array
4. Iterate through 2x2 pixel tiles in the image and find the matching ASCII character from the dictionary. Append the character to the output string
5. Return the output string

## Greyscale Algorithms:
Greyscale algorithms are a little more complicated and performed marginally slower than the Black/White algorithms. This algorithm works by taking a greyscale image where pixels range from 0-255 (representing pixel intensity) and maps that value onto a grey ramp. A grey ramp is a string of characters slowly increasing from least dense to more dense, which can be used to create more or less shading. The mapping works by first normalizing the pixel intensity to a range of 0-1 by dividing the pixel density by 255. Then we multiply that normalized pixel density by the length of the grey ramp to get the index of the character that matches the shade of the pixel within the greyramp.

```
 ....________,:;\'`^"l!i><~+_-?][}{1)*#(|/tfjrxnuvczmwqpdbkhaoIXYUJCLQ0OZMW&8%B@$
```
After playing around this is the grey ramp I settled on. Repeating characters in a grey ramp isn't problematic, and I found that I needed to add more low density characters was necessary to create more negative space.<br />
My implementation:
1. Initialize the chosen grey ramp
2. Resize the image
3. Convert the image to a greyscale type L Pillow image using Pillow functions
4. Define the mapping function to map a pixel intensity to the grey ramp
5. Iterate through every pixel in the image and replace the character with a matching character <br />

Within this function I also chose to implement the ability to color the pixels. I did this using ANSI codes and a helper function to convert the RGB values of a pixel to an ANSI code. ANSI escape codes are a special escape code that can be used to change the color and background of text by adding them right before the text and (most) terminals are capable of rendering text using ANSI codes to print in color. Using ANSI codes, an inbuilt system feature, allowed me to avoid using more unnecessary dependencies. My default text editor couldn't render ANSI codes so to view the ASCII art in color you can use the cat command in bash.

## Edge Detection Algorithms:
The final (and most challenging) algorithm was edge detection for ASCII art. While greyscale and black/white algorithms make art by shading with ASCII characters, edge detection aims to detect lines of shapes and then replace those shapes with lines to make line art. This algorithm was by and far way more difficult. <br />

There are two parts of this algorithm: detect edges and then the line replacement.
<br /> <br />
Edge detection alone is a complicated endeavor. The process of edge detection begins applying a gaussian blur to smooth the image. This provides much more defined lines later. After that we pass the image through a convolutional filter-either a sobel filter or a laplacian filter - which outputs an image with white edges and black negative space. After this we apply non-max suppression to thin the edges, double thresholding to sort the pixels into strong, weak, and irrelevant, and then hysteresis for the final product. <br />
To replace lines...I had to get creative. There are a couple ways to approach this, but all have a common denominator in that, similar to black/white algorithms, we take a tile of pixels from our image with edges and replace them with a matching character. The most naive implementation is essentially to apply black/white algorithms on the outputted image with edges. A particularly difficult but effective method is to classify the tile using a Convolutional Neural Network (CNN) with a character, but that was more work than I was interested in and extremely computationally intensive.
<br /> <br />
My approach worked by taking images of my chosen ASCII characters and then to turn them into vectors. Then, for every tile, I converted the tile into a vector and used the euclidean distance formula to find the character vector that was most similar to the tile vector. I would then add the character matching that vector to my string. 

My implementation:
1. Enhance image for better output using gaussian blur
2. Convert the image to a greyscale type L Pillow image using Pillow functions
3. Apply sobel of laplace convolutional filters to the image
4. Resize the Image
5. Convert the Image to 0s and 1s for a black and white image
6. Take 10x10 tiles and replace with ASCII character using Euclidean Distance Formulas

You may notice that I missed some steps of edge detection in my implementation. Truthfully, the most important step is to apply the sobel filtering, and since I was using Pillow, applying non-max suppression, hysteresis, and double thresholding was difficult as well as EXTREMELY computationally intensive. I found a really good repository from user Stefan Pitur which uses openCV and does all the steps of edge detection (and quite well) but while testing and comparing the time to compute edges using my implementation vs Pitur's, his appeared to be about 20 times slower. My implementation was satisfactory on images with relatively hard edges and was much more viable for real time conversion. This is likely due to how Pillow is implemented as a module optimized for image processing; I think it is possible to make Pitur's implementation more optimized for the best performance as well but I wasn't interested in doing that.  
<br /><br />
For video conversions, I also implemented the naive black/white conversion of the edge detection because it could render output much faster as well. This was as simple as gaussian blurring, convoluting with sobel/laplace filters, and then using the black/white conversion algorithm on the convoluted image. This had the best edge detection performance for videos (I recommend trying it with some anime clips).
# Potential Improvements:
There are without a doubt numerous ways to optimize this. If this was done in a compiled language like C++ or Java this would probably be faster. For the black/white image, higher resolutions could be achieved using 3x3 tiles with more combinations of pixels and matching characters and for the greyscale algorithm a more fine tuned greyramp could create a better image. The image processing functions, especially for edge detection, would also be better if implemented as modules written in low level assembly or with vectorization. With edge detection, I had to pick between a tradeoff of speed and accuracy and I'm sure that with time it could be optimized to do both. Also, other algorithms for line replacement exist that might be much better-a CNN could be used to classify edges as characters instead of using euclidean distance. Alternatively a generative AI might be able to skip edge detection altogether and create an ASCII image directly. A GPU with CUDA cores (or anything better than my mid macbook) might also increase performance as well, but more testing is necessary for that.
# Citations and Resources:
When researching for this project I found a lot of great resources that I definitely need to credit <br />
1. [Stefan Pitur's Edge detection](https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py): This github repository does a much better job explaining and implementing edge detection. His algorithm is much more effective and the code I used in canny_edge_detection_pitur.py belongs to him. 
2. [JavE image2ascii](http://www.jave.de/image2ascii/algorithms.html): This blog was the basis for my understanding the algorithms used to make ascii art. There are a lot of websites with the same information but this one was by far the simplest and most digestible despite it's old age, although I had to modify the implementation details to get the algorithms to work.
3. [ANSI Codes Wiki](https://en.wikipedia.org/wiki/ANSI_escape_code): This wiki was super insightful on the history and features of ANSI codes. I learned about ANSI codes exclusively for this project and reading up on the history was pretty cool.
4. [AsciiArtist](https://github.com/JuliaPoo/AsciiArtist): This repository is a project that uses a convolutional neural network for ASCII conversion, which I briefly mentioned as a possibility in the previous section. I didn't use this algorithm but it's a nice alternative approach that I didn't cover which I think is worth checking out.
# Conclusion:
This project was a fun little exercise in algorithms and image processing, and I really enjoyed working with modules like openCV and PIL. ASCII art algorithms can be difficult to digest but I tried to show a diverse range of implementations and features used to make ASCII art. If anyone has suggestions or corrections feel free to email me about it at sai.c.coumar1@gmail.com, and I hope others might've found this project useful as well! 

<p align="center">
<img width="100%" height="auto" src="https://github.com/saiccoumar/ascii_converter/assets/55699636/3617c295-d8b7-456e-96b5-5fb4f023a274">
	
</p>


<p align="center">
<img width="100%" height="auto" src="https://github.com/saiccoumar/ascii_converter/assets/55699636/bb6fd06a-9aa3-48b5-ad08-c92bf1373b9a">
	
</p>
                                                                                                                             





                                                              .- .        
                                                                                                                ,.['.['      
                                                                                                               ,['.['...     
                                                                                                              ' ,['.:/ -     
                                                                                                            - .,- ,.- '      
                                                                                                         - ,---; .,          
      --' ,.                                                                                          .'- .[-,/-, .          
     ---,:['.,                                                                                    - ,.-'.['.['.[ .-'         
    - .[''.[- ,                                                                                .'- .['.['.[,,['..-           
      .,[,,,.- .                                                                           - ' ./-['.['/';-,|.[, .           
       .-' ,--- ,,..-                                                                   . ,-''/-'..:/.['/.[;';-,- .          
           ,-'.[[----' ,...                                                          .' .,[,//-,['./-;''./..:[-, .           
          ' .''..[,/-;'...-''---                                                  -' ./'.[,/./.[,[''.['['././.[ ,            
           .-'/.[,/-[-['./.['..- ,'- .                                           ,--['.['.[-'./.[,[['.[,['./-'.' ,           
            .--'.['./-.['./.[''.[,:...,,' .                                     '- [-['.:,[,['./.[,.:,['.['.[[' ,.'          
           - -''/.[''./.['./.[['.['.[''/.,.,,' .           . - .- .- .- .- .-  ., '.[-['.|.[,['./.[,/;-,[,['..:,-            
            ' '/-'.:[,./.['./.,[;-['.[[-'/-,[,.-    - . ,' .' ................, .,.' [-['/'.[,['./.['.,[,/.['..'             
            -' '.[''.[,/'.['./;'..,['.['/-,[,.. ..,' ..['/'.[['./-['/'''.['./-,:.../-   ,'/'.[,['./.[;'.[''.:[' '            
           ' -'''.:['.['/'.[''.[';'.:/.['.[,/.., ./'/-'.['/'..:/-[-['//['.['.,[-['./.,:,' .,'.[,/--'.,[;-[,['..,.-           
              -';-;'/'.['/'.:[;-,,['./'.['.[-, .,/-['.:/.['/'./-,.[-[-'.[;'''/.[-['./'.::[- ,- ['/'/'/.[-,/.[' .             
             '  ['.['/'.['/./'..:|.['.['.:,[-[, ./.,['./'.['/-'.[;-[\[J]\[;'[''.[-['.:/-'.,[[- .-''.[''.['./.:, '            
              ,' ['.['/'.[-''/.['/'.['.:['/'.\-,.\'/.['.[;-[''/][0@I]I]]]I0@J::/.[-['./.[;'..,[- .,''/;'.['./- ,.            
               ,'.['.['/'/.:[-'.['/'.:['.['/'..['.[''.['/-,.J00J00J@]I][0@]JJ0@\'.[-['./'.[[;'.[[- ,''.['.['.,-              
              ' ,'.:,[,/'../'/'/.['/./.['.['/-,,['.[,[,['.[;/00J@IJJ0@]II]]]]I]]:/.[-['.:/...:,[-,[,..'.[''- ,-'             
               ' './;-['/.['/-[-'.:./-'.[;''.,[|.['.[['.['.000@I;I]]I]]0@],['I]]J'.[--[-'./'./;-,/.[- .- '- .,               
               ' ,'.[-.[-'.['.['/./.,,:/'.[[;'.['.:,[-,[,['00@'./-I]I][0JJ[,[,JJ:['.,[-,['.[''.['./.[,.-'                    
                ,.,'.['.[['/''.[-'./'[-;''./-['.['.|.[;-,[,II::/-[@]I]00J0J/.JJ:/.[;'.[;-['/'['.['./.['. '                   
                 -' .-' ,.['/['.[['.|.[-[,['..:[,['/'.[-,['.|0]]]I]]I]IIJ0J@I[,.\'.,,,[-,.['/.[;-['./.['.,'                  
                      -'  . . ,'..['/'.[-,[,/./.[,['/'.[;-,[['/;;|[0@'0:/0J,/-['.,''-[- ''-[-;'.[-['./.[- ,-                 
                     '     , /-,/'.['/'.,[,/./-'.[,['/'.,.,[-['..[[|[,|/'|/''..,-\J@@@@@JJ.  ,/-,\-['./'.,'                  
                          , ,/.['/'.:/';'.[-'.[['.[,, ,-\\]-- .'' .,,[['/'.,,-\@@@@@@@@@@@@@J\ .[-,.['.:// ' ,               
                          ,''./.['/./-|.[;'.['..[' -]]@@@@@@@@@@]\\\[ .' ,-]]@@@@@@@@@@@@@@@0;@\  ['.['.\' ,.                
                         ' ,['./.[-'.[''.[''.:[' \@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@0| ;0;@@\ ,/'/--,,.                 
                          , ['.\'.[['.[['.[,[' -J0I@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@0| .' ,-@@@@J '/-,,[, ,                
                         ,' ,['.['..[;-,['.:,-;@@@\''II@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:-' -'-''@@@@@:-'.[[,-'                 
                         -' [,['.[['/-.[,['.'-@@@J','   ,|I@@@@@@@@@@@@@@@@@@@@@@@@@@J -' - - I@@@@@ -'..[ - ,               
                          -' [,['..:/./.[,,-;@@@@@@ .,'--  @@@@@@@@@@@@@@@@@@@@@@@@@@@/ -' - .I@@@@@: /'/ .-'                
                           .'-,,['./-'./.[, ]@@@@@J- .,  ' @@@@@@@@@@@@@@@@@@@@@@@@@@@@\  ' --@@@@@@:-''.- .                 
                           -' [|.[,/.['././ @@@@@@@ - .,,' @@@@@@@@@@@@@|||0@@@@@@@@@@@@@@]]@@@@@@@@: ;' -                   
                          ' -' ['.[-;-['./ .I@@@@@@@   .'-I@@@@@@@@@@@0-]IJ.;@@@@@@@@@@@@@@@@@@@@@@@ '- '                    
                             -' :/'.[-.['., ;@@@@@@@@@]\@@@@@@@@@@@@@@@.',,\]@@@@@@@@@@@@@@@@@@@@@@,., '                     
                              .,-'/'.['.['/' I@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@0I@@@@@@@@@@@@@@@@@@@| - ,                      
                               - .'/'.['.:/'. I@@@@@@@@@@@@@@@@@@@@@@J'II@@I|\@@@@@@@@@@@@@@@@@@0,.,-  ,                     
                                 ., ,/.[,\'/-, |@@@@@@@@@@@@@@@@@@@@@@@JJ\\J@@@@@@@@@@@@@@@@@0|- .-'                         
                                   '   ,'.['.[,. |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@I| ., .                            
                                    ' ,' . ,'.['., ||@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@00|'   ,                                
                                       .  ,' - .''- .-ftree'|I@@@@@@@@@@@@@@@@@@@000||,,,  ---' ,  ,                              
                                  ..     , ['./--'- . -    ,,,,,,,,,,' .-     ----,.[,[-,/.,'  .'----                        
                               -'-.\' ,'-,:,['.[,[,['/.[,['..\-.[--./.[--;'/;;-,/-' ,' ,' ''''.-;/00\'                       
                              - /JI[J' ,,' ,,,' ,,'-['.['.:/'.,/-,['-'.,[-['..['.,-;@@@]\' [' .  0J00 ,'                     
                               -'00J[ ,., ,\JJJJJ]\  ['.[,\'/-['.,,-- '''..['/.['' ]@@@@@@ .    ,  . '                       
                                  -     .,@@@@@@@@@J, [;-[-['.,' \]@@@@\\- ,,./.:,;. I@@@@: .    .                           
                                 .     . ]@@@@@@@@@@J,.,..[ ,--J@@@@@@@@@@@JJ.. ---|. @@@@ .                                 
                                        ,;@@@@@@@||@@0  ,' \]@@@@@@@@@@@@@@@@@,/]I[J ,'@0,- '                                
                                    .  .' I@@@@@@J ;,\I[J.|@@@@@@@@@@@@@@@@@@@:'JI[0,- @J]@:'                                
                          .,'-' - .   - ]].]@@@@@J'- JII/J @@@@@@@@@@@@@@@@@@@@\.,[-]: ]@@@,.,                               
                          -'.../-- ,. . I@@@@@@@@, ]\'|,|-]@@@@@@@@@@@@@@@@@@@@@@@@@@: ;0|,-                                 
                         ' -['./..-    , |@@@@@I, J@@@J@]@@@@@@@@@@@@@@@@@@@@@@@@@@@@., --'                                  
                         -'--['./- '    ,.   .--\@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '-                                     
                          - ,.['.,[ ,'    -' -'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@0 ,                                      
                         - ., .,//'/'.' ,    . '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ,                                       
                               . '''/-,[-,'.,.,.@@@@@@@@@@@@@@@@@0|I@@@@@@@@@@@@@@@J.' ,                                     
                                ' ,. ,'.,','- . @@@@@@@@@@@@@@@@0 . I@@@@@@@@@@@@@@0 ,-                                      
                                    ,' -'-'- . ./@@@@@@@@@@@@@@0 . '-@@@@@@@@@@@@@@@@].-                                     
                                             \@@@@@@@@@@@@@@@@@:, '--@@@@@@@@@@@@@@@@@@:'-                                   
                                           ,;@@@@@@@@@@@@@@@@@@\ , .'@@@@@@@@@@@@@@@@@@@ .                                   
                                          ' I@@@@@@@@@@@@@@@@@@:-  - |I@@@@@@@@@@@@@@@@,-                                    
                                           , I@@@@@@@@@@@@@@@|,-  ,  -  |||00@@@I000|, .                                     
                                            ,  ''||I|0I|||,,-'         , ..-      . . ,                                      
                                             ' ,--' - . ,.'    '               -                                             
                                                        .                                                                    
                                                                                                                             

                                                                                                                             
                                                                 \\   \ /           \ \\/////                                
                                                              /\    \/    \/   \\            /                               
                                                             \  /\      \/  \      \       /  \                              
                            \///// /                        \  \          \\  /\            /  /                             
                         \            \ //      /\\\          \            \/  /             \ \                             
                        \  /\     \\///                 ///  \        \////     /            /  /                            
                       /   /            \ //  \\          /         /  \//          \/ \\       \/                           
                          \                                 |         \  \  \     \/      /\////  \/                         
                                                         \          / \/\  /        \\ \            /                        
                                                          \ \        \/\  \          |  \/        /                          
                                                          /  \\        \  /          \  /  /      \                          
                       /  /                                \\   ///\\      /        \  \   \     /  /                        
                       \     \                               \/    //\\  \\ \//  \\       \     //                           
                        /    \                                             \//     /  \        \                             
                        /   /                                                    /  /        \     \                         
                        /  /                                                      \/   //\\    \    /                        
                       \  \                                                         \//   / \     /  \                       
                      /                                                                                                      
                         \                                                                        \  \/\///////              
                                                                                              \       //\  /\/\              
                                                                                             \//\  |                         
                                                                                \   /              |                         
                         /   \           ////                                  \                                             
               \              \         \    /                                       /         \          \//                
             \/\//\      \                                                          \                \   ///                 
                       /  /             /    \              \\    /                              \  /                        
                       \   / \//         \                 \ \     \\                           /   /                        
                     /\/      \                            / /   \ /                        \/      //                       
                   |   /\/   /                              \////\                       /\\      \     \//                  
                          \/  \/  \\/                                                   \            //\/ \                  
                            /     /\/                                                  \  \    //////                        
                          /\        /                                                /\/  /           /  \/                  
                        \   \\  \/    \///                                      \\/       /            \  \                  
                       / \\        \//\     \ /////\                / / \/\\        \\ \/  /                                 
                       \           \\                                        \   \\      /  \\          /  /                 
                               /\      \  /   \    /  \////////////\//    \    \\  /      \\  /        \  \                  
                             \    \\     \  /       \ \\               / /       /  \       /  \\    //  \                   
                          /\    \       \  \         /  //          \\  \         /  \       \  \/ \\  //                    
                         \   \\  \/    \  \           \/   \\////    /\            \  \       \\     /\                      
                       //  \   //  \/\\  \                 /////\\/                 \  \     \\    \                         
                      /   /      \/     /                                            /  \\\    \\                            
                      /  /          /                                                 /    /\                                
                     /  |           \  \                                              \                                      
                     \  \          /  /                                                /  /                                  
                      /  \\        \                                                      \                                  
                       \\  \//  \ \                                                     /  /                                 
                         \/          |                                                  |  /                                 
                             \       /                                                  \  |                                 
                                                                                           \                                 
                                  /  //                       \\/                   /\ /\  /                                 
                                  /                                                        /                                 
                                     \  \      ///////////////  ///////////\            \  /                                 
                                  \                          |  |                       /  |                                 
                                     \                       |  |                       \  \                                 
                                   /  /                      |  |                      /  /                                  
                                    /  \\                    \   \                    \   /                                  
                                     /\  \//               \      //               \    \                                    
                                        /                    \/  /                  / \                                      
                                              //////////\            \////////\                                              
                                                                                                                             

