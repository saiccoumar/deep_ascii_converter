from legacy_dynamic import *

if __name__ == "__main__":
    # Argument Parser to take in command line arguments
    parser = argparse.ArgumentParser(prog='ASCII')

    parser.add_argument('--media', help='type of media to use',required=True,type=str)
    parser.add_argument('--algorithm','-a', help='algorithm to run',type=str)
    parser.add_argument('-c', '--color', help='colored output if using greyscale algorithm', type=str)
    parser.add_argument('-f', '--factor', help='factor to scale output ASCII by',type=float,default=2.5)
    parser.add_argument('-s', '--save', help='save output',type=str)
    parser.add_argument('-cf', '--convolutionalfilter', help='filter to apply edge detection (sobel or laplace)',type=str,default='sobel')
    try:
        args = parser.parse_args()
        print(vars(args))
        media = vars(args)['media']
        algo = vars(args)['algorithm']
        factor = vars(args)['factor']
        color = vars(args)['color']
        save = vars(args)['save']
        cf = vars(args)['convolutionalfilter']
        fin = ""
        if (media=='cam'):
            print('cam')
            vid = cv2.VideoCapture(0)
            while(True):
                ret, frame = vid.read()
                try:
                    image = Image.fromarray(frame).convert("RGB")
                except:
                    continue
                cv2.imshow('Frame',frame)
                text = ""
                if(vars(args)['algorithm']=='grey'):
                    text = convert_grey(image,color,factor)
                elif(vars(args)['algorithm']=='edge'):
                    text = convert_edge(image,cf,factor)
                elif(vars(args)['algorithm']=='edge-bnw'):
                    text = convert_edge_bnw(image,cf,factor)
                else:
                    text = convert_blackwhite(image,factor)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print(text)
                fin+=text
            vid.release()
            cv2.destroyAllWindows()
            # print(rgb('white')+"quit")

        else:
            print('video')
            vid = cv2.VideoCapture(media)
            

            if (vid.isOpened()== False): 
                print("Error opening video stream or file, check file path")
            while(vid.isOpened()):
                ret, frame = vid.read()
                try:
                    image = Image.fromarray(frame).convert("RGB")
                except:
                    break
                cv2.imshow('Frame',frame)
                text = ""
                if(vars(args)['algorithm']=='grey'):
                    text = convert_grey(image,color,factor)
                elif(vars(args)['algorithm']=='edge'):
                    text = convert_edge(image,cf,factor)
                elif(vars(args)['algorithm']=='edge-bnw'):
                    text = convert_edge_bnw(image,cf,factor)
                else:
                    text = convert_blackwhite(image,factor)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                print(text)
                fin+=text
            vid.release()
            cv2.destroyAllWindows()
        if(save):
            with open(save,'w') as f:
                f.write(fin)

    except argparse.ArgumentError as e:
        print(e.message)
    print()

print(rgb_string('white')+"Thank you for using the ascii_terminal!")


