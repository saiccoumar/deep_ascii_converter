from convert_static import *
import time


testfile = "test/ascii.png"

img = Image.open(testfile)
fa = 2.5
co=True
convfilt = 'sobel'

start = time.time()
convert_blackwhite(img,save=None,factor=fa)
end = time.time()
bnw_time = str(end-start)

start = time.time()
convert_grey(img,save=None,color=co,factor=fa)
end = time.time()
greyscale_time = str(end-start)

start = time.time()
convert_edge(img,cf=convfilt,save=None,factor=fa)
end = time.time()
edge_detection_time = str(end-start)



print("Black and White Execution Time:" + bnw_time)
print("Greyscale Execution Time:" + greyscale_time)
print("Edge Detection Execution Time:" + edge_detection_time)


start = time.time()
convert_edge_cv2(testfile,save=None,factor=4)
end = time.time()
edge_detection_time_pitur = str(end-start)
print("Edge Detection Execution Time (Pitur):" + edge_detection_time_pitur)

