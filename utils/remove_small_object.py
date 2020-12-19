import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt
from skimage import morphology
import os
path=os.getcwd()
data_path = path + '/mask/'
store_path=path+'/segmentation/'
data_dir_list = os.listdir(data_path)
img_data_list=[]
ak=len(data_dir_list)
i=0
name=[]
kernel = np.ones((3,3),np.uint8)
for dataset in data_dir_list:
    img=cv2.imread(data_path+dataset,0)
    img=img.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1
# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 10  

#your answer image
    img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    name_img=dataset[:-4]
    cv2.imwrite(store_path+name_img+'.png',img2)
