import cv2
import numpy as np
import os
path=os.getcwd()
img_path=path+"/rc_img/" # set the list
a=os.listdir(img_path)
ori_img=path+"/target1/" # target image
for i in a:
    i2=i[:-4]
    img=cv2.imread(ori_img+i2+".png")
    if img is None:
        print(i)
#    img2=cv2.resize(img,(1920,1280))
    cv2.imwrite(path+"//rc_mask//"+i,img) # target folder