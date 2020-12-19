import cv2
import numpy as np
import os

path=os.getcwd()
folder_name="/Bridge_Images/"
data_path=os.listdir(path+folder_name)# path + target folder name my case is Bridge_Images
for i in data_path:
    i2=i[:-4]
    seg=cv2.imread(path+"/br3/"+i2+".png") # I use png but you can use png or jpg
    #seg=cv2.bitwise_not(seg)
    if seg is not None:
        b,g,r=  cv2.split(seg)
        g[g==255]=0
        b[b==255]=0
        seg = cv2.merge((b, g, r))
        
        img=cv2.imread(path+folder_name + i)
        if img is not None: #exception handling
            img=cv2.addWeighted(img, 0.7, seg, 0.3, 0)
        
            cv2.imwrite(path+"/merge/"+i2+".png",img) # target folder
        else:
            print("img",i2) #check error
    else:
        print("seg",i2) # check error