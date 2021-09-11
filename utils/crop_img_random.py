import cv2

import os
import numpy as np

import random

def test_this_image(filenamee):


    path=os.getcwd()+"//img//" # base path
    test_path=os.getcwd()+"//img2/" # test image path

    data_list=os.listdir(path)

    import cv2
    img=cv2.imread(path+filenamee)
    #img=cv2.resize(img,(3072,))
    (h, w) = img.shape[:2]
    if h>=w:
        img=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    img_orig = np.copy(img)
    if w>3000:
        for i in range(4):
            img2=cv2.resize(img_orig,None,fx=0.6,fy=0.6)
            #print(img.shape)
            (h, w) = img2.shape[:2]
            h1=512
            w1=1024
            ax=h-h1-1
            bx=w-w1-1
            a=random.sample(range(0,ax),1)
            b=random.sample(range(0,bx),1)
            print(a,b)
            y=0+a[0]
            x=0+b[0]
            crop_img = img2[y:y+h1, x:x+w1]
            cv2.imwrite(test_path+filenamee[:-4]+"_%s.png"%i,crop_img)
    else:
        print(w)
        cv2.imwrite(test_path+filenamee[:-4]+".png",img)
if __name__ == '__main__':
    path=os.getcwd()

    test_path=path+"//img//"
    tl=os.listdir(test_path)
    for i in tl:
        test_this_image(i)
