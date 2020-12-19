import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel 
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
import torch._utils
import torch.nn.functional as F
from torchvision.transforms import Normalize


#from mobilenet_csp2 import MobileNetV3
from sdd_func2 import Xception
def test_this_image(filenamee):


    path=os.getcwd()+"//test//" # base path
    test_path=path+"//Bridge_Images/" # test image path

    data_list=os.listdir(test_path)

    model_path=os.getcwd()+"/models/"
    import cv2
    img=cv2.imread(test_path+filenamee)
    x=4608 # image width
    y=3456 # image height
    divx=3
    divy=3
    j=1
    k=1
    model_list=os.listdir(model_path)

    counter1 = 0
    counter2 = 1
    imgl = []
    for f in range(0,divx*divy):
        counter1 = str(counter1)
        full_name = (filenamee+counter1)
        imgl.append(full_name)
        counter1 = counter2
        counter2+=1
    for i in range(0,divx*divy):
        imgl[i]=img[int(k*(y/divy)-(y/divy)):int(k*(y/divy)), int(j*(x/divx)-(x/divx)):int(j*(x/divx))]
        j=j+1
        if j==divx+1:
            j=1
            k=k+1
    j=0
    for i in imgl:
        j=j+1
        # img = cv2.rectangle(i, (2, 2), (int(x/divx), int(y/divy)), (255,0,0), 5)
        cv2.imwrite(path+"/br/"+ 'img %s.jpg' %j , i)
    tlist=[]
    rlist=[]
    nlist=[]
    j=0
    i=0
    for i in range(1,divx*divy+1):
        image=cv2.imread(path+"/br/"+"./img %s.jpg" %i,0)
        # put your deep learing code in this part calcualte your inference
  
        ret3,th3 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        sx=(int(y/divy),int(x/divx))
        th2=np.zeros(sx)
        frame=15# remove the frame effect
        th2[int(0+frame):int((y/divy)-frame), int(0+frame):int((x/divx-frame))]=th3[int(0+frame):int((y/divy)-frame), int(0+frame):int((x/divx-frame))]
        imgl[i-1]=th2   
        cv2.imwrite(path+"/br2/"+"%s.jpg" %i,imgl[i-1])
    imageh=[]
    i=0
    for i in range(0,divy):
        imagehl=np.hstack(imgl[i*divx:(divx)+(i*divx)])
        imageh.insert(i,imagehl)

    image=np.vstack(imageh)
    cv2.imwrite(path+"/br3/" + filenamee[:-4]+".png",image)
if __name__ == '__main__':
    path=os.getcwd()

    test_path=path+"/test/"+"/Bridge_Images/"
    tl=os.listdir(test_path)
    for i in tl:
        test_this_image(i)
