import os


from torch import nn
#from sdd_m2 import Xception
from mobilenetv3 import MobileNetV3
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import torchvision.datasets as datasets
from effnet import EfficientNet
from sdd_func2 import Xception
path=os.getcwd()
dataset_path = path+"/data/total4"
model_weight_save_path = path+"/model/"
from simple_cls import SuperLightMobileNet
from adamp import AdamP
batch_size = 4
num_workers = 4
lr = 0.01

#total_epoch = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





# Data loading code
traindir = os.path.join(dataset_path, 'train')
testdir = os.path.join(dataset_path, 'val')

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((1000,2200)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #normalize,
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=False)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize((1000,2200)),
        transforms.ToTensor(),
        #normalize,
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)) 
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
num_classes = 2
feature_extract = True
model_name = "resnet"
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 1700

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = (1000,2200)

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

#model = MobileNetV3(num_classes).to(device)
#model=SuperLightMobileNet(num_classes)
#model=EfficientNet(1,1,num_classes)
#model = Xception(num_classes).to(device)
#self.criterion = nn.CrossEntropyLoss().to(self.device)
CEloss = nn.CrossEntropyLoss()
optimizer = AdamP(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=1e-2)
#optimizer = torch.optim.Adam(, lr=lr,weight_decay=)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,min_lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,100], gamma=0.1)
model=nn.DataParallel(model)
model.to(device)

import numpy as np
total_epoch = 50
total_iteration_per_epoch = int(np.ceil(len(train_dataset)/batch_size))

for epoch in range(1, total_epoch + 1):
    model.train()
    for itereation, (input, target) in enumerate(train_loader):
        images = input.to(device)
        labels = target.to(device)

        # Forward pass
        outputs = model(images)
        loss = CEloss(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
    scheduler.step()
    #if epoch % 2 == 0:
    torch.save(model.state_dict(), model_weight_save_path + 'model_' + str(epoch) + ".pt")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for input, target in test_loader:
            images = input.to(device)
            labels = target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)
            correct += (predicted == labels).sum().item()

        print('Epoch [{}/{}], Test Accuracy of the model on the {} test images: {} %'.format(epoch, total_epoch, total, 100 * correct / total))