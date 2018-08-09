#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/image_classifier_part2
#
# # PURPOSE: Image Classifier Python app. application should be a pair of
#          Python scripts that run from the command line.
#          > Train a network on a data set with train.py
#          Prints out training loss, validation loss, and validation
#          accuracy as the network trains
#          Use GPU for training: python train.py data_dir --gpu
#
#          > Predict flower name from an image with predict.py along with
#          the probability of that name. A single image will be passed
#          /path/to/image and return the flower name and class probability.
##

# Imports python module
import argparse

# Imports

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from PIL import Image


# Creates parse and Define CLA
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Path to datasets')
parser.add_argument('--path_flo',type=str, help='flowers/test/28/image_05242.jpg')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--arch', type=str, help='Model of your choice: vgg11, vgg13 or Densenet121')

args = parser.parse_args()

# Data sets paths
data_dir='flowers'
train_dir=data_dir + '/train'
valid_dir=data_dir + '/valid'
test_dir=data_dir + '/test'

# Transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(90),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Label mapping
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def main():
    # Importing the model that the user choosed --arch vgg11 or 13 or densenet121
    from traingit import load_yo_model

    # Loading of the checkpoint and rebuilds the model

    def load_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = load_yo_model()
        model.classifier = checkpoint['classifier']
        model.load_state_dict = checkpoint['state_dict']
        #optimizer.load_state_dict(checkpoint['optimizer'])

        return model

    model = load_checkpoint('checkpoint.pth')
    print("Checkpoint loaded!")

    # Processing the image
    def process_image(img_path, max_size=400, shape=None):

        # load image and convert PIL images to tensors
        image = Image.open(img_path)

        # no large images & transformation image as requested
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape

        in_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

        # instead of / 255
        image = in_transform(image).float()

        # move to an array with float between 0-1
        image = np.array(image)

        image = image.transpose((1,2,0))
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image)

        return  image

    ## Class Prediction

    model.class_to_idx = train_data.class_to_idx
    def predict(image_path, model, topk=3, gpu=False):

#        if args.path_flo:
#            image_path = args.path_flo

        if args.gpu:
            gpu = args.gpu
        if gpu and torch.cuda.is_available():
            print("GPU is on.")
            device = torch.device("cuda:0")
            model.cuda()
        else:
            print("GPU is off, CPU on.")
            device = torch.device("cpu")
            model.cpu()

        # Prediction of the class from an image file

        global image
        image = process_image(image_path)
        image.unsqueeze_(0)
        image.requires_grad_(False)
        model = model

        image = Variable(image.cuda())

        model.eval()
        model.to('cuda')
        output = model.forward(image)
        ps = torch.exp(output)

        probs, indices = ps.topk(topk)

        indices = indices.cpu().numpy()[0]
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in indices]
        names = [cat_to_name[str(j)] for j in classes]

        return probs, classes, names

    # sanity checking with 5 flower classes/names
    if args.path_flo:
            image_path = args.path_flo
    probs, classes, names = predict(image_path, model)
    print('picture path:', image_path)
    print('probability of classification:', probs)
    print('classes number associated:', classes)
    print('names of flower associated:', names)


# Call to main function to run the program

if __name__== "__main__":
    main()
