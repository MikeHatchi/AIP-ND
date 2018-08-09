#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/udacity_image_classifier_project
#
#
# PURPOSE: Image Classifier Python app. application should be a pair of
#          Python scripts that run from the command line.
#          > Train a network on a data set with train.py
#          Prints out training loss, validation loss, and validation
#          accuracy as the network trains
#          Use GPU for training: python train.py data_dir --gpu
#
#          > Predict flower name from an image with predict.py along with
#          the probability of that name. A single image will be passed
#          /path/to/image and return the flower name and class probability.
#

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

# Label mapping
import json

# Replace classifier by new untrained feed-forward network hyperparameters
from collections import OrderedDict

# Import a function that helps to create a classifier that adapt itself
# depending on the number of # hidden_units commanded

from loopclassifier import layer

# import validation() function
from validation import validation

# Creates parse and Define CLA
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers',
                    help='Path to datasets')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--arch', type=str, help='Model of your choice: vgg11,\
                    vgg13 or Densenet121')
parser.add_argument('--input_size', type=int, help='Number of inputs')
parser.add_argument('--hidden_size', nargs="*", type=int,
                    default=[4096, 1000, 256], help='Number of hidden units')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, help='learning_rate')
parser.add_argument('--path_flo', type=str,
                    help='flowers/test/28/image_05242.jpg')

args = parser.parse_args()


# Data sets paths
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
                   transforms.RandomRotation(90),
                   transforms.RandomResizedCrop(224),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([
                   transforms.Resize(256),
                   transforms.CenterCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                          shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# loading of mapping doc (flower class with picture)
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# interacting with user
print("You are about to use an Image Classifier Python app. \
      based on a neural network.",
      "You can command --arch --input_size --hidden_size --lr --epochs --gpu\
      --path_flo.", sep="\n")


# Step 1: Loading of the model
def load_yo_model(arch='vgg11'):

    # Loading of the VGG11 pretrainded model
    if args.arch == 'vgg11':
        model = models.vgg11(pretrained=True)
    # Loading of the VGG13 pretrainded model
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    # Loading of the Densenet-121 pretrainded model
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError('Perdu, try again among the 3 given models: vgg11, \
                         vgg13, densenet121.')

    print("Your chosen {} model has been correctly downloaded, \
          thanks".format(args.arch))
    return model


def main():

    # Step 2: Building of neural network

    # Freeze of parameters
    model = load_yo_model()

    for param in model.parameters():
        param.requires_grad = False

    # User can change torch.models, input, hidden units, lr, epochs and gpu.
    def hyperpara(input_size=25088, hidden_size=[4096, 1000, 256],
                  lr=0.0013, epochs=6, gpu=False):

        if args.input_size:
            input_size = args.input_size

        if args.hidden_size:
            hidden_size = args.hidden_size

        if args.lr:
            lr = args.lr

        if args.epochs:
            epochs = args.epochs

        if args.gpu:
            gpu = args.gpu
        if gpu and torch.cuda.is_available():
            print("GPU is on.")
            device = torch.device("cuda:0")
            model.cuda()
        else:
            print("GPU is off, CPU on.")
            device = torch.device("cpu")

        return input_size, hidden_size, lr, epochs, gpu

    # Reminder to the  user what he/she has commanded
    input_size, hidden_size, lr, epochs, gpu = hyperpara()
    print("""This is what you've commanded:
          arch: {0}
          input_size: {1}
          hidden_size: {2}
          lr: {3}
          epochs: {4}
          gpu: {5}.""".format(args.arch, args.input_size, args.hidden_size,
                              args.lr, args.epochs, args.gpu))

    output_size = len(cat_to_name)

    # Using the layer function that built a dict classifier based on the
    # hidden units commanded

    classifier = layer(input_size, hidden_size, output_size)
    model.classifier = classifier

    # Criterion and Optimizer for backpropagation
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    # Step 3: training and eval on valid_data set

    epochs
    print_every = 20
    steps = 0
    running_loss = 0

    # Change to cuda
    model.to('cuda')

    for e in range(epochs):
        model.train()
        for ii, (image, label) in enumerate(trainloader):
            steps += 1

            image = Variable(image.cuda())
            label = Variable(label.cuda())

            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.to('cuda')
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader,
                                                     criterion)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training Loss: {:.3f}..".format(
                                                running_loss/print_every),
                      "Validation Loss: {:.3f}..".format(
                                                  test_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}..".format(
                                                    accuracy/len(validloader)))

                running_loss = 0
                model.train()

    # Step 4: Test of network's accuracy on Test data set

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            model.eval()
            image = Variable(image.cuda())
            label = Variable(label.cuda())
            model.to('cuda')
            optimizer.zero_grad()
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print('Accuracy of the network on test_data set: \
          %d %%' % (100 * correct / total))

    # Step 5: Saving the checkpoint file

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': input_size,
                  'output_size': len(cat_to_name),
                  'hidden_size': hidden_size,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')

    print("checkpoint saved!")

    # Call to main function to run the program


if __name__ == "__main__":
    main()
