import os
from PIL import Image
import numpy as np


train_images = []
train_labels = []
# reading normal train images from disk
folder_address = "Datasets//Train//Train Normal"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Train//Train Normal//" + name)
    array = np.asarray(image)
    train_images.append(array)
    train_labels.append(0)

# reading pneumonia train images from disk
folder_address = "Datasets//Train//Train Pneumonia"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Train//Train Pneumonia//" + name)
    array = np.asarray(image)
    train_images.append(array)
    train_labels.append(1)

test_images = []
test_labels = []
# reading normal test images from disk
folder_address = "Datasets//Test//Test Normal"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Test//Test Normal//" + name)
    array = np.asarray(image)
    test_images.append(array)
    test_labels.append(0)

# reading pneumonia test images from disk
folder_address = "Datasets//Test//Test Pneumonia"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Test//Test Pneumonia//" + name)
    array = np.asarray(image)
    test_images.append(array)
    test_labels.append(1)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)







