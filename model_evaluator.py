"""
This script gets the address of a trained model for pneumonia detection and evaluates the input model using 500 normal images and
500 pneumonia images. At the end,  it prints the accuracy of the input model.
"""

from keras.models import load_model
import os
from PIL import Image
import numpy as np


# loading test data and the labels
test_images = []
test_labels = []
# loading normal test images from disk
folder_address = "Datasets//Test//Test Normal"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Test//Test Normal//" + name)
    array = np.asarray(image)
    test_images.append(array)
    test_labels.append(0)

# loading pneumonia test images from disk
folder_address = "Datasets//Test//Test Pneumonia"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Test//Test Pneumonia//" + name)
    array = np.asarray(image)
    test_images.append(array)
    test_labels.append(1)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
model_address = input("Enter the full address of the model which you want to evaluate (example: models//model.h5): ")
model = load_model(model_address)
# evaluating the model
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Accuracy: " + str(round(test_acc * 100, 2)) + " %")








