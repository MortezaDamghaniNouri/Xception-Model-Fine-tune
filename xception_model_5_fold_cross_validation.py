"""
This script trains a model for pneumonia detection, and it uses fine-tuning method. It uses the Xception model
as the base model. 5-fold cross validation is used for evaluation, and after the evaluation the accuracy of
each fold and the mean accuracy will be printed.
"""

import os
from PIL import Image
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import copy


train_images = []
train_labels = []
print("Loading images from disk...")
# loading normal train images from disk
folder_address = "Datasets//Train//Train Normal"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Train//Train Normal//" + name)
    array = np.asarray(image)
    train_images.append(array)
    train_labels.append(0)

# loading pneumonia train images from disk
folder_address = "Datasets//Train//Train Pneumonia"
for name in os.listdir(folder_address):
    image = Image.open("Datasets//Train//Train Pneumonia//" + name)
    array = np.asarray(image)
    train_images.append(array)
    train_labels.append(1)

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

train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

all_images = []
for i in train_images:
    all_images.append(i)
for i in test_images:
    all_images.append(i)
all_labels = []
for i in train_labels:
    all_labels.append(i)
for i in test_labels:
    all_labels.append(i)

# k-fold cross-validation is implemented here
fold_number = 5
i = 1
accuracy_list = []
while i <= fold_number:
    print("Fold: " + str(i))
    test_images = []
    test_labels = []
    train_images = []
    train_labels = []
    all_images_copy = copy.deepcopy(all_images)
    all_labels_copy = copy.deepcopy(all_labels)
    j = int((i - 1) * (len(all_images) / fold_number))
    while j < (i * (len(all_images) / fold_number)):
        test_images.append(all_images_copy[j])
        test_labels.append(all_labels_copy[j])
        j += 1
    j = int((i - 1) * (len(all_images) / fold_number))
    k = 1
    while k <= (len(all_images) / fold_number):
        all_images_copy.pop(j)
        all_labels_copy.pop(j)
        k += 1
    train_images = np.array(all_images_copy)
    train_labels = np.array(all_labels_copy)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    # loading the Xception model
    base_model = keras.applications.Xception(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
    # freezing the weights of the base model
    base_model.trainable = False
    # creating a new model on top
    inputs = keras.Input(shape=(256, 256, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    # training the model
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    model.summary()
    history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
    fig, axis = plt.subplots(2, 1)
    axis[0].plot(history.history["val_accuracy"])
    axis[0].set_title("Validation Data Accuracy (transfer learning)")
    axis[1].plot(history.history["val_loss"])
    axis[1].set_title("Validation Data Loss (transfer learning)")
    plt.show()
    print("Unfreezing the base model and starting fine tuning...")
    # unfreezing the base model
    base_model.trainable = True
    # fine tuning the model
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    history = model.fit(train_images, train_labels, epochs=2, batch_size=32, validation_data=(test_images, test_labels))
    fig, axis = plt.subplots(2, 1)
    axis[0].plot(history.history["val_accuracy"])
    axis[0].set_title("Validation Data Accuracy (fine tuning)")
    axis[1].plot(history.history["val_loss"])
    axis[1].set_title("Validation Data Loss (fine tuning)")
    plt.show()
    # evaluating the model
    print("Evaluating the model...")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    accuracy_list.append(test_acc)
    print("Accuracy: " + str(round(test_acc * 100, 2)) + " %")
    # saving the trained model
    print("Saving the model...")
    model.save("model " + str(i) + ".h5")
    print("Model saved as model " + str(i) + ".h5")
    i += 1

# printing the results
i = 0
while i < len(accuracy_list):
    print("fold " + str(i + 1) + " accuracy: " + str(round(accuracy_list[i] * 100, 2)) + " %")
    i += 1
summation = 0
for i in accuracy_list:
    summation += round(i * 100, 2)
print("Mean accuracy: " + str(round(summation / len(accuracy_list), 2)) + " %")









