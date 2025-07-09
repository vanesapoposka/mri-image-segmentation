import os

from tensorflow.python.layers.core import dropout
from tensorflow.python.ops.metrics_impl import false_negatives

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from random import shuffle

from keras.src.layers import Dropout, Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate
from numpy.f2py.crackfortran import verbose
from tensorflow.python.layers.pooling import AveragePooling2D
from tensorflow.python.profiler.profiler_client import monitor
import pandas as pd # to be able to read the content of the csv file
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
import cv2 as cv
from skimage import io

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import plotly.express as px
import random
import glob
from sklearn.preprocessing import StandardScaler, normalize
from IPython.display import display

import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import InceptionV3
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPool2D
from keras.losses import binary_crossentropy

image_locations = []
# to recursively search all the subdirectories in the kaggle_3m folder and map each image to its folder
for dir in glob.glob("./kaggle_3m/*"):
    if os.path.isdir(dir):
        dir_name = os.path.basename(dir)
        for file in os.listdir(dir):
            image_path = os.path.join(dir,file)
            # to convert the image path, so it will be able to correctly work in both windows and linux environment
            image_path = image_path.replace("\\","/")
            image_locations.append((dir_name,image_path))


images = [image_location[1] for image_location in image_locations if "mask" not in image_location[1]]
masks = [image_location[1] for image_location in image_locations if "mask" in image_location[1]]

# to be able to have the images that belong to the same type of mri scan shown in a consecutive order
images = sorted(images, key = lambda x: int(x.split("_")[-1].split(".")[0]))
masks = sorted(masks, key = lambda x: int(x.split("_")[-2]))

# to check if the patient has tumor based on the maximum pixel value
def check_patient_diagnosis(mask_path):
    value = np.max(cv.imread(mask_path))
    if value > 0:
        return 1 # tumor has been detected
    return 0 # tumor hasn't been detected

brain_data = []
for i in range(len(images)):
    brain_data.append([images[i],masks[i],check_patient_diagnosis(masks[i])])

counter = 0
figure, axes = plt.subplots(10,3,figsize=(20,42))
for i in range(len(brain_data)):
    if counter == 10:
        break
    if brain_data[i][2] == 1:
        img = cv.imread(brain_data[i][0])
        mask = cv.imread(brain_data[i][1])

        gray_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, binary_mask = cv.threshold(gray_mask, 1, 255, cv.THRESH_BINARY)

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask_rgb = cv.cvtColor(mask, cv.COLOR_BGR2RGB)

        axes[counter][0].title.set_text("Brain MRI")
        axes[counter][0].imshow(img_rgb)

        axes[counter][1].title.set_text("Mask")
        axes[counter][1].imshow(mask_rgb, cmap='gray')

        img_with_red_mask = img_rgb.copy()
        img_with_red_mask[binary_mask > 0] = [255, 0, 0]

        axes[counter][2].title.set_text("MRI with mask")
        axes[counter][2].imshow(img_with_red_mask)

        counter += 1

plt.show()


# to train the neural network only on images that contain tumor
brain_data_mask = [data for data in brain_data if data[2] == 1]

# comment out the line before so you can work with the whole dataset and uncomment out the following lines

# i seperated the set into two subsets, one which contains only masks that contain tumor and the other set has masks that don't contain tumor
# by splitting the data into two subsets i am making sure that the train, validation and test sets will have masks that contain and don't contain tumor

# brain_data0 = [data for data in brain_data if data[2] == 0]
# brain_data1 = [data for data in brain_data if data[2] == 1]
#
# train_set0, validation_set0 = train_test_split(brain_data0,test_size=0.15)
# test_set0, validation_set0 = train_test_split(validation_set0, test_size=0.5)
#
# train_set1, validation_set1 = train_test_split(brain_data1,test_size=0.15)
# test_set1, validation_set1 = train_test_split(validation_set1, test_size=0.5)
#
# train_set = train_set1 + train_set0
# validation_set = validation_set1 + validation_set0
# test_set = test_set1 + test_set0

# to convert the value of the mask to string format to be able to use categorical mode in flow_from_dataframe

train_set, validation_set = train_test_split(brain_data_mask,test_size=0.15)
test_set, validation_set = train_test_split(validation_set,test_size=0.5)

print(f"The size of the train set is: {len(train_set)}\nThe size of the validation set is: {len(validation_set)}\nThe size of the test set is: {len(test_set)}")

train_images = [t[0] for t in train_set]
train_masks = [t[1] for t in train_set]

validation_images = [v[0] for v in validation_set]
validation_masks = [v[1] for v in validation_set]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, mask, image_dir='./', batch_size=16, img_h=256, img_w=256, shuffle=True):
        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    # to get the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.ids)) / self.batch_size)

    # to generate a batch of data
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        list_ids = [self.ids[i] for i in indexes]

        list_mask = [self.mask[i] for i in indexes]

        x, y = self.__data_generation(list_ids, list_mask)

        return x, y

    # to update the indices after each epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    # to generate data corresponding to the indexes in a given batch of images
    def __data_generation(self, list_ids, list_mask):
        x = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        for i in range(len(list_ids)):
            img_path = str(list_ids[i])

            mask_path = str(list_mask[i])

            img = io.imread(img_path)
            img = cv.resize(img, (self.img_h, self.img_w))
            img = np.array(img, dtype=np.float64)
            img -= img.mean()
            img /= (img.std() + 1e-8)

            mask = io.imread(mask_path)
            mask = cv.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64)
            mask -= mask.mean()
            mask /= (mask.std() + 1e-8)

            x[i,] = img

            y[i,] = np.expand_dims(mask, axis=2)

        y = (y > 0).astype(int)

        return x, y

train_data = DataGenerator(train_images,train_masks)
validation_data = DataGenerator(validation_images,validation_masks)


# to define the resblock function
def resblock(x, f):
    x_copy = x  # to make a copy of the input

    # Main path
    x = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Shortcut path
    x_copy = Conv2D(f, kernel_size=(1, 1), kernel_initializer='he_normal')(x_copy)
    x_copy = BatchNormalization()(x_copy)

    # to add the output from the main and shortcut path together
    x = Add()([x, x_copy])
    x = Activation('relu')(x)

    return x

# to define the upsampling of the image
def upsample_concat(x, skip):
    x_upsample = UpSampling2D((2, 2))(x)
    merge = Concatenate()([x_upsample, skip])

    return merge

input_shape = (256,256,3)
input_data = Input(input_shape) # to initiate tensor of input shape

# Stage 1
convolution_layer1 = Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(input_data)
convolution_layer1 = BatchNormalization()(convolution_layer1)
convolution_layer1 = Conv2D(16,3,activation="relu",padding="same",kernel_initializer="he_normal")(convolution_layer1)
convolution_layer1 = BatchNormalization()(convolution_layer1)
pooled_output1 = MaxPool2D((2,2))(convolution_layer1)

# Stage 2
convolution_layer2 = resblock(pooled_output1,32)
pooled_output2 = MaxPool2D((2,2))(convolution_layer2)

# Stage 3
convolution_layer3 = resblock(pooled_output2,64)
pooled_output3 = MaxPool2D((2,2))(convolution_layer3)

# Stage 4
convolution_layer4 = resblock(pooled_output3,128)
pooled_output4 = MaxPool2D((2,2))(convolution_layer4)

# Stage 5 (bottleneck)
convolution_layer5 = resblock(pooled_output4,256)

# Upsample Stage 1
upsample1 = upsample_concat(convolution_layer5,convolution_layer4)
upsample1 = resblock(upsample1,128)

# Upsample Stage 2
upsample2 = upsample_concat(upsample1,convolution_layer3)
upsample2 = resblock(upsample2,64)

# Upsample Stage 3
upsample3 = upsample_concat(upsample2,convolution_layer2)
upsample3 = resblock(upsample3,32)

# Upsample Stage 4
upsample4 = upsample_concat(upsample3,convolution_layer1)
upsample4 = resblock(upsample4,16)

# Final output
output = Conv2D(1,(1,1),kernel_initializer="he_normal", padding="same",activation="sigmoid")(upsample4)

segmentation_model = Model(input_data,output)

segmentation_model.summary()

# to define a custom loss function
# reference link from https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

smooth = 1

def tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast to match y_pred
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

# compling model and callbacks functions
adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon=0.1) # for the first model i set the value of learning_rate to 0.001 and didn't add any value for epsilon
segmentation_model.compile(optimizer = adam,
                  loss = focal_tversky,
                  metrics = [tversky]
                 )
#callbacks

# to stop early if the model is overfitting the training data
early_stopping = EarlyStopping(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=20
                             )
# to save the best model with lower validation loss
check_pointer = ModelCheckpoint(filepath="ResUNet-segmentation-model-weights.keras",
                               verbose=1,
                               save_best_only=True
                              )

# to reduce the learning rate of the model
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              verbose=1,
                              patience=10,
                              min_delta=0.0001,
                              factor=0.2
                             )

history_object = segmentation_model.fit(train_data,
                  epochs = 60,
                  validation_data = validation_data,
                  callbacks = [check_pointer, early_stopping, reduce_learning_rate]
                 )

# saving model achitecture in json file
seg_model_json = segmentation_model.to_json()
with open("ResUNet-segmentation-model.json", "w") as json_file:
    json_file.write(seg_model_json)

print(history_object.history.keys())

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title("Segmentation Model focal tversky loss")
plt.ylabel("Focal tversky loss")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.subplot(1,2,2)
plt.plot(history_object.history['tversky'])
plt.plot(history_object.history['val_tversky'])
plt.title("Segmentation Model tversky score")
plt.ylabel("Tversky accuracy")
plt.xlabel("Epochs")
plt.legend(['train', 'val'])

plt.show()

test_images = [t[0] for t in test_set]
test_masks = [t[1] for t in test_set]

test_data = DataGenerator(test_images,test_masks)

loss, metric = segmentation_model.evaluate(test_data)

print(f"Segmentation tversky is : {metric*100}")


# to predict the masks of the images
def prediction(test_set, model_seg):
    predicted_data = []

    for item in test_set:
        img_path = item[0]
        mask_path = item[1]

        img = cv.imread(img_path)
        img = cv.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        img -= img.mean()
        img /= (img.std() + 1e-8)

        img_input = np.reshape(img, (1, 256, 256, 3))
        predict = model_seg.predict(img_input)

        mask_detected = 1 if predict.round().astype(int).sum() > 0 else 0
        predicted_data.append([img_path, mask_path, predict[0], mask_detected])

    return predicted_data

brain_data_predicted = prediction(test_set, segmentation_model)

counter = 0
figure, axes = plt.subplots(10, 5, figsize=(30, 70))

for i in range(len(brain_data_predicted)):
    if counter == 10:
        break

    img_path = brain_data_predicted[i][0]
    mask_path = brain_data_predicted[i][1]
    predicted_mask = brain_data_predicted[i][2]

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    axes[counter][0].imshow(img)
    axes[counter][0].title.set_text("Brain MRI")

    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    axes[counter][1].imshow(mask, cmap='gray')
    axes[counter][1].title.set_text("Original Mask")

    pred_mask = (predicted_mask.squeeze() > 0.5).astype(np.uint8) * 255
    axes[counter][2].imshow(pred_mask, cmap='gray')
    axes[counter][2].title.set_text("Predicted Mask")

    img_with_original_mask = img.copy()
    img_with_original_mask[mask > 0] = [255, 0, 0]
    axes[counter][3].imshow(img_with_original_mask)
    axes[counter][3].title.set_text("MRI with original mask")

    img_with_pred_mask = img.copy()
    img_with_pred_mask[pred_mask > 0] = [0, 0, 255]
    axes[counter][4].imshow(img_with_pred_mask)
    axes[counter][4].title.set_text("MRI with predicted mask")

    counter += 1

figure.tight_layout()
plt.show()


