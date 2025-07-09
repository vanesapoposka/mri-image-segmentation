import os
import cv2 as cv
import numpy as np
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.optimizers import Adam
from skimage import io

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def focal_tversky(y_true, y_pred, gamma=0.75):
    pt_1 = tversky(y_true, y_pred)
    return tf.pow((1 - pt_1), gamma)


# to load the three models
model1 = load_model(
    "ResUNet-segmentation-model1-weights.keras",
    custom_objects={
        'focal_tversky': focal_tversky,
        'tversky': tversky
    }
)

model2 = load_model(
    "ResUNet-segmentation-model2-weights.keras",
    custom_objects={
        'focal_tversky': focal_tversky,
        'tversky': tversky
    }
)

model3 = load_model(
    "ResUNet-segmentation-model3-weights.keras",
    custom_objects={
        'focal_tversky': focal_tversky,
        'tversky': tversky
    }
)
# to compile the models
model1.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_tversky,
    metrics=[tversky]
)

model2.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_tversky,
    metrics=[tversky]
)

model3.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_tversky,
    metrics=[tversky]
)

def prediction(test_set, model_seg):
    predicted_data = []

    for item in test_set:
        img_path = item[0]
        mask_path = item[1]

        # to read and process image
        img = cv.imread(img_path)
        img = cv.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        # to normalise the image
        img -= img.mean()
        img /= (img.std() + 1e-8)

        # to reshape and predict
        img_input = np.reshape(img, (1, 256, 256, 3))
        predict = model_seg.predict(img_input)

        # to determine if mask has been detected
        mask_detected = 1 if predict.round().astype(int).sum() > 0 else 0
        predicted_data.append([img_path, mask_path, predict[0], mask_detected])

    return predicted_data


import matplotlib.pyplot as plt

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

# to be able to have the images that belong to the same type of mri scan to be shown in a consecutive order
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

# to train the neural network only on images that contain tumor
brain_data_mask = [data for data in brain_data if data[2] == 1]

# to convert the value of the mask to string format to be able to use categorical mode in flow_from_dataframe
brain_data_train = [[img,mask,str(val)] for img,mask,val in brain_data]

train, test = train_test_split(brain_data_train,test_size=0.15, shuffle=False)

train_set, validation_set = train_test_split(brain_data_mask,test_size=0.15, shuffle=False)
test_set, validation_set = train_test_split(validation_set,test_size=0.5, shuffle=False)

test_images = [t[0] for t in test_set]
test_masks = [t[1] for t in test_set]


brain_data1_predicted = prediction(test_set, model1)
brain_data2_predicted = prediction(test_set, model2)
brain_data3_predicted = prediction(test_set, model3)

counter = 0
figure, axes = plt.subplots(10, 5, figsize=(30, 70))

import random

# to get 10 random images from the set
random_idx = []
for i in range(len(brain_data1_predicted)):
    if len(random_idx) == 10:
        break
    random_num = random.randint(0,len(brain_data1_predicted))
    if random_num not in random_idx:
        random_idx.append(random_num)

for i in random_idx:
    if counter == 10:
        break

    img_path = brain_data1_predicted[i][0]
    mask_path = brain_data1_predicted[i][1]
    predicted_mask1 = brain_data1_predicted[i][2]
    predicted_mask2 = brain_data2_predicted[i][2]
    predicted_mask3 = brain_data3_predicted[i][2]

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    axes[counter][0].imshow(img)
    axes[counter][0].title.set_text("Brain MRI")

    mask = cv.imread(mask_path)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    pred_mask1 = (predicted_mask1.squeeze() > 0.5).astype(np.uint8) * 255
    pred_mask2 = (predicted_mask2.squeeze() > 0.5).astype(np.uint8) * 255
    pred_mask3 = (predicted_mask3.squeeze() > 0.5).astype(np.uint8) * 255

    img_with_original_mask = img.copy()
    img_with_original_mask[mask > 0] = [255, 0, 0]
    axes[counter][1].imshow(img_with_original_mask)
    axes[counter][1].title.set_text("MRI with original mask")

    img_with_pred_mask1 = img.copy()
    img_with_pred_mask1[pred_mask1 > 0] = [0, 0, 255]
    axes[counter][2].imshow(img_with_pred_mask1)
    axes[counter][2].title.set_text("MRI with predicted mask from Model 1")

    img_with_pred_mask2 = img.copy()
    img_with_pred_mask2[pred_mask2 > 0] = [0, 255, 0]
    axes[counter][3].imshow(img_with_pred_mask2)
    axes[counter][3].title.set_text("MRI with predicted mask from Model 2")

    img_with_pred_mask3 = img.copy()
    img_with_pred_mask3[pred_mask3 > 0] = [0, 255, 255]
    axes[counter][4].imshow(img_with_pred_mask3)
    axes[counter][4].title.set_text("MRI with predicted mask from Model 3")

    counter += 1

figure.tight_layout()
plt.show()

