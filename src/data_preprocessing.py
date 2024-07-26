#CREATE A NEW MODEL CLASS FOR ABSTRACTION

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

""" Global parameters """ ""
H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "../data/histology", "*.tif")))
    masks = sorted(glob(os.path.join(path, "../data/masks", "*.tif")))
    measurements = sorted(glob(os.path.join(path, "../data/measurements, *.csv")))
    split_size = int(len(images) * split) #ratio of train:validation:test
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
    train_mes, valid_mes = train_test_split(measurements,test_size=split_size, random_state=42)
    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)
    train_mes, test_mes = train_test_split(measurements,test_size=split_size, random_state=42)
    
    return (train_x, train_y, train_mes), (valid_x, valid_y, valid_mes), (test_x, test_y, test_mes)

#resizes images to (H, W) without losing aspect ratio (uses padding)
def resize_with_aspect_ratio(type, x, size, direction):
    if type == 'mask':
        aspect_ratio = x.shape[1] / x.shape[0]
        new_size = (size)
        if aspect_ratio > 1:
            new_width = new_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = new_size[1]
            new_width = int(new_height * aspect_ratio)
        if direction == 'up':
            resized_image = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        if direction == 'down':
            resized_image = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padded_image = np.zeros((new_size[1], new_size[0]), dtype=np.uint8) * 255
        padding_left = (new_size[0] - new_width) // 2
        padding_top = (new_size[1] - new_height) // 2
        padded_image[padding_top:padding_top + new_height, padding_left:padding_left + new_width] = resized_image
    if type == 'image':
        aspect_ratio = x.shape[1] / x.shape[0]
        new_size = (size)
        if aspect_ratio > 1:
            new_width = new_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = new_size[1]
            new_width = int(new_height * aspect_ratio)
        if direction == 'up':
            resized_image = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        if direction == 'down':
            resized_image = cv2.resize(x, (new_width, new_height), interpolation=cv2.INTER_AREA)
        padded_image = np.ones((new_size[1], new_size[0], 3), dtype=np.uint8) * 255
        padding_left = (new_size[0] - new_width) // 2
        padding_top = (new_size[1] - new_height) // 2
        padded_image[padding_top:padding_top + new_height, padding_left:padding_left + new_width] = resized_image
    return padded_image

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = resize_with_aspect_ratio('image', x, (H, W))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (h, w)
    x = resize_with_aspect_ratio('mask', x, (H, W))  ## (h, w)
    x = x / 255.0  ## (h, w)
    x = x.astype(np.float32)  ## (h, w)
    x = np.expand_dims(x, axis=-1)  ## (h, w, 1)
    return x

#reads both image and mask for preprocessing
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32]) #? --> numpy function?
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

#creates a tensorflow dataset using batchs --> tensor_slices
def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset
