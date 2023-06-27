
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet import build_unet
from metrics import dice_loss, dice_coef

""" Global parameters """
H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "HistogramFinal", "*.tif")))
    masks = sorted(glob(os.path.join(path, "SegmentedFinal/masks", "*.tif")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def resize_with_aspect_ratio(type, x, size):
  if type == 'mask':
    aspect_ratio = x.shape[1] / x.shape[0]
    new_size = (size)

    if aspect_ratio > 1:
      new_width = new_size[0]
      new_height = int(new_width / aspect_ratio)
    else:
      new_height = new_size[1]
      new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(x, (new_width, new_height))

    padded_image = np.zeros((new_size[1], new_size[0]), dtype=np.uint8) * 255


    padding_left = (new_size[0] - new_width) // 2
    padding_top = (new_size[1] - new_height) // 2

    padded_image[padding_top:padding_top+new_height, padding_left:padding_left+new_width] = resized_image

  if type == 'image':
    aspect_ratio = x.shape[1] / x.shape[0]
    new_size = (size)

    if aspect_ratio > 1:
      new_width = new_size[0]
      new_height = int(new_width / aspect_ratio)
    else:
      new_height = new_size[1]
      new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(x, (new_width, new_height))

    padded_image = np.ones((new_size[1], new_size[0], 3), dtype=np.uint8) * 255


    padding_left = (new_size[0] - new_width) // 2
    padding_top = (new_size[1] - new_height) // 2

    padded_image[padding_top:padding_top+new_height, padding_left:padding_left+new_width] = resized_image
  return padded_image


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize_with_aspect_ratio('image', x, (512, 512))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (h, w)
    x = cv2.resize_with_aspect_ratio('mask', x, (512, 512))   ## (h, w)
    x = x / 255.0               ## (h, w)
    x = x.astype(np.float32)    ## (h, w)
    x = np.expand_dims(x, axis=-1)## (h, w, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 16
    lr = 1e-4
    num_epochs = 500
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    dataset_path = r"C:\Users\wdgst\Data\ShiData\WDG"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Model """
    model = build_unet((H, W, 3))
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
