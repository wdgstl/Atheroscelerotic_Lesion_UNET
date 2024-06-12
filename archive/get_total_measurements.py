#go through dir of test images
import pandas as pd
#run model on test hists - get the total area measurements

#grab the total area measurements from segs

#add to db
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import CustomObjectScope
import shutil
from data_preprocessing import resize_with_aspect_ratio, load_dataset
from metrics import dice_coef, dice_loss
from post_processing import measure_rois_no_save


def get_mask(image_path):
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(
            os.path.join(r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23\files", "model.h5"))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  ## [H, w, 3]

    image = resize_with_aspect_ratio('image', image, (256, 256), 'down')  ## [H, w, 3]\

    x = image / 255.0  ## [H, w, 3]
    x = np.expand_dims(x, axis=0)  ## [1, H, w, 3]

    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)

    return y_pred

if __name__ == "__main__":

    csv_path = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23\SegmentedFinal\measurements"
    csv_path_new = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23\SegmentedFinal\measurements\test"

    csv_list = [f for f in os.listdir(csv_path) if os.path.isfile(os.path.join(csv_path, f))]

    dataset_path = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23"

    np.random.seed(42)

    tf.random.set_seed(42)
    _, _, (test_x, test_y) = load_dataset(dataset_path)

    """ Prediction and Evaluation """
    tot = 0
    for x, y in tqdm(zip(test_x, test_y), total=2):
        #os get basename of x
        name = os.path.basename(x)
        name = name.replace('.tif', '')
        print(f'name: {name}')
        #if something in csv_path starts with basename - add to new dir
        for file in csv_list:
            #print(f'file: {file}')
            if name in file:
                tot = tot + 1
                src_path = os.path.join(csv_path, file)
                dest_path = os.path.join(csv_path_new, file)
                shutil.copy(src_path, dest_path)
                print(f'copied {name}')

    print(tot)