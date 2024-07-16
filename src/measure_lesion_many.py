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
import datetime
from data_preprocessing import resize_with_aspect_ratio, load_dataset
from metrics import dice_coef, dice_loss
from post_processing import measure_rois_no_save, measure_rois
from measure_lesion import get_mask


if __name__ == "__main__":


    dataset_path = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23"

    np.random.seed(42)

    tf.random.set_seed(42)
    _, _, (test_x, test_y) = load_dataset(dataset_path)

    """ Prediction and Evaluation """
    measurements = []
    for x, y in tqdm(zip(test_x, test_y), total=2):
        start = datetime.datetime.now()
        mask = get_mask(x)
        measure_rois(x, mask)


