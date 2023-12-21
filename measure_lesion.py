#preprocess hist image (resize w aspect ratio)
import os

import cv2
import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.utils import CustomObjectScope

from data_preprocessing import resize_with_aspect_ratio
from metrics import dice_coef, dice_loss
from post_processing import measure_rois



def get_mask(image_path):
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(
            os.path.join(r"C:\Users\wdgst\Data\ShiData\WDG\Atheroscelerotic_Lesion_UNET\files", "model.h5"))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  ## [H, w, 3]



    image = resize_with_aspect_ratio('image', image, (256, 256), 'down')  ## [H, w, 3]
    image_rgb = image[:, :, ::-1]

    # Then display the image
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    x = image / 255.0  ## [H, w, 3]
    x = np.expand_dims(x, axis=0)  ## [1, H, w, 3]

    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)


    plt.imshow(y_pred, cmap='gray')
    plt.axis('off')
    plt.show()

    return y_pred

if __name__ == "__main__":

    image_path = r"C:\Users\wdgst\Data\ShiData\WDG\HistogramFinal\8454-41 3-7 4X.tif"
    mask = get_mask(image_path)
    measure_rois(mask)

