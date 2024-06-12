import os
import cv2
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
            os.path.join(r"files", "model.h5"))

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  ## [H, w, 3]


    image = resize_with_aspect_ratio('image', image, (256, 256), 'down')  ## [H, w, 3]\
    image_rgb = image[:, :, ::-1]

    image_number = os.path.basename(image_path)
    # Save image locally
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig(f"results/histology_{image_number}.png")
    plt.close()
    print(f"Histology Saved")

    x = image / 255.0  ## [H, w, 3]
    x = np.expand_dims(x, axis=0)  ## [1, H, w, 3]

    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)


    # Save mask locally
    plt.imshow(y_pred, cmap='gray')
    plt.axis('off')
    plt.savefig(f'results/mask_{image_number}.png')
    print("------------------------")
    print(f"Segmentation Mask Saved")

    return y_pred

if __name__ == "__main__":

    image_path = input(r"Enter path to Histology Image: ")
    mask = get_mask(image_path)
    measure_rois(image_path, mask)


