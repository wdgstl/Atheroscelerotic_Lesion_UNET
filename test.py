
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#test

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef
from data_preprocessing import load_dataset, resize_with_aspect_ratio
import imageio
from data_preprocessing import create_dir

""" Global parameters """
H = 256
W = 256

def save_results(form, image, mask, y_pred, save_image_path):
    if form == "cat":
        mask = np.expand_dims(mask, axis=-1)
        mask = np.concatenate([mask, mask, mask], axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred * 255
        line = np.ones((H, 10, 3)) * 255
        cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
        cv2.imwrite(save_image_path+ ".jpg", cat_images)
    if form == "pred":
        y_pred = y_pred *255
        y_pred = y_pred.astype(np.uint8)
        imageio.imwrite(save_image_path, y_pred, format = 'TIFF')

def evaluate_model(dataset_path, model):
    np.random.seed(42)
    tf.random.set_seed(42)
    _, _, (test_x, test_y) = load_dataset(dataset_path)
    """ Prediction and Evaluation """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = os.path.basename(x)
        #        print("NAME:",name)

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  ## [H, w, 3]
        image = resize_with_aspect_ratio('image', image, (W, H))  ## [H, w, 3]
        x = image / 255.0  ## [H, w, 3]
        x = np.expand_dims(x, axis=0)  ## [1, H, w, 3]

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = resize_with_aspect_ratio('mask', mask, (W, H))

        """ Prediction """
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = os.path.join(r"C:\Users\wdgst\Data\ShiData\WDG\Atheroscelerotic_Lesion_UNET\results", name)
        save_results("pred", image, mask, y_pred, save_image_path)
        #        print("image", image)
        #        print("mask", mask)
        #        print("pred", y_pred)
        #        print(save_image_path)
        """ Flatten the array """
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"F1: {score[0]:0.5f}")
    print(f"Jaccard: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")

if __name__ == "__main__":
    """ Directory for storing files """
    create_dir("results")

    dataset_path = r"C:\Users\wdgst\Data\ShiData\WDG"

    """ Load the model """
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join(r"C:\Users\wdgst\Data\ShiData\WDG\Atheroscelerotic_Lesion_UNET\files", "model.h5"))

    evaluate_model(dataset_path, model)




