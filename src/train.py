import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from unet import build_unet
from metrics import dice_loss, dice_coef
from data_preprocessing import create_dir, load_dataset, tf_dataset

""" Global parameters """
H = 256
W = 256

def print_dataset_lengths(train_x, train_y, valid_x, valid_y, test_x, test_y):
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")


def train_unet(dataset_path,
        epochs, batch_size, lr):
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Num GPUs Used for Training: ", len(tf.config.list_physical_devices('GPU')))

    create_dir("files")

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print_dataset_lengths(train_x, train_y, valid_x, valid_y, test_x, test_y)

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)


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


if __name__ == "__main__":

    """ Hyperparameters """
    batch_size = 32
    lr = 0.001
    num_epochs = 500
    
    # The code snippet `current_dir = os.path.dirname(__file__)` is used to get the directory of the
    # current Python script file. It retrieves the directory path where the current Python script is
    # located.
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    model_path = os.path.join(parent_dir, "files", "model.keras")
    csv_path = os.path.join(parent_dir, "files", "log.csv")

    """ Dataset """
    
    dataset_path = os.path.join(parent_dir, "data")
     
    train_unet(dataset_path, epochs = num_epochs, batch_size = batch_size, lr = lr)





