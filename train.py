import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
import fire
import glob
import importlib
import os
import sys
from keras.preprocessing.image import ImageDataGenerator


def train(model_name, model, train_datagen, validation_datagen, optimizer,
          epochs):

    model.compile(optimizer=optimizer,
                  metrics=['accuracy'],
                  loss=['categorical_crossentropy'])
                  
    print(model.summary())


    log_folder = os.path.join('log', '{}'.format(model_name))
    tensorboard = TensorBoard(log_dir=log_folder,
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)
    checkpoint_folder = os.path.join('checkpoints', '{}'.format(model_name))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    filepath = os.path.join(
        checkpoint_folder,
        'weights-improvement-vacc{val_acc:.4f}-vloss{val_loss:.4f}-e{epoch:04d}.hdf5'
    )
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=2,
                                 save_best_only=True,
                                 mode='min')

    model.fit_generator(generator=train_datagen,
                        validation_data=validation_datagen,
                        epochs=epochs,
                        callbacks=[checkpoint, tensorboard])


def count_image(folder, image_format):
    num = 0
    for dir_name, subdir_list, file_list in os.walk(folder):
        for file in file_list:
            if str('.' + image_format) in file:
                num += 1
    return num


def main(data_path="data",
         batch_size=32,
         epochs=5,
         model_name='test',
         model='test_model/model.h5',
         trainable=False,
         lr = 0.001):
    optimizer=Adam(lr, decay=1e-6)

    num_train = count_image(data_path + "/expression/train", "jpg")
    num_test = count_image(data_path + "/expression/test", "jpg")
    num_val = count_image(data_path + "/expression/validation", "jpg")
    print("images found: train={}, test={}, val={}".format(
        num_train, num_test, num_val))

    datagen = ImageDataGenerator()
    train_gen = datagen.flow_from_directory(directory=data_path +
                                            "/expression/train",
                                            target_size=(128, 128),
                                            class_mode="categorical",
                                            batch_size=batch_size)

    # test_gen = datagen.flow_from_directory(directory=data_path +
    #                                        "/expression/test",
    #                                        target_size=(128, 128),
    #                                        class_mode="categorical",
    #                                        batch_size=batch_size)

    val_gen = datagen.flow_from_directory(directory=data_path +
                                          "/expression/validation",
                                          target_size=(128, 128),
                                          class_mode="categorical",
                                          batch_size=batch_size)

    with CustomObjectScope({
            'relu6': relu6,
            'DepthwiseConv2D': keras.layers.DepthwiseConv2D
    }):
        ks_model = load_model(model)
        if trainable == True:
            base_model = ks_model.get_layer('model_1')
            for layer in base_model.layers:
                layer.trainable = True
    train(model_name, ks_model, train_gen, val_gen, optimizer, epochs)

if __name__ == "__main__":
    fire.Fire(main)