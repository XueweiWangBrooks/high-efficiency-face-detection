from keras.models import load_model
from keras.models import Model
import keras
from keras.utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6

from keras.utils.vis_utils import plot_model
# from keras.layers import ReLU(6.)
#from keras.layers import DepthwiseConv2D
import fire
import sys
import os
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, BatchNormalization
from keras.models import Model, Sequential

def build_model():
    with CustomObjectScope({'relu6': relu6,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        base_model = load_model('mobilenet.h5')
        for layer in base_model.layers:
            layer.trainable = False
        
        print(base_model.summary())

        x = GlobalAveragePooling2D()(base_model.layers[-2].output)
        x = Dropout(1e-3)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(output_dim = 8, activation = 'softmax')(x)
        return Model(inputs=[base_model.input], outputs=[base_model.output, x])

def save_model(model, folder):
    plot_model(model, folder + "/" + "model_plot.png", True)
    model.save(folder + "/model.h5")
    print('model saved.')

def main(folder):
    model = build_model()
    print(model.summary())
    os.makedirs(folder)
    save_model(model, folder)

if __name__ == "__main__":
    fire.Fire(main)	
